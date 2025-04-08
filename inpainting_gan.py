import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from monai.networks.nets import SwinUNETR
from tqdm import tqdm
import random
import glob
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import argparse
from torchvision.utils import make_grid
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from skimage.morphology import binary_dilation
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import SimpleITK as sitk
from glob import glob
import re
import warnings
import argparse
import random
import time
import logging
import sys
import scipy
import nibabel as nib
from scipy.ndimage import binary_dilation, gaussian_filter


class LesionInpaintingDataset(Dataset):
    """Dataset for training a GAN to inpaint synthetic lesions in ADC maps.
    
    Args:
        adc_dir (str): Directory containing ADC maps.
        label_dir (str): Directory containing real lesion label masks.
        synthetic_lesions_dir (str): Directory containing synthetic lesions.
        split (str): Either 'train' or 'val'.
        adc_mean_atlas_path (str, optional): Path to the mean ADC atlas. Default is None.
        adc_std_atlas_path (str, optional): Path to the standard deviation ADC atlas. Default is None.
        dilation_radius (int, optional): Radius for dilating the lesion mask. Default is 3.
        smoothing_iterations (int, optional): Number of iterations for Gaussian smoothing. Default is 5.
    """
    def __init__(
        self, 
        adc_dir, 
        label_dir, 
        synthetic_lesions_dir, 
        split='train', 
        adc_mean_atlas_path=None, 
        adc_std_atlas_path=None,
        dilation_radius=3,
        smoothing_iterations=5
    ):
        self.adc_dir = adc_dir
        self.label_dir = label_dir
        self.synthetic_lesions_dir = synthetic_lesions_dir
        self.split = split
        self.adc_mean_atlas_path = adc_mean_atlas_path
        self.adc_std_atlas_path = adc_std_atlas_path
        self.dilation_radius = dilation_radius
        self.smoothing_iterations = smoothing_iterations
        
        # Load atlases if provided
        self.adc_mean_atlas = None
        self.adc_std_atlas = None
        if adc_mean_atlas_path and os.path.exists(adc_mean_atlas_path):
            try:
                self.adc_mean_atlas = nib.load(adc_mean_atlas_path).get_fdata()
                print(f"Loaded mean ADC atlas from {adc_mean_atlas_path}, shape: {self.adc_mean_atlas.shape}")
            except Exception as e:
                print(f"Failed to load mean ADC atlas from {adc_mean_atlas_path}: {str(e)}")
        else:
            print(f"WARNING: Mean ADC atlas not found at {adc_mean_atlas_path}. Will use smooth inpainting method.")
        
        if adc_std_atlas_path and os.path.exists(adc_std_atlas_path):
            try:
                self.adc_std_atlas = nib.load(adc_std_atlas_path).get_fdata()
                print(f"Loaded std ADC atlas from {adc_std_atlas_path}, shape: {self.adc_std_atlas.shape}")
            except Exception as e:
                print(f"Failed to load std ADC atlas from {adc_std_atlas_path}: {str(e)}")
                
        # Extract patient IDs from ADC files
        adc_files = sorted(glob.glob(os.path.join(adc_dir, "*.nii.gz")))
        patient_ids = [os.path.basename(f).split("_")[0] for f in adc_files]
        patient_ids = list(set(patient_ids))  # Remove duplicates
        
        # Filter patients based on whether they have the necessary files
        valid_patient_ids = []
        for patient_id in patient_ids:
            label_files = glob.glob(os.path.join(label_dir, f"{patient_id}_*.nii.gz"))
            synthetic_lesion_dir = os.path.join(synthetic_lesions_dir, patient_id)
            
            if label_files and os.path.exists(synthetic_lesion_dir):
                valid_patient_ids.append(patient_id)
        
        # Split patients into train and validation sets
        random.seed(42)  # For reproducibility
        random.shuffle(valid_patient_ids)
        train_ratio = 0.8
        train_size = int(len(valid_patient_ids) * train_ratio)
        
        if split == 'train':
            self.patient_ids = valid_patient_ids[:train_size]
        else:  # val
            self.patient_ids = valid_patient_ids[train_size:]
        
        print(f"Number of patients in {split} set: {len(self.patient_ids)}")
        
        # Create sample pairs for each patient
        self.sample_pairs = []
        for patient_id in self.patient_ids:
            adc_files = sorted(glob.glob(os.path.join(adc_dir, f"{patient_id}_*.nii.gz")))
            label_files = sorted(glob.glob(os.path.join(label_dir, f"{patient_id}_*.nii.gz")))
            synthetic_lesion_dir = os.path.join(synthetic_lesions_dir, patient_id)
            synthetic_lesion_files = sorted(glob.glob(os.path.join(synthetic_lesion_dir, "*.nii.gz")))
            
            for adc_path, label_path in zip(adc_files, label_files):
                for synthetic_lesion_path in synthetic_lesion_files:
                    self.sample_pairs.append({
                        "adc_path": adc_path,
                        "label_path": label_path,
                        "synthetic_lesion_path": synthetic_lesion_path
                    })
        
        print(f"Number of training samples in {split} set: {len(self.sample_pairs)}")
    
    def register_atlas_to_subject(self, subject_img, subject_array, label_array):
        """
        Registruje atlas k danému subjektu, vyhýbá se oblasti s lézemi.
        
        Args:
            subject_img: SimpleITK obraz subjektu
            subject_array: numpy pole subjektu
            label_array: maska léze
            
        Returns:
            registered_atlas: registrovaný atlas
            registered_std_atlas: registrovaný atlas směrodatných odchylek
        """
        if not self.has_atlas:
            print("Atlas není k dispozici pro registraci.")
            return None, None
        
        try:
            print("Registruji atlas k subjektu...")
            
            # Kontrola velikosti atlasu a subjektu
            subject_size = subject_array.shape
            atlas_size = self.adc_mean_atlas.shape
            
            print(f"Velikost subjektu: {subject_size}, velikost atlasu: {atlas_size}")
            
            # Ověříme, že atlas není prázdný nebo obsahuje NaN hodnoty
            if np.isnan(self.adc_mean_atlas).any():
                print("VAROVÁNÍ: Atlas obsahuje NaN hodnoty!")
                self.adc_mean_atlas = np.nan_to_num(self.adc_mean_atlas)
            
            # Zkusíme použít SimpleITK pro registraci
            try:
                print("Používám SimpleITK pro registraci...")
                
                # Vytvořit masku, kde není léze (používáme jako oblast zájmu)
                valid_mask = np.logical_not(label_array).astype(np.float32)
                
                # Konvertovat masku na SimpleITK obraz
                mask_img = sitk.GetImageFromArray(valid_mask)
                mask_img.CopyInformation(subject_img)
                
                # Nastavit orientaci atlasu stejně jako subjekt
                atlas_img = sitk.GetImageFromArray(self.adc_mean_atlas)
                
                # Zkopírovat metadata z předmětu do atlasu
                atlas_img.SetSpacing(subject_img.GetSpacing())
                atlas_img.SetOrigin(subject_img.GetOrigin())
                atlas_img.SetDirection(subject_img.GetDirection())
                
                # Nejprve musíme provést resample atlasu na stejnou velikost jako subjekt
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(subject_img)
                resampler.SetInterpolator(sitk.sitkLinear)
                resampler.SetDefaultPixelValue(0)
                atlas_img_resampled = resampler.Execute(atlas_img)
                
                print("Resampled atlas to subject dimensions")
                
                # Provést registraci pomocí SimpleElastix
                elastixImageFilter = sitk.ElastixImageFilter()
                elastixImageFilter.SetFixedImage(subject_img)
                elastixImageFilter.SetMovingImage(atlas_img_resampled)
                elastixImageFilter.SetFixedMask(mask_img)  # Použít masku k vynechání lézí
                
                # Nastavení parametrů registrace
                parameterMap = sitk.GetDefaultParameterMap('affine')
                elastixImageFilter.SetParameterMap(parameterMap)
                
                # Provést registraci
                try:
                    elastixImageFilter.Execute()
                    registered_atlas_img = elastixImageFilter.GetResultImage()
                    
                    # Konvertovat zpět na numpy pole
                    registered_atlas = sitk.GetArrayFromImage(registered_atlas_img)
                    
                    # Pokud máme atlas směrodatných odchylek, registrujeme i ten
                    if self.adc_std_atlas is not None:
                        std_atlas_img = sitk.GetImageFromArray(self.adc_std_atlas)
                        std_atlas_img.CopyInformation(atlas_img)
                        
                        # Nastavit transformaci z předchozí registrace
                        transformixImageFilter = sitk.TransformixImageFilter()
                        transformixImageFilter.SetMovingImage(std_atlas_img)
                        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
                        transformixImageFilter.Execute()
                        
                        registered_std_atlas_img = transformixImageFilter.GetResultImage()
                        registered_std_atlas = sitk.GetArrayFromImage(registered_std_atlas_img)
                    else:
                        registered_std_atlas = None
                        
                    print("SimpleITK registrace úspěšná")
                    
                    # Zkontrolujeme výsledek registrace na díry (black spots)
                    registered_atlas = self.fix_registration_holes(registered_atlas, subject_array)
                    
                    return registered_atlas, registered_std_atlas
                
                except Exception as e:
                    print(f"SimpleITK registrace selhala: {e}")
                    # Pokračujeme k záložní metodě
            
            except Exception as e:
                print(f"Chyba při použití SimpleITK pro registraci: {e}")
            
            # Záložní metoda: použít scipy pro jednoduchou registraci
            print("Používám scipy resample jako záložní metodu registrace...")
            
            # Vytvořit nový atlas s velikostí subjektu
            registered_atlas = np.zeros_like(subject_array)
            
            # Vypočítat faktory zoomu pro každou dimenzi
            zoom_factors = (subject_size[0] / atlas_size[0],
                           subject_size[1] / atlas_size[1],
                           subject_size[2] / atlas_size[2])
            
            # Použít scipy zoom pro resize
            from scipy import ndimage
            resized_atlas = ndimage.zoom(self.adc_mean_atlas, zoom_factors, order=1)
            
            # Oříznout nebo doplnit, pokud velikosti nejsou přesně stejné
            if resized_atlas.shape != subject_array.shape:
                print(f"Po resample se velikosti neshodují: subjekt {subject_array.shape} vs atlas {resized_atlas.shape}")
                
                # Vytvořit nové pole správné velikosti
                registered_atlas = np.zeros_like(subject_array)
                
                # Vypočítat minimální společné rozměry
                min_shape = [min(subject_array.shape[i], resized_atlas.shape[i]) for i in range(3)]
                
                # Zkopírovat dostupná data
                registered_atlas[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                    resized_atlas[:min_shape[0], :min_shape[1], :min_shape[2]]
            else:
                registered_atlas = resized_atlas
            
            # Registrovat standardní odchylku, pokud je k dispozici
            if self.adc_std_atlas is not None:
                resized_std_atlas = ndimage.zoom(self.adc_std_atlas, zoom_factors, order=1)
                
                if resized_std_atlas.shape != subject_array.shape:
                    registered_std_atlas = np.zeros_like(subject_array)
                    min_shape = [min(subject_array.shape[i], resized_std_atlas.shape[i]) for i in range(3)]
                    registered_std_atlas[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                        resized_std_atlas[:min_shape[0], :min_shape[1], :min_shape[2]]
                else:
                    registered_std_atlas = resized_std_atlas
            else:
                registered_std_atlas = None
            
            print("Scipy registrace dokončena")
            
            # Zkontrolujeme výsledek registrace na díry (black spots)
            registered_atlas = self.fix_registration_holes(registered_atlas, subject_array)
            
            return registered_atlas, registered_std_atlas
            
        except Exception as e:
            print(f"Registrace atlasu selhala: {e}")
            return None, None
    
    def fix_registration_holes(self, registered_atlas, subject_array):
        """
        Opraví díry v registrovaném atlasu pomocí interpolace a prahování.
        
        Args:
            registered_atlas: Registrovaný atlas s možnými dírami
            subject_array: Původní obraz subjektu pro referenci
            
        Returns:
            Opravený registrovaný atlas bez děr
        """
        print("Kontroluji a opravuji díry v registrovaném atlasu...")
        
        # Vytvořit kopii atlasu
        fixed_atlas = registered_atlas.copy()
        
        # Identifikovat díry (příliš nízké hodnoty v oblastech, kde subjekt má signál)
        # Předpokládáme, že díry mají hodnotu 0 nebo blízko 0
        # a objevují se tam, kde subjekt má nenulový signál
        subject_mask = subject_array > 0.01
        hole_mask = (fixed_atlas < 0.01) & subject_mask
        
        if np.sum(hole_mask) > 0:
            print(f"Nalezeno {np.sum(hole_mask)} voxelů s dírami")
            
            # Použít morfologické operace k identifikaci malých děr
            # Dilatace a následná eroze může vyplnit malé díry
            filled_mask = ndimage.binary_closing(~hole_mask, structure=np.ones((3,3,3)), iterations=2)
            
            # Interpolovat hodnoty pomocí okolí
            # Způsob 1: Použít medián filtr pro vyplnění děr
            temp_atlas = fixed_atlas.copy()
            temp_atlas[hole_mask] = np.nan  # Označit díry jako NaN
            
            # Vytvořit masku pro median filtr (pouze neNaN hodnoty)
            median_filtered = ndimage.median_filter(np.nan_to_num(temp_atlas), size=5)
            
            # Pouze nahradit hodnoty v dírách
            fixed_atlas[hole_mask] = median_filtered[hole_mask]
            
            # Způsob 2: Pro větší díry použít vzdálenostně váženou interpolaci
            from scipy.interpolate import griddata
            
            # Pro každý řez zvlášť (pro efektivitu)
            for z in range(fixed_atlas.shape[0]):
                slice_holes = hole_mask[z]
                if np.sum(slice_holes) > 0:
                    # Získat souřadnice známých bodů a jejich hodnoty
                    y_known, x_known = np.where(~slice_holes)
                    values_known = fixed_atlas[z, y_known, x_known]
                    
                    # Souřadnice bodů k interpolaci (díry)
                    y_holes, x_holes = np.where(slice_holes)
                    
                    if len(y_known) > 0 and len(y_holes) > 0:
                        # Připravit body pro interpolaci
                        points = np.column_stack((y_known, x_known))
                        holes = np.column_stack((y_holes, x_holes))
                        
                        # Interpolovat hodnoty
                        try:
                            interpolated = griddata(points, values_known, holes, method='linear', fill_value=np.mean(values_known))
                            fixed_atlas[z, y_holes, x_holes] = interpolated
                        except Exception as e:
                            print(f"Interpolace selhala pro řez {z}: {e}")
            
            print("Díry v atlasu byly opraveny")
        else:
            print("Žádné díry v atlasu nebyly nalezeny")
        
        return fixed_atlas
    
    def __len__(self):
        return len(self.sample_pairs)
    
    def __getitem__(self, idx):
        """Get a sample pair (input, target) for training.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (input, target) tensors for training.
                input: [2, D, H, W] tensor with healthy brain and synthetic lesion mask
                target: [1, D, H, W] tensor with brain containing lesion
        """
        sample = self.samples[idx]
        patient_id = sample['patient_id']
        syn_lesion_idx = sample['syn_lesion_idx']
        
        try:
            # Load ADC map
            adc_path = os.path.join(self.adc_dir, f"{patient_id}_adc.nii.gz")
            adc_nib = nib.load(adc_path)
            adc_array = adc_nib.get_fdata()
            
            # Normalize ADC map to [0, 1]
            adc_min = np.min(adc_array)
            adc_max = np.max(adc_array)
            if adc_max > adc_min:
                adc_array_norm = (adc_array - adc_min) / (adc_max - adc_min)
            else:
                adc_array_norm = adc_array.copy()
            
            # Load real lesion mask
            label_path = os.path.join(self.label_dir, f"{patient_id}_label.nii.gz")
            label_array = nib.load(label_path).get_fdata() > 0.5
            
            # Create a "healthy brain" by replacing lesions using our smooth inpainting function
            healthy_brain = create_smooth_healthy_brain(
                adc_array_norm, 
                label_array,
                dilation_radius=self.dilation_radius,
                smoothing_iterations=self.smoothing_iterations
            )
            
            # Load synthetic lesion
            syn_lesion_path = os.path.join(self.synthetic_lesions_dir, patient_id, f"syn_lesion_{syn_lesion_idx}.nii.gz")
            syn_lesion_array = nib.load(syn_lesion_path).get_fdata() > 0.5
            
            # Calculate overlap between synthetic and real lesions
            overlap = np.sum(syn_lesion_array & label_array) / np.sum(syn_lesion_array) if np.sum(syn_lesion_array) > 0 else 0
            
            # If synthetic lesion overlaps significantly with real lesion,
            # use the original brain with real lesion as target.
            # Otherwise, create a target with only the synthetic lesion.
            if overlap > 0.5:
                target = adc_array_norm.copy()
            else:
                # For synthetic lesions that don't overlap with real ones,
                # insert a lesion-like intensity into the healthy brain
                target = healthy_brain.copy()
                
                # Only modify the synthetic lesion area
                if np.any(syn_lesion_array):
                    # Calculate average intensity in real lesions for reference
                    if np.any(label_array):
                        lesion_intensity = np.mean(adc_array_norm[label_array])
                    else:
                        # If no real lesions, use a typical lesion intensity
                        # (ADC lesions are typically hyperintense, around 0.7-0.9 in normalized scale)
                        lesion_intensity = 0.8
                    
                    # Insert the lesion with some random variation
                    random_variation = np.random.normal(0, 0.05, size=np.count_nonzero(syn_lesion_array))
                    target[syn_lesion_array] = np.clip(lesion_intensity + random_variation, 0, 1)
            
            # Convert to PyTorch tensors
            healthy_brain_tensor = torch.from_numpy(healthy_brain).float().unsqueeze(0)
            syn_lesion_tensor = torch.from_numpy(syn_lesion_array.astype(np.float32)).unsqueeze(0)
            target_tensor = torch.from_numpy(target).float().unsqueeze(0)
            
            # Combine healthy brain and synthetic lesion mask as input
            input_tensor = torch.cat([healthy_brain_tensor, syn_lesion_tensor], dim=0)
            
            return input_tensor, target_tensor
            
        except Exception as e:
            print(f"Error processing sample {patient_id}, syn_lesion_{syn_lesion_idx}: {str(e)}")
            # Return a dummy sample pair as fallback
            dummy_shape = (2, 128, 128, 128)
            dummy_target_shape = (1, 128, 128, 128)
            return torch.zeros(dummy_shape, dtype=torch.float32), torch.zeros(dummy_target_shape, dtype=torch.float32)

# Přidat pomocnou funkci pro vytvoření základního zdravého mozku
def create_basic_healthy_brain(self, adc_normalized, label_array):
    """Vytvoří jednoduchý model zdravého mozku pomocí průměrných hodnot zdravé tkáně."""
    # Rozšířit masku léze pomocí dilace pro lepší zachycení oblasti
    kernel = np.ones((3,3,3))
    dilated_mask = ndimage.binary_dilation(label_array, structure=kernel, iterations=2)
    
    # Vytvořit hranici okolo léze (dilated - original)
    border_mask = dilated_mask & np.logical_not(label_array)
    
    # Vypočítat průměrnou hodnotu zdravé tkáně v okolí léze
    if np.sum(border_mask) > 10:
        healthy_tissue_value = np.mean(adc_normalized[border_mask])
        print(f"Using border healthy tissue value: {healthy_tissue_value:.4f}")
    else:
        # Fallback: použít průměrnou hodnotu všech voxelů mimo lézi
        healthy_tissue_value = np.mean(adc_normalized[np.logical_not(label_array)])
        print(f"Using global healthy tissue value: {healthy_tissue_value:.4f}")
    
    # Vytvořit zdravý mozek nahrazením oblasti léze
    healthy_brain = adc_normalized.copy()
    healthy_brain[label_array] = healthy_tissue_value
    
    return healthy_brain

# Přidat funkci pro vytvoření cílového obrazu se syntetickou lézí
def create_target_with_synthetic_lesion(self, healthy_brain, synthetic_lesion_array, original_adc):
    """Vytvoří cílový obraz s vloženou syntetickou lézí."""
    target = healthy_brain.copy()
    
    # Vypočítat průměrnou intenzitu skutečných lézí
    avg_lesion_intensity = np.mean(original_adc[original_adc < 0.3])  # Předpokládáme, že léze mají nízkou intenzitu
    
    if np.isnan(avg_lesion_intensity) or avg_lesion_intensity < 0.01:
        # Fallback - použít fixní intenzitu
        avg_lesion_intensity = 0.2
        print(f"Using default lesion intensity: {avg_lesion_intensity:.4f}")
    else:
        print(f"Using measured lesion intensity: {avg_lesion_intensity:.4f}")
    
    # Přidat náhodnou variabilitu k intenzitě léze
    lesion_intensity = avg_lesion_intensity * np.random.uniform(0.8, 1.2)
    
    # Přidat syntetickou lézi s realistickou intenzitou
    target[synthetic_lesion_array] = lesion_intensity
    
    return target

# Přidat funkci pro vizualizaci procesu registrace
def create_registration_visualization(adc, atlas, mask, healthy, ratio, output_path):
    """Vytvoří vizualizaci procesu registrace atlasu a tvorby zdravého mozku."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LinearSegmentedColormap
        import os
        
        # Vytvořit adresář pro výstup, pokud neexistuje
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Najít střední řez s nejvíce maskou
        mask_sum_per_slice = np.sum(mask, axis=(1, 2))
        middle_slice = np.argmax(mask_sum_per_slice) if np.max(mask_sum_per_slice) > 0 else mask.shape[0] // 2
        
        # Vytvořit barevnou mapu pro rozdíl
        diff_cmap = LinearSegmentedColormap.from_list('diff', ['blue', 'white', 'red'], N=256)
        
        # Vypočítat rozdíl
        diff = adc - atlas
        diff_norm = np.clip(diff + 0.5, 0, 1)  # Normalizace rozdílu pro vizualizaci
        
        # Vytvořit obrázek
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 4, figure=fig)
        
        # První řada
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(adc[middle_slice], cmap='gray')
        ax1.set_title('Původní ADC')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(atlas[middle_slice], cmap='gray')
        ax2.set_title(f'Registrovaný atlas')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(diff_norm[middle_slice], cmap=diff_cmap)
        ax3.set_title(f'Rozdíl (ADC - Atlas)')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(healthy[middle_slice], cmap='gray')
        ax4.set_title(f'Výsledný "zdravý mozek"')
        ax4.axis('off')
        
        # Druhá řada
        # Zobrazit ADC s maskou
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(adc[middle_slice], cmap='gray')
        masked = np.ma.masked_where(~mask[middle_slice], np.ones_like(adc[middle_slice]))
        ax5.imshow(masked, cmap='autumn', alpha=0.5)
        ax5.set_title('ADC s maskou léze')
        ax5.axis('off')
        
        # Zobrazit atlas s maskou
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(atlas[middle_slice], cmap='gray')
        ax6.imshow(masked, cmap='autumn', alpha=0.5)
        ax6.set_title('Atlas s maskou léze')
        ax6.axis('off')
        
        # Zobrazit hodnoty intenzit podél řádky
        ax7 = fig.add_subplot(gs[1, 2:])
        # Najít řádek s nejvíce maskou
        mask_sum_per_row = np.sum(mask[middle_slice], axis=1)
        middle_row = np.argmax(mask_sum_per_row) if np.max(mask_sum_per_row) > 0 else mask.shape[1] // 2
        
        # Vykreslit profily intenzit
        ax7.plot(adc[middle_slice, middle_row, :], 'b-', label='ADC')
        ax7.plot(atlas[middle_slice, middle_row, :] * ratio, 'g-', label=f'Atlas (scaled by {ratio:.2f})')
        ax7.plot(healthy[middle_slice, middle_row, :], 'r--', label='Zdravý mozek')
        
        # Zvýraznit oblast léze
        mask_indices = np.where(mask[middle_slice, middle_row, :])[0]
        if len(mask_indices) > 0:
            min_idx, max_idx = np.min(mask_indices), np.max(mask_indices)
            ax7.axvspan(min_idx, max_idx, color='gray', alpha=0.3, label='Oblast léze')
        
        ax7.set_title('Intenzity podél řádky')
        ax7.set_xlabel('Pozice (voxel)')
        ax7.set_ylabel('Normalizovaná intenzita')
        ax7.legend()
        ax7.grid(True)
        
        # Statistiky v horní části
        adc_in_mask = adc[mask]
        atlas_in_mask = atlas[mask]
        healthy_in_mask = healthy[mask]
        
        # Výpočet statistik
        adc_mean = np.mean(adc_in_mask) if len(adc_in_mask) > 0 else float('nan')
        atlas_mean = np.mean(atlas_in_mask) if len(atlas_in_mask) > 0 else float('nan')
        healthy_mean = np.mean(healthy_in_mask) if len(healthy_in_mask) > 0 else float('nan')
        diff_mean = np.mean(np.abs(adc_in_mask - atlas_in_mask)) if len(adc_in_mask) > 0 else float('nan')
        
        plt.suptitle(
            f'Registrace atlasu a vytvoření zdravého mozku\n'
            f'Statistiky v oblasti léze - '
            f'ADC: {adc_mean:.4f}, Atlas: {atlas_mean:.4f}, '
            f'Zdravý: {healthy_mean:.4f}, Rozdíl: {diff_mean:.4f}, Poměr: {ratio:.4f}',
            fontsize=12
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Registration visualization saved to {output_path}")
        
    except Exception as e:
        print(f"Error creating registration visualization: {e}")


class PatchGANDiscriminator(nn.Module):
    """
    3D PatchGAN discriminator for lesion inpainting.
    """
    def __init__(self, in_channels=2, ndf=64, n_layers=3):
        """
        Args:
            in_channels: Number of input channels (ZADC + mask)
            ndf: Number of filters in the first conv layer
        """
        super(PatchGANDiscriminator, self).__init__()
        
        sequence = [
            nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, 
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1),
            nn.BatchNorm3d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x, mask):
        """
        Args:
            x: Input image
            mask: Lesion mask to focus attention on specific regions
        """
        # Concatenate image and mask
        x_and_mask = torch.cat([x, mask], dim=1)
        return self.model(x_and_mask)


class LesionInpaintingGAN:
    """
    GAN model for HIE lesion inpainting.
    """
    def __init__(self, 
                 img_shape=(96, 96, 96),
                 device=None):
        """
        Args:
            img_shape: Shape of the input image
            device: Device to run the model on
        """
        self.img_shape = img_shape
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generator - SwinUNETR for high-quality 3D inpainting
        self.generator = SwinUNETR(
            img_size=img_shape,
            in_channels=2,  # Healthy brain + synthetic lesion mask
            out_channels=1,  # Inpainted ZADC
            feature_size=24,
            use_checkpoint=True
        ).to(self.device)
        
        # Discriminator - PatchGAN for assessing realism
        self.discriminator = PatchGANDiscriminator(
            in_channels=2  # ZADC + lesion mask
        ).to(self.device)
        
        # Optimizers
        self.optimizer_G = Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_D = Adam(self.discriminator.parameters(), lr=4e-4, betas=(0.5, 0.999))
        
        # Learning rate schedulers
        self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=200)
        self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=200)
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
    def train_step(self, data):
        """
        Execute one training step.
        
        Args:
            data: Dictionary containing 'input', 'target', 'synthetic_mask'
        
        Returns:
            Dictionary of losses
        """
        # Set models to training mode
        self.generator.train()
        self.discriminator.train()
        
        # Get data
        real_brain = data['target'].to(self.device)
        input_data = data['input'].to(self.device)
        synthetic_mask = data['synthetic_mask'].to(self.device)
        
        # -----------------
        # Train Generator
        # -----------------
        self.optimizer_G.zero_grad()
        
        # Generate inpainted image
        fake_brain = self.generator(input_data)
        
        # Mask predictions from discriminator (only care about lesion area)
        pred_fake = self.discriminator(fake_brain, synthetic_mask)
        
        # Calculate generator losses
        valid = torch.ones_like(pred_fake).to(self.device)
        adv_loss = self.adversarial_loss(pred_fake, valid)
        
        # Pixel-wise loss (only for lesion area)
        mask_expanded = synthetic_mask.expand_as(real_brain)
        l1_loss = self.l1_loss(fake_brain * mask_expanded, real_brain * mask_expanded) * 100
        
        # Context loss (for surrounding area)
        inv_mask = 1 - synthetic_mask
        inv_mask_expanded = inv_mask.expand_as(real_brain)
        context_loss = self.l1_loss(fake_brain * inv_mask_expanded, real_brain * inv_mask_expanded) * 50
        
        # Total generator loss
        g_loss = adv_loss + l1_loss + context_loss
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # -----------------
        # Train Discriminator
        # -----------------
        self.optimizer_D.zero_grad()
        
        # Real brain with real lesions
        pred_real = self.discriminator(real_brain, synthetic_mask)
        real_labels = torch.ones_like(pred_real).to(self.device)
        real_loss = self.adversarial_loss(pred_real, real_labels)
        
        # Generated brain with fake lesions
        pred_fake = self.discriminator(fake_brain.detach(), synthetic_mask)
        fake_labels = torch.zeros_like(pred_fake).to(self.device)
        fake_loss = self.adversarial_loss(pred_fake, fake_labels)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        self.optimizer_D.step()
        
        return {
            'g_loss': g_loss.item(),
            'adv_loss': adv_loss.item(),
            'l1_loss': l1_loss.item(),
            'context_loss': context_loss.item(),
            'd_loss': d_loss.item()
        }
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Dictionary of validation metrics
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {
            'g_loss': 0,
            'adv_loss': 0,
            'l1_loss': 0,
            'context_loss': 0,
            'd_loss': 0
        }
        
        with torch.no_grad():
            for batch in dataloader:
                real_brain = batch['target'].to(self.device)
                input_data = batch['input'].to(self.device)
                synthetic_mask = batch['synthetic_mask'].to(self.device)
                
                # Generate inpainted image
                fake_brain = self.generator(input_data)
                
                # Discriminator predictions
                pred_fake = self.discriminator(fake_brain, synthetic_mask)
                pred_real = self.discriminator(real_brain, synthetic_mask)
                
                # Adversarial loss
                valid = torch.ones_like(pred_fake).to(self.device)
                adv_loss = self.adversarial_loss(pred_fake, valid)
                
                # Pixel-wise loss for lesion area
                mask_expanded = synthetic_mask.expand_as(real_brain)
                l1_loss = self.l1_loss(fake_brain * mask_expanded, real_brain * mask_expanded) * 100
                
                # Context loss for surrounding area
                inv_mask = 1 - synthetic_mask
                inv_mask_expanded = inv_mask.expand_as(real_brain)
                context_loss = self.l1_loss(fake_brain * inv_mask_expanded, real_brain * inv_mask_expanded) * 50
                
                # Total generator loss
                g_loss = adv_loss + l1_loss + context_loss
                
                # Discriminator loss
                real_labels = torch.ones_like(pred_real).to(self.device)
                fake_labels = torch.zeros_like(pred_fake).to(self.device)
                real_loss = self.adversarial_loss(pred_real, real_labels)
                fake_loss = self.adversarial_loss(pred_fake, fake_labels)
                d_loss = (real_loss + fake_loss) / 2
                
                # Accumulate losses
                val_losses['g_loss'] += g_loss.item()
                val_losses['adv_loss'] += adv_loss.item()
                val_losses['l1_loss'] += l1_loss.item()
                val_losses['context_loss'] += context_loss.item()
                val_losses['d_loss'] += d_loss.item()
                
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(dataloader)
            
        return val_losses
    
    def save_models(self, save_dir, epoch):
        """
        Save model checkpoints.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'scheduler_G': self.scheduler_G.state_dict(),
            'scheduler_D': self.scheduler_D.state_dict(),
            'epoch': epoch
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    def load_models(self, checkpoint_path):
        """
        Load model checkpoints.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])
        return checkpoint['epoch']
    
    def inference(self, healthy_brain, synthetic_lesion_mask):
        """
        Perform inference on a single sample.
        
        Args:
            healthy_brain: Healthy brain scan (tensor of shape [1, D, H, W])
            synthetic_lesion_mask: Synthetic lesion mask (tensor of shape [1, D, H, W])
        
        Returns:
            Inpainted brain scan with lesion
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Ensure inputs are on the correct device
            healthy_brain = healthy_brain.to(self.device)
            synthetic_lesion_mask = synthetic_lesion_mask.to(self.device)
            
            # Combine inputs
            combined_input = torch.cat([healthy_brain, synthetic_lesion_mask], dim=1)
            
            # Generate inpainted image
            inpainted_brain = self.generator(combined_input)
            
        return inpainted_brain


def train(adc_dir, label_dir, synthetic_lesions_dir, output_dir,
         adc_mean_atlas_path=None, adc_std_atlas_path=None,
         num_epochs=200, batch_size=4,
         dilation_radius=3, smoothing_iterations=5):
    """
    Train the lesion inpainting GAN.
    
    Args:
        adc_dir (str): Directory containing ADC maps.
        label_dir (str): Directory containing lesion labels.
        synthetic_lesions_dir (str): Directory containing synthetic lesions.
        output_dir (str): Directory to save model checkpoints and results.
        adc_mean_atlas_path (str, optional): Path to ADC mean atlas. Defaults to None.
        adc_std_atlas_path (str, optional): Path to ADC standard deviation atlas. Defaults to None.
        num_epochs (int, optional): Number of training epochs. Defaults to 200.
        batch_size (int, optional): Batch size for training. Defaults to 4.
        dilation_radius (int, optional): Radius for dilating lesion masks. Defaults to 3.
        smoothing_iterations (int, optional): Number of iterations for smoothing. Defaults to 5.
    """
    # Validate directories exist
    for dir_path, dir_name in [(adc_dir, 'ADC'), (label_dir, 'Label'), (synthetic_lesions_dir, 'Synthetic lesions')]:
        if not os.path.exists(dir_path):
            raise ValueError(f"{dir_name} directory not found: {dir_path}")
    
    # Validate atlas paths if provided
    if adc_mean_atlas_path and not os.path.exists(adc_mean_atlas_path):
        raise ValueError(f"ADC mean atlas not found: {adc_mean_atlas_path}")
    if adc_std_atlas_path and not os.path.exists(adc_std_atlas_path):
        raise ValueError(f"ADC std atlas not found: {adc_std_atlas_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Create dataset and data loaders
    train_dataset = LesionInpaintingDataset(
        adc_dir=adc_dir, 
        label_dir=label_dir, 
        synthetic_lesions_dir=synthetic_lesions_dir,
        train=True,
        split=0.8,
        adc_mean_atlas_path=adc_mean_atlas_path,
        adc_std_atlas_path=adc_std_atlas_path,
        dilation_radius=dilation_radius,
        smoothing_iterations=smoothing_iterations
    )
    
    val_dataset = LesionInpaintingDataset(
        adc_dir=adc_dir, 
        label_dir=label_dir, 
        synthetic_lesions_dir=synthetic_lesions_dir,
        train=False,
        split=0.8,
        adc_mean_atlas_path=adc_mean_atlas_path,
        adc_std_atlas_path=adc_std_atlas_path,
        dilation_radius=dilation_radius,
        smoothing_iterations=smoothing_iterations
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Initialize GAN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    gan_model = LesionInpaintingGAN().to(device)
    
    # Setup optimizers
    optimizer_G = torch.optim.Adam(gan_model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(gan_model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_g_loss, train_d_loss = 0.0, 0.0
        gan_model.train()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Train discriminator
            optimizer_D.zero_grad()
            
            # Generate fake images
            fake_images = gan_model.generator(inputs)
            
            # Real loss
            real_validity = gan_model.discriminator(targets, inputs)
            
            # Fake loss
            fake_validity = gan_model.discriminator(fake_images.detach(), inputs)
            
            # Discriminator loss
            d_loss = torch.mean(fake_validity) - torch.mean(real_validity)
            d_loss.backward()
            optimizer_D.step()
            
            # Apply weight clipping to discriminator (WGAN)
            for p in gan_model.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            
            # Train generator every n_critic iterations
            n_critic = 5
            if batch_idx % n_critic == 0:
                optimizer_G.zero_grad()
                
                # Generate fake images
                fake_images = gan_model.generator(inputs)
                
                # Adversarial loss
                fake_validity = gan_model.discriminator(fake_images, inputs)
                adversarial_loss = -torch.mean(fake_validity)
                
                # Pixel-wise loss (L1)
                pixel_loss = F.l1_loss(fake_images, targets)
                
                # Content loss (focus on preserving non-lesion areas)
                syn_lesion_mask = inputs[:, 1:2]  # Extract synthetic lesion mask
                non_lesion_mask = 1 - syn_lesion_mask  # Invert mask to focus on non-lesion areas
                content_loss = F.l1_loss(fake_images * non_lesion_mask, targets * non_lesion_mask)
                
                # Total generator loss - weighted sum of the above
                g_loss = adversarial_loss + 10.0 * pixel_loss + 5.0 * content_loss
                g_loss.backward()
                optimizer_G.step()
                
                train_g_loss += g_loss.item()
            
            train_d_loss += d_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}: G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")
        
        # Validation
        val_g_loss, val_d_loss = 0.0, 0.0
        gan_model.eval()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Generate fake images
                fake_images = gan_model.generator(inputs)
                
                # Compute losses (same as training, but without backprop)
                real_validity = gan_model.discriminator(targets, inputs)
                fake_validity = gan_model.discriminator(fake_images, inputs)
                
                d_loss = torch.mean(fake_validity) - torch.mean(real_validity)
                
                adversarial_loss = -torch.mean(fake_validity)
                pixel_loss = F.l1_loss(fake_images, targets)
                
                syn_lesion_mask = inputs[:, 1:2]
                non_lesion_mask = 1 - syn_lesion_mask
                content_loss = F.l1_loss(fake_images * non_lesion_mask, targets * non_lesion_mask)
                
                g_loss = adversarial_loss + 10.0 * pixel_loss + 5.0 * content_loss
                
                val_g_loss += g_loss.item()
                val_d_loss += d_loss.item()
        
        # Print epoch results
        train_g_loss /= len(train_loader)
        train_d_loss /= len(train_loader)
        val_g_loss /= len(val_loader)
        val_d_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train G loss: {train_g_loss:.4f}, D loss: {train_d_loss:.4f}")
        print(f"  Val G loss: {val_g_loss:.4f}, D loss: {val_d_loss:.4f}")
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'generator_state_dict': gan_model.generator.state_dict(),
            'discriminator_state_dict': gan_model.discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'train_g_loss': train_g_loss,
            'train_d_loss': train_d_loss,
            'val_g_loss': val_g_loss,
            'val_d_loss': val_d_loss,
        }, os.path.join(output_dir, 'checkpoints', f'epoch_{epoch+1}.pth'))
        
        # Generate and save visualizations
        if (epoch + 1) % 5 == 0:
            # Get a sample from validation set
            sample_idx = np.random.randint(0, len(val_dataset))
            sample_input, sample_target = val_dataset[sample_idx]
            
            sample_input = sample_input.unsqueeze(0).to(device)
            sample_target = sample_target.unsqueeze(0).to(device)
            
            gan_model.eval()
            with torch.no_grad():
                sample_output = gan_model.generator(sample_input)
            
            # Convert tensors to numpy for visualization
            input_healthy = sample_input[0, 0].cpu().numpy()
            input_mask = sample_input[0, 1].cpu().numpy()
            generated = sample_output[0, 0].cpu().numpy()
            target = sample_target[0, 0].cpu().numpy()
            
            # Calculate change map (difference between healthy and generated)
            change_map = np.abs(generated - input_healthy)
            
            # Save central slices for visualization
            slice_idx = input_healthy.shape[0] // 2
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            axes[0].imshow(input_healthy[slice_idx], cmap='gray')
            axes[0].set_title('Input (Healthy)')
            axes[0].axis('off')
            
            axes[1].imshow(input_mask[slice_idx], cmap='gray')
            axes[1].set_title('Synthetic Lesion Mask')
            axes[1].axis('off')
            
            axes[2].imshow(generated[slice_idx], cmap='gray')
            axes[2].set_title('Generated')
            axes[2].axis('off')
            
            axes[3].imshow(target[slice_idx], cmap='gray')
            axes[3].set_title('Target')
            axes[3].axis('off')
            
            axes[4].imshow(change_map[slice_idx], cmap='hot')
            axes[4].set_title('Change Map')
            axes[4].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', f'epoch_{epoch+1}.png'))
            plt.close(fig)
    
    print("Training complete!")


def apply_inpainting(gan_model, adc_file, synthetic_lesion_file, output_file):
    """
                pdf.savefig(fig, dpi=150)
                plt.close()
        
        print(f"Uložena vizualizace tvorby zdravého mozku do PDF: {output_path}")
    else:
        # Jeden obrázek pro méně než 15 řezů
        fig, axs = plt.subplots(4, len(lesion_slices), figsize=(4*len(lesion_slices), 16))
        
        # Pro případ jednoho řezu
        if len(lesion_slices) == 1:
            axs = axs.reshape(4, 1)
        
        for j, slice_idx in enumerate(lesion_slices):
            # Původní ADC řez
            axs[0, j].imshow(adc_array[slice_idx], cmap='gray')
            axs[0, j].set_title(f'Řez {slice_idx}')
            axs[0, j].axis('off')
            
            # Maska léze jako překryv
            mask = label_array[slice_idx] > 0
            input_with_mask = np.stack([adc_array[slice_idx], adc_array[slice_idx], adc_array[slice_idx]], axis=2)
            input_with_mask[mask, 0] = 1.0  # Červený kanál
            input_with_mask[mask, 1] = 0.0  # Zelený kanál
            input_with_mask[mask, 2] = 0.0  # Modrý kanál
            
            axs[1, j].imshow(input_with_mask)
            axs[1, j].set_title('Označení léze')
            axs[1, j].axis('off')
            
            # Zdravý mozek - rekonstrukce
            axs[2, j].imshow(healthy_brain[slice_idx], cmap='gray')
            axs[2, j].set_title('Rekonstruovaný zdravý mozek')
            axs[2, j].axis('off')
            
            # Mapa rozdílů - co bylo změněno
            diff_map = np.abs(healthy_brain[slice_idx] - adc_array[slice_idx])
            
            # Vypočítat průměrnou změnu pro tento konkrétní řez v oblasti léze
            slice_mask = label_array[slice_idx] > 0
            if np.any(slice_mask):
                slice_orig = adc_array[slice_idx][slice_mask]
                slice_healthy = healthy_brain[slice_idx][slice_mask]
                slice_mean_change = np.mean(np.abs(slice_healthy - slice_orig))
                diff_title = f'Mapa změn (Δ={slice_mean_change:.4f})'
            else:
                diff_title = 'Mapa změn'
            
            axs[3, j].imshow(diff_map, cmap='hot')
            axs[3, j].set_title(diff_title)
            axs[3, j].axis('off')
        
        # Přidat popisky řad
        axs[0, 0].set_ylabel('Původní ADC')
        axs[1, 0].set_ylabel('Léze')
        axs[2, 0].set_ylabel('Zdravý mozek')
        axs[3, 0].set_ylabel('Změny')
        
        # Celkový titulek
        if title:
            plt.suptitle(f"{title}\n{stats_text}")
        else:
            plt.suptitle(stats_text)
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Uložena vizualizace tvorby zdravého mozku: {output_path}")


def create_and_visualize_healthy_brain(adc_file, label_file, adc_mean_atlas_path, adc_std_atlas_path, output_file):
    """
    Vytvoří a vizualizuje zdravý mozek pomocí atlasu.
    
    Args:
        adc_file: Cesta k ADC souboru
        label_file: Cesta k souboru s maskou léze
        adc_mean_atlas_path: Cesta k průměrnému ADC atlasu
        adc_std_atlas_path: Cesta k atlasu směrodatných odchylek ADC
        output_file: Cesta pro uložení vizualizace
    """
    # Import potřebných knihoven
    import traceback
    import numpy as np
    import matplotlib.pyplot as plt
    import SimpleITK as sitk
    import os
    from scipy import ndimage
    
    # Initialize some variables
    adc_array = None
    label_array = None
    
    try:
        print("### DEBUG: Starting visualization function")
        print(f"### DEBUG: Output will be saved to {output_file}")
        
        # Vytvořit výstupní adresář, pokud neexistuje
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Načíst atlas, pokud existuje
        has_atlas = False
        adc_mean_atlas = None
        adc_std_atlas = None
        
        if adc_mean_atlas_path and os.path.exists(adc_mean_atlas_path):
            try:
                print(f"### DEBUG: Loading ADC atlas from {adc_mean_atlas_path}")
                atlas_img = sitk.ReadImage(adc_mean_atlas_path)
                adc_mean_atlas = sitk.GetArrayFromImage(atlas_img)
                print(f"### DEBUG: Atlas loaded, shape: {adc_mean_atlas.shape}, dtype: {adc_mean_atlas.dtype}")
                print(f"### DEBUG: Atlas min: {np.min(adc_mean_atlas)}, max: {np.max(adc_mean_atlas)}")
                
                if adc_std_atlas_path and os.path.exists(adc_std_atlas_path):
                    print(f"### DEBUG: Loading ADC std atlas from {adc_std_atlas_path}")
                    std_atlas_img = sitk.ReadImage(adc_std_atlas_path)
                    adc_std_atlas = sitk.GetArrayFromImage(std_atlas_img)
                    print(f"### DEBUG: Std atlas shape: {adc_std_atlas.shape}")
                
                has_atlas = True
            except Exception as e:
                print(f"### ERROR: Failed to load atlas: {e}")
                traceback.print_exc()
                has_atlas = False
        
        # Načíst ADC a label data
        try:
            print(f"### DEBUG: Loading ADC file: {adc_file}")
            adc_img = sitk.ReadImage(adc_file)
            adc_array = sitk.GetArrayFromImage(adc_img)
            print(f"### DEBUG: ADC loaded, shape: {adc_array.shape}, dtype: {adc_array.dtype}")
            print(f"### DEBUG: ADC min: {np.min(adc_array)}, max: {np.max(adc_array)}")
            
            print(f"### DEBUG: Loading label file: {label_file}")
            label_img = sitk.ReadImage(label_file)
            label_array = sitk.GetArrayFromImage(label_img).astype(bool)
            print(f"### DEBUG: Label loaded, shape: {label_array.shape}, dtype: {label_array.dtype}")
            print(f"### DEBUG: Label sum: {np.sum(label_array)} (number of voxels in lesion)")
        except Exception as e:
            print(f"### ERROR: Failed to load ADC or label file: {e}")
            traceback.print_exc()
            
            # Create a simple error image and save it
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error loading data:\n{str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12)
            plt.axis('off')
            plt.savefig(output_file)
            plt.close()
            return
        
        # Naměřit statistiky o obrazu
        try:
            print("### DEBUG: Calculating image statistics")
            non_zero_mask = adc_array > 0
            if np.sum(non_zero_mask) > 0:
                adc_mean = np.mean(adc_array[non_zero_mask])
                adc_std = np.std(adc_array[non_zero_mask])
                non_zero = adc_array[non_zero_mask]
                p01 = np.percentile(non_zero, 1) if len(non_zero) > 0 else 0
                p99 = np.percentile(non_zero, 99) if len(non_zero) > 0 else 1
            else:
                adc_mean = 0
                adc_std = 0
                p01 = 0
                p99 = 1
                print("### WARNING: No non-zero values in ADC array!")
            
            print(f"### DEBUG: ADC stats - mean: {adc_mean:.4f}, std: {adc_std:.4f}, p01: {p01:.4f}, p99: {p99:.4f}")
        except Exception as e:
            print(f"### ERROR: Failed to calculate statistics: {e}")
            traceback.print_exc()
            p01, p99 = 0, 1  # Default values
        
        # Normalizace ADC
        try:
            print("### DEBUG: Normalizing ADC")
            adc_array_norm = np.copy(adc_array)
            adc_array_norm[adc_array_norm < p01] = p01
            adc_array_norm[adc_array_norm > p99] = p99
            adc_array_norm = (adc_array_norm - p01) / (p99 - p01 + 1e-8)  # Avoid division by zero
            print(f"### DEBUG: Normalized ADC min: {np.min(adc_array_norm)}, max: {np.max(adc_array_norm)}")
        except Exception as e:
            print(f"### ERROR: Failed to normalize ADC: {e}")
            traceback.print_exc()
            # Create a normalized version anyway for fallback
            adc_array_norm = (adc_array - np.min(adc_array)) / (np.max(adc_array) - np.min(adc_array) + 1e-8)
        
        # Vytvořit zdravý mozek
        try:
            healthy_brain = None
            
            if has_atlas:
                print("### DEBUG: Creating healthy brain using atlas")
                
                # Funkce pro registraci atlasu
                registered_atlas, registered_std_atlas = register_atlas_standalone(
                    adc_img, adc_array, adc_mean_atlas, adc_std_atlas, label_array)
                
                if registered_atlas is not None:
                    print("### DEBUG: Atlas registration successful")
                    print(f"### DEBUG: Registered atlas shape: {registered_atlas.shape}")
                    print(f"### DEBUG: Registered atlas min: {np.min(registered_atlas)}, max: {np.max(registered_atlas)}")
                    
                    # Normalizovat atlas do stejného rozsahu jako ADC
                    atlas_normalized = np.copy(registered_atlas)
                    atlas_normalized[atlas_normalized < p01] = p01
                    atlas_normalized[atlas_normalized > p99] = p99
                    atlas_normalized = (atlas_normalized - p01) / (p99 - p01 + 1e-8)
                    
                    # Vypočítat poměr mezi hodnotami subjektu a atlasu v okolí léze
                    kernel = np.ones((3,3,3))
                    dilated_mask = ndimage.binary_dilation(label_array, structure=kernel, iterations=2)
                    border_mask = dilated_mask & np.logical_not(label_array)
                    
                    if np.sum(border_mask) > 0:
                        print(f"### DEBUG: Border mask sum: {np.sum(border_mask)}")
                        atlas_border_mean = np.mean(atlas_normalized[border_mask] + 1e-8)
                        subject_border_mean = np.mean(adc_array_norm[border_mask])
                        ratio = subject_border_mean / atlas_border_mean
                        print(f"### DEBUG: Border ratio: {ratio:.4f} (subject: {subject_border_mean:.4f}, atlas: {atlas_border_mean:.4f})")
                        ratio = np.clip(ratio, 0.5, 2.0)
                    else:
                        ratio = 1.0
                        print("### DEBUG: No border found, using default ratio 1.0")
                    
                    # Vytvořit zdravý mozek
                    healthy_brain = adc_array_norm.copy()
                    healthy_brain[label_array] = atlas_normalized[label_array] * ratio
                    
                    print("### DEBUG: Healthy brain created using atlas registration")
                else:
                    print("### DEBUG: Atlas registration failed, using basic method")
                    healthy_brain = create_basic_healthy_brain_standalone(adc_array_norm, label_array)
            else:
                print("### DEBUG: No atlas available, using basic method")
                healthy_brain = create_basic_healthy_brain_standalone(adc_array_norm, label_array)
            
            print(f"### DEBUG: Healthy brain min: {np.min(healthy_brain)}, max: {np.max(healthy_brain)}")
        except Exception as e:
            print(f"### ERROR: Failed to create healthy brain: {e}")
            traceback.print_exc()
            
            # Create a simple healthy brain as fallback
            healthy_brain = adc_array_norm.copy()
            if label_array is not None and np.sum(label_array) > 0:
                avg_value = np.mean(adc_array_norm[~label_array])
                healthy_brain[label_array] = avg_value
        
        # Vypočítat statistiky
        try:
            print("### DEBUG: Calculating statistics")
            if np.sum(label_array) > 0:
                mean_orig = np.mean(adc_array_norm[label_array])
                mean_healthy = np.mean(healthy_brain[label_array])
                mean_change = np.mean(np.abs(healthy_brain[label_array] - adc_array_norm[label_array]))
                median_change = np.median(np.abs(healthy_brain[label_array] - adc_array_norm[label_array]))
                
                stats_text = (f"Statistika oblasti léze:\n"
                            f"Průměrná hodnota původní: {mean_orig:.4f}\n"
                            f"Průměrná hodnota zdravá: {mean_healthy:.4f}\n"
                            f"Průměrná absolutní změna: {mean_change:.4f}\n"
                            f"Mediánová absolutní změna: {median_change:.4f}")
            else:
                stats_text = "Statistika oblasti léze: Žádná léze nenalezena"
            
            print(f"### DEBUG: {stats_text}")
        except Exception as e:
            print(f"### ERROR: Failed to calculate statistics: {e}")
            traceback.print_exc()
            stats_text = "Statistiky nebylo možné vypočítat"
        
        # Najít všechny řezy, kde se vyskytuje léze
        try:
            print("### DEBUG: Finding slices with lesion")
            lesion_slices = []
            for slice_idx in range(label_array.shape[0]):
                if np.any(label_array[slice_idx]):
                    lesion_slices.append(slice_idx)
            
            print(f"### DEBUG: Found {len(lesion_slices)} slices with lesion")
            
            if len(lesion_slices) == 0:
                print("### WARNING: No lesion found in mask!")
                # Pokud není nalezena žádná léze, použít prostřední řez a několik okolních
                middle_slice = label_array.shape[0] // 2
                num_slices = min(5, label_array.shape[0])
                start_slice = max(0, middle_slice - num_slices // 2)
                lesion_slices = list(range(start_slice, start_slice + num_slices))
                print(f"### DEBUG: Using slices {lesion_slices} as fallback")
        except Exception as e:
            print(f"### ERROR: Failed to find lesion slices: {e}")
            traceback.print_exc()
            # Fallback - use middle slice
            lesion_slices = [adc_array.shape[0] // 2]
        
        # Vytvořit vizualizaci
        try:
            print("### DEBUG: Creating visualization")
            # Omezit počet řezů pro vizualizaci
            max_slices = 6  # Maximální počet řezů pro přehlednost
            if len(lesion_slices) > max_slices:
                # Vybrat rovnoměrně rozložené řezy
                indices = np.round(np.linspace(0, len(lesion_slices) - 1, max_slices)).astype(int)
                lesion_slices = [lesion_slices[i] for i in indices]
                print(f"### DEBUG: Limited visualization to {max_slices} slices: {lesion_slices}")
            
            # Počet řádků a sloupců závisí na dostupnosti metod
            n_rows = 4  # 4 řádky pro plynulou metodu
            if atlas_healthy_brain is not None:
                n_rows = 5  # Přidat řádek pro atlasovou metodu
            
            n_cols = len(lesion_slices)
            
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
            if n_cols == 1:
                axs = axs.reshape(n_rows, 1)
            
            for j, slice_idx in enumerate(lesion_slices):
                print(f"### DEBUG: Processing slice {slice_idx} (index {j})")
                # Původní ADC
                axs[0, j].imshow(adc_array_norm[slice_idx], cmap='gray')
                axs[0, j].set_title(f'Řez {slice_idx}')
                axs[0, j].axis('off')
                
                # Maska léze
                axs[1, j].imshow(adc_array_norm[slice_idx], cmap='gray')
                masked = np.ma.masked_where(~label_array[slice_idx], np.ones_like(adc_array_norm[slice_idx]))
                axs[1, j].imshow(masked, cmap='autumn', alpha=0.5)
                axs[1, j].set_title(f'Léze')
                axs[1, j].axis('off')
                
                # Zdravý mozek - plynulá metoda
                axs[2, j].imshow(healthy_brain[slice_idx], cmap='gray')
                axs[2, j].set_title(f'Zdravý mozek')
                axs[2, j].axis('off')
                
                # Mapa změn (rozdíl)
                diff_healthy = np.abs(healthy_brain[slice_idx] - adc_array_norm[slice_idx])
                if np.isnan(diff_healthy).any():
                    print(f"### WARNING: NaN values in difference map for slice {slice_idx}")
                    diff_healthy = np.nan_to_num(diff_healthy)
                
                # Vypočítat průměrnou změnu pro tento konkrétní řez v oblasti léze
                slice_mask = label_array[slice_idx] > 0
                if np.any(slice_mask):
                    slice_orig = adc_array_norm[slice_idx][slice_mask]
                    slice_healthy = healthy_brain[slice_idx][slice_mask]
                    # Ošetření NaN hodnot
                    valid_mask = ~np.isnan(slice_orig) & ~np.isnan(slice_healthy)
                    if np.any(valid_mask):
                        slice_orig = slice_orig[valid_mask]
                        slice_healthy = slice_healthy[valid_mask]
                        slice_mean_change = np.mean(np.abs(slice_healthy - slice_orig))
                        diff_title = f'Mapa změn (Δ={slice_mean_change:.4f})'
                    else:
                        diff_title = 'Mapa změn (chybí data)'
                else:
                    diff_title = 'Mapa změn'
                
                axs[3, j].imshow(diff_healthy, cmap='hot', vmin=0, vmax=np.max(diff_healthy) if np.max(diff_healthy) > 0 else 1)
                axs[3, j].set_title(diff_title)
                axs[3, j].axis('off')
                
                # Pokud máme atlas, přidáme i jeho výsledek
                if atlas_healthy_brain is not None:
                    axs[4, j].imshow(atlas_healthy_brain[slice_idx], cmap='gray')
                    axs[4, j].set_title(f'Zdravý mozek (atlas)')
                    axs[4, j].axis('off')
            
            # Přidat popisky řad
            axs[0, 0].set_ylabel('Původní ADC')
            axs[1, 0].set_ylabel('Léze')
            axs[2, 0].set_ylabel('Zdravý mozek\n(plynulá metoda)')
            axs[3, 0].set_ylabel('Mapa změn')
            if atlas_healthy_brain is not None:
                axs[4, 0].set_ylabel('Zdravý mozek\n(atlas)')
            
            # Celkový titulek
            plt.suptitle(f"Vizualizace vytvoření zdravého mozku\n{stats_text}")
            
            plt.tight_layout()
            print(f"### DEBUG: Saving visualization to {output_file}")
            plt.savefig(output_file, dpi=150)
            plt.close()
            
            print(f"### DEBUG: Visualization saved successfully to {output_file}")
        except Exception as e:
            print(f"### ERROR: Failed to create visualization: {e}")
            traceback.print_exc()
            
            # Create a simple error image and save it
            try:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"Error creating visualization:\n{str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12)
                plt.axis('off')
                plt.savefig(output_file)
                plt.close()
                print(f"### DEBUG: Saved error message to {output_file}")
            except Exception as e2:
                print(f"### ERROR: Even failed to save error message: {e2}")
    
    except Exception as e:
        print(f"### ERROR: Unexpected error in create_and_visualize_healthy_brain: {e}")
        traceback.print_exc()
        
        # Final fallback - save an error message
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Unexpected error:\n{str(e)}", 
                   horizontalalignment='center', verticalalignment='center', 
                   fontsize=12)
            plt.axis('off')
            plt.savefig(output_file)
            plt.close()
            print(f"### DEBUG: Saved unexpected error message to {output_file}")
        except:
            print("### ERROR: Could not save error visualization, giving up.")

# Samostatná funkce pro vytvoření základního zdravého mozku (nezávislá na třídě)
def create_basic_healthy_brain_standalone(adc_normalized, label_array):
    """
    Vytvoří jednoduchý model zdravého mozku pomocí průměrných hodnot zdravé tkáně.
    
    Args:
        adc_normalized: Normalizovaný ADC obraz
        label_array: Maska léze
        
    Returns:
        Zdravý mozek s nahrazenými lézemi
    """
    from scipy import ndimage
    
    # Rozšířit masku léze pomocí dilace pro lepší zachycení oblasti
    kernel = np.ones((3,3,3))
    dilated_mask = ndimage.binary_dilation(label_array, structure=kernel, iterations=2)
    
    # Vytvořit hranici okolo léze (dilated - original)
    border_mask = dilated_mask & np.logical_not(label_array)
    
    # Vypočítat průměrnou hodnotu zdravé tkáně v okolí léze
    if np.sum(border_mask) > 10:
        healthy_tissue_value = np.mean(adc_normalized[border_mask])
        print(f"Používám hodnotu zdravé tkáně z okolí hranice: {healthy_tissue_value:.4f}")
    else:
        # Fallback: použít průměrnou hodnotu všech voxelů mimo lézi
        healthy_tissue_value = np.mean(adc_normalized[np.logical_not(label_array)])
        print(f"Používám globální hodnotu zdravé tkáně: {healthy_tissue_value:.4f}")
    
    # Vytvořit zdravý mozek nahrazením oblasti léze
    healthy_brain = adc_normalized.copy()
    healthy_brain[label_array] = healthy_tissue_value
    
    return healthy_brain

# Funkce pro samostatnou registraci atlasu bez závislosti na třídě
def register_atlas_standalone(subject_img, subject_array, adc_mean_atlas, adc_std_atlas, label_array):
    """
    Registruje atlas k danému subjektu, vyhýbá se oblasti s lézemi.
    Samostatná verze bez závislosti na třídě LesionInpaintingDataset.
    
    Args:
        subject_img: SimpleITK obraz subjektu
        subject_array: numpy pole subjektu
        adc_mean_atlas: Průměrný ADC atlas jako numpy pole
        adc_std_atlas: Atlas směrodatných odchylek jako numpy pole (nebo None)
        label_array: maska léze
        
    Returns:
        registered_atlas: registrovaný atlas
        registered_std_atlas: registrovaný atlas směrodatných odchylek
    """
    try:
        import SimpleITK as sitk
        from scipy import ndimage
        import numpy as np
        
        print("Registruji atlas k subjektu...")
        
        # Kontrola velikosti atlasu a subjektu
        subject_size = subject_array.shape
        atlas_size = adc_mean_atlas.shape
        
        print(f"Velikost subjektu: {subject_size}, velikost atlasu: {atlas_size}")
        
        # Ověříme, že atlas není prázdný nebo obsahuje NaN hodnoty
        if np.isnan(adc_mean_atlas).any():
            print("VAROVÁNÍ: Atlas obsahuje NaN hodnoty!")
            adc_mean_atlas = np.nan_to_num(adc_mean_atlas)
        
        # Zkusíme použít jednoduchou registraci, protože ElastixImageFilter není dostupný
        print("Používám SimpleITK pro základní registraci...")
        
        # Vytvořit kopii atlasu do SimpleITK obrazu
        atlas_img = sitk.GetImageFromArray(adc_mean_atlas)
            
        # Zkopírovat metadata z předmětu do atlasu
        atlas_img.SetSpacing(subject_img.GetSpacing())
        atlas_img.SetOrigin(subject_img.GetOrigin())
        atlas_img.SetDirection(subject_img.GetDirection())
        
        # Nejprve musíme provést resample atlasu na stejnou velikost jako subjekt
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(subject_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        atlas_img_resampled = resampler.Execute(atlas_img)
        
        print("Resampled atlas to subject dimensions")
        
        # Konvertovat zpět na numpy pole
        registered_atlas = sitk.GetArrayFromImage(atlas_img_resampled)
        
        # Registrovat standardní odchylku, pokud je k dispozici
        registered_std_atlas = None
        if adc_std_atlas is not None:
            std_atlas_img = sitk.GetImageFromArray(adc_std_atlas)
            std_atlas_img.CopyInformation(atlas_img)
            
            # Použít stejné převzorkování pro atlas směrodatných odchylek
            std_atlas_img_resampled = resampler.Execute(std_atlas_img)
            registered_std_atlas = sitk.GetArrayFromImage(std_atlas_img_resampled)
        
        # Zkontrolujeme výsledek registrace na díry (black spots)
        registered_atlas = fix_registration_holes_standalone(registered_atlas, subject_array)
        
        return registered_atlas, registered_std_atlas
            
    except Exception as e:
        print(f"Registrace atlasu selhala: {e}")
        import traceback
        traceback.print_exc()
        
        # Záložní metoda: použít scipy pro jednoduchou registraci
        print("Používám scipy resample jako záložní metodu registrace...")
        
        try:
            from scipy import ndimage
            
            # Vytvořit nový atlas s velikostí subjektu
            registered_atlas = np.zeros_like(subject_array)
            
            # Vypočítat faktory zoomu pro každou dimenzi
            zoom_factors = (subject_size[0] / atlas_size[0],
                           subject_size[1] / atlas_size[1],
                           subject_size[2] / atlas_size[2])
            
            # Použít scipy zoom pro resize
            resized_atlas = ndimage.zoom(adc_mean_atlas, zoom_factors, order=1)
            
            # Oříznout nebo doplnit, pokud velikosti nejsou přesně stejné
            if resized_atlas.shape != subject_array.shape:
                print(f"Po resample se velikosti neshodují: subjekt {subject_array.shape} vs atlas {resized_atlas.shape}")
                
                # Vytvořit nové pole správné velikosti
                registered_atlas = np.zeros_like(subject_array)
                
                # Vypočítat minimální společné rozměry
                min_shape = [min(subject_array.shape[i], resized_atlas.shape[i]) for i in range(3)]
                
                # Zkopírovat dostupná data
                registered_atlas[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                    resized_atlas[:min_shape[0], :min_shape[1], :min_shape[2]]
            else:
                registered_atlas = resized_atlas
            
            # Registrovat standardní odchylku, pokud je k dispozici
            registered_std_atlas = None
            if adc_std_atlas is not None:
                resized_std_atlas = ndimage.zoom(adc_std_atlas, zoom_factors, order=1)
                
                if resized_std_atlas.shape != subject_array.shape:
                    registered_std_atlas = np.zeros_like(subject_array)
                    min_shape = [min(subject_array.shape[i], resized_std_atlas.shape[i]) for i in range(3)]
                    registered_std_atlas[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                        resized_std_atlas[:min_shape[0], :min_shape[1], :min_shape[2]]
                else:
                    registered_std_atlas = resized_std_atlas
            
            print("Scipy registrace dokončena")
            
            # Zkontrolujeme výsledek registrace na díry (black spots)
            registered_atlas = fix_registration_holes_standalone(registered_atlas, subject_array)
            
            return registered_atlas, registered_std_atlas
            
        except Exception as e:
            print(f"Záložní registrace atlasu také selhala: {e}")
            return None, None

# Samostatná funkce pro opravu děr v registrovaném atlasu
def fix_registration_holes_standalone(registered_atlas, subject_array):
    """
    Opraví díry v registrovaném atlasu pomocí interpolace a prahování.
    
    Args:
        registered_atlas: Registrovaný atlas s možnými dírami
        subject_array: Původní obraz subjektu pro referenci
        
    Returns:
        Opravený registrovaný atlas bez děr
    """
    try:
        from scipy import ndimage
        import numpy as np
        
        print("Kontroluji a opravuji díry v registrovaném atlasu...")
        
        # Vytvořit kopii atlasu
        fixed_atlas = registered_atlas.copy()
        
        # Identifikovat díry (příliš nízké hodnoty v oblastech, kde subjekt má signál)
        # Předpokládáme, že díry mají hodnotu 0 nebo blízko 0
        # a objevují se tam, kde subjekt má nenulový signál
        subject_mask = subject_array > 0.01
        hole_mask = (fixed_atlas < 0.01) & subject_mask
        
        if np.sum(hole_mask) > 0:
            print(f"Nalezeno {np.sum(hole_mask)} voxelů s dírami")
            
            # Použít morfologické operace k identifikaci malých děr
            # Dilatace a následná eroze může vyplnit malé díry
            filled_mask = ndimage.binary_closing(~hole_mask, structure=np.ones((3,3,3)), iterations=2)
            
            # Interpolovat hodnoty pomocí okolí
            # Způsob 1: Použít medián filtr pro vyplnění děr
            temp_atlas = fixed_atlas.copy()
            temp_atlas[hole_mask] = np.nan  # Označit díry jako NaN
            
            # Vytvořit masku pro median filtr (pouze neNaN hodnoty)
            median_filtered = ndimage.median_filter(np.nan_to_num(temp_atlas), size=5)
            
            # Pouze nahradit hodnoty v dírách
            fixed_atlas[hole_mask] = median_filtered[hole_mask]
            
            # Způsob 2: Pro větší díry použít vzdálenostně váženou interpolaci
            try:
                from scipy.interpolate import griddata
                
                # Pro každý řez zvlášť (pro efektivitu)
                for z in range(fixed_atlas.shape[0]):
                    slice_holes = hole_mask[z]
                    if np.sum(slice_holes) > 0:
                        # Získat souřadnice známých bodů a jejich hodnoty
                        y_known, x_known = np.where(~slice_holes)
                        values_known = fixed_atlas[z, y_known, x_known]
                        
                        # Souřadnice bodů k interpolaci (díry)
                        y_holes, x_holes = np.where(slice_holes)
                        
                        if len(y_known) > 0 and len(y_holes) > 0:
                            # Připravit body pro interpolaci
                            points = np.column_stack((y_known, x_known))
                            holes = np.column_stack((y_holes, x_holes))
                            
                            # Interpolovat hodnoty
                            try:
                                interpolated = griddata(points, values_known, holes, method='linear', fill_value=np.mean(values_known))
                                fixed_atlas[z, y_holes, x_holes] = interpolated
                            except Exception as e:
                                print(f"Interpolace selhala pro řez {z}: {e}")
            except ImportError:
                print("griddata není k dispozici, používám pouze mediánový filtr")
            
            print("Díry v atlasu byly opraveny")
        else:
            print("Žádné díry v atlasu nebyly nalezeny")
        
        return fixed_atlas
        
    except Exception as e:
        print(f"Oprava děr v atlasu selhala: {e}")
        import traceback
        traceback.print_exc()
        return registered_atlas  # Vrátit původní atlas, pokud oprava selže

# Nová funkce pro plynulou úpravu hodnot v oblasti léze
def create_smooth_healthy_brain(adc_array, label_array, dilation_radius=3, smoothing_iterations=5):
    """
    Creates a realistic healthy brain by smoothly inpainting lesion areas.
    
    This function replaces lesion areas with values that smoothly transition from the
    surrounding healthy tissue, creating a more realistic appearance than simple
    average-based replacement.
    
    Args:
        adc_array (np.ndarray): The original ADC map
        label_array (np.ndarray): Binary mask indicating lesion areas (1 = lesion)
        dilation_radius (int): Radius for dilating the lesion mask to find boundary
        smoothing_iterations (int): Number of smoothing iterations to apply
        
    Returns:
        np.ndarray: A "healthy" version of the ADC map with lesions smoothly inpainted
    """
    # Make a copy of the original array to avoid modifying it
    healthy_brain = adc_array.copy()
    
    # Only proceed if there are lesions
    if not np.any(label_array):
        return healthy_brain
    
    try:
        # Create a dilated mask to find the boundary
        dilated_mask = binary_dilation(label_array, iterations=dilation_radius)
        
        # Create a boundary mask (dilated area excluding the original lesion)
        boundary_mask = dilated_mask & np.logical_not(label_array)
        
        # If no boundary was found, use a simple approach
        if not np.any(boundary_mask):
            print("Warning: No boundary found after dilation. Using fallback method.")
            healthy_tissue_value = np.mean(adc_array[np.logical_not(label_array)])
            healthy_brain[label_array] = healthy_tissue_value
            return healthy_brain
        
        # Calculate average value in the boundary region
        boundary_values = adc_array[boundary_mask]
        boundary_mean = np.mean(boundary_values)
        
        # Initialize the lesion area with the boundary mean value
        healthy_brain[label_array] = boundary_mean
        
        # Create a smoothed version by applying multiple iterations of Gaussian filtering
        # but only inside and around the lesion
        combined_mask = binary_dilation(label_array, iterations=dilation_radius + 2)
        
        # Apply iterative smoothing within the extended mask
        for _ in range(smoothing_iterations):
            # Create a temporary copy for this iteration
            temp_brain = healthy_brain.copy()
            
            # Apply Gaussian filter to the entire brain
            smoothed = gaussian_filter(temp_brain, sigma=1.0)
            
            # Replace only the lesion and nearby areas with smoothed values
            healthy_brain[combined_mask] = smoothed[combined_mask]
        
        # Make a final pass to ensure values are within a reasonable range
        # Calculate stats of healthy tissue for reference
        healthy_mean = np.mean(adc_array[np.logical_not(label_array)])
        healthy_std = np.std(adc_array[np.logical_not(label_array)])
        
        # Limit values to within 2.5 standard deviations of the healthy mean
        min_value = healthy_mean - 2.5 * healthy_std
        max_value = healthy_mean + 2.5 * healthy_std
        
        # Apply limits only to the inpainted region
        healthy_brain[label_array] = np.clip(healthy_brain[label_array], min_value, max_value)
        
        return healthy_brain
    
    except Exception as e:
        print(f"Error in smooth inpainting: {str(e)}")
        # Fallback to simple replacement method
        try:
            healthy_tissue_value = np.mean(adc_array[np.logical_not(label_array)])
            healthy_brain[label_array] = healthy_tissue_value
        except:
            print("Critical error in inpainting fallback, returning original array")
            return adc_array
        
    return healthy_brain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lesion Inpainting GAN")
    subparsers = parser.add_subparsers(dest="command")
    
    # Training parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--adc_dir", required=True, help="Directory containing ADC maps")
    train_parser.add_argument("--label_dir", required=True, help="Directory containing lesion labels")
    train_parser.add_argument("--synthetic_lesions_dir", required=True, help="Directory containing synthetic lesions")
    train_parser.add_argument("--output_dir", required=True, help="Directory to save model checkpoints and results")
    train_parser.add_argument("--adc_mean_atlas_path", help="Path to ADC mean atlas")
    train_parser.add_argument("--adc_std_atlas_path", help="Path to ADC standard deviation atlas")
    train_parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    train_parser.add_argument("--dilation_radius", type=int, default=3, help="Radius for dilating lesion masks")
    train_parser.add_argument("--smoothing_iterations", type=int, default=5, help="Number of iterations for smoothing")
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Apply trained model for inference")
    infer_parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    infer_parser.add_argument("--adc_file", required=True, help="Path to ADC file")
    infer_parser.add_argument("--synthetic_lesion_file", required=True, help="Path to synthetic lesion file")
    infer_parser.add_argument("--output_file", required=True, help="Path to output file")
    
    args = parser.parse_args()
    
    if args.command == "train":
        print(f"Training with the following parameters:")
        print(f"  ADC directory: {args.adc_dir}")
        print(f"  Label directory: {args.label_dir}")
        print(f"  Synthetic lesions directory: {args.synthetic_lesions_dir}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  ADC mean atlas path: {args.adc_mean_atlas_path}")
        print(f"  ADC std atlas path: {args.adc_std_atlas_path}")
        print(f"  Number of epochs: {args.num_epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Dilation radius: {args.dilation_radius}")
        print(f"  Smoothing iterations: {args.smoothing_iterations}")
        
        train(
            adc_dir=args.adc_dir,
            label_dir=args.label_dir,
            synthetic_lesions_dir=args.synthetic_lesions_dir,
            output_dir=args.output_dir,
            adc_mean_atlas_path=args.adc_mean_atlas_path,
            adc_std_atlas_path=args.adc_std_atlas_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            dilation_radius=args.dilation_radius,
            smoothing_iterations=args.smoothing_iterations
        )
    elif args.command == "infer":
        print(f"Inference with the following parameters:")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  ADC file: {args.adc_file}")
        print(f"  Synthetic lesion file: {args.synthetic_lesion_file}")
        print(f"  Output file: {args.output_file}")
        
        apply_inpainting(
            checkpoint_path=args.checkpoint,
            adc_file=args.adc_file,
            synthetic_lesion_file=args.synthetic_lesion_file,
            output_file=args.output_file
        )
    else:
        parser.print_help()
