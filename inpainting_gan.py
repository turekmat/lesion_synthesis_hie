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


class LesionInpaintingDataset(Dataset):
    """
    Dataset for HIE lesion inpainting.
    
    It pairs:
    1. ADC maps without lesions in a specific area (truly healthy brain created using atlas)
    2. Binary masks for synthetic lesions (where to place new lesions)
    3. Ground truth ADC maps with real lesions (for learning what lesions look like)
    """
    def __init__(self, 
                 adc_dir,
                 label_dir, 
                 synthetic_lesions_dir,
                 adc_mean_atlas_path=None,
                 adc_std_atlas_path=None,
                 patch_size=(96, 96, 96),
                 mode='train',
                 transform=None):
        """
        Args:
            adc_dir: Directory with ADC maps
            label_dir: Directory with lesion masks
            synthetic_lesions_dir: Directory with synthetic lesions
            adc_mean_atlas_path: Path to the mean ADC atlas (for healthy brain approximation)
            adc_std_atlas_path: Path to the standard deviation ADC atlas
            patch_size: Size of the patches to extract
            mode: 'train' or 'val'
            transform: Optional transform to be applied on a sample
        """
        self.adc_dir = adc_dir
        self.label_dir = label_dir
        self.synthetic_lesions_dir = synthetic_lesions_dir
        self.adc_mean_atlas_path = adc_mean_atlas_path
        self.adc_std_atlas_path = adc_std_atlas_path
        self.patch_size = patch_size
        self.mode = mode
        self.transform = transform
        
        # Load ADC atlases if provided
        self.has_atlas = False
        if adc_mean_atlas_path and os.path.exists(adc_mean_atlas_path):
            print(f"Loading ADC mean atlas from: {adc_mean_atlas_path}")
            self.adc_mean_atlas = sitk.ReadImage(adc_mean_atlas_path)
            self.adc_mean_atlas_array = sitk.GetArrayFromImage(self.adc_mean_atlas)
            
            if adc_std_atlas_path and os.path.exists(adc_std_atlas_path):
                print(f"Loading ADC standard deviation atlas from: {adc_std_atlas_path}")
                self.adc_std_atlas = sitk.ReadImage(adc_std_atlas_path)
                self.adc_std_atlas_array = sitk.GetArrayFromImage(self.adc_std_atlas)
                self.has_atlas = True
            else:
                print("WARNING: ADC standard deviation atlas not provided or not found.")
                self.adc_std_atlas = None
                self.adc_std_atlas_array = None
        else:
            print("WARNING: ADC mean atlas not provided or not found. Will use simple interpolation method.")
            self.adc_mean_atlas = None
            self.adc_mean_atlas_array = None
            self.adc_std_atlas = None
            self.adc_std_atlas_array = None
        
        # Debugging info
        print(f"\n----- Dataset Initialization ({mode}) -----")
        print(f"ADC directory: {adc_dir}")
        print(f"Label directory: {label_dir}")
        print(f"Synthetic lesions directory: {synthetic_lesions_dir}")
        
        # Verify directories exist
        if not os.path.exists(adc_dir):
            print(f"WARNING: ADC directory {adc_dir} does not exist!")
        if not os.path.exists(label_dir):
            print(f"WARNING: Label directory {label_dir} does not exist!")
        if not os.path.exists(synthetic_lesions_dir):
            print(f"WARNING: Synthetic lesions directory {synthetic_lesions_dir} does not exist!")
        
        # Get all ADC files
        self.adc_files = sorted(glob.glob(os.path.join(adc_dir, "*.mha")))
        print(f"Found {len(self.adc_files)} ADC files")
        
        if len(self.adc_files) == 0:
            print(f"Contents of ADC directory: {os.listdir(adc_dir) if os.path.exists(adc_dir) else 'Directory not found'}")
        
        # Filter for patients with corresponding synthetic lesions
        valid_patients = []
        for adc_file in self.adc_files:
            # Extract base filename
            base_filename = os.path.basename(adc_file)
            
            # Check if the filename starts with prefixes and remove them
            if base_filename.startswith("Zmap_"):
                # Remove prefix for ZADC files
                base_without_prefix = base_filename[5:]  # Skip the first 5 characters "Zmap_"
            else:
                base_without_prefix = base_filename
            
            # Extract the patient ID from the first two segments
            parts = base_without_prefix.split('-')
            if len(parts) >= 2:
                patient_id = parts[0] + '-' + parts[1]
                
                # Check if corresponding synthetic lesion directory exists
                synthetic_dir = os.path.join(synthetic_lesions_dir, patient_id)
                if os.path.exists(synthetic_dir):
                    valid_patients.append(patient_id)
                else:
                    print(f"No synthetic lesions found for patient {patient_id} at {synthetic_dir}")
            else:
                print(f"Could not extract patient ID from filename: {base_filename}")
        
        self.valid_patients = valid_patients
        print(f"Found {len(self.valid_patients)} valid patients with synthetic lesions")
        
        if len(self.valid_patients) == 0:
            # Print the first few ADC filenames to help diagnose the issue
            if len(self.adc_files) > 0:
                print(f"Sample ADC files (first 5):")
            
            print(f"Contents of synthetic lesions dir: {os.listdir(synthetic_lesions_dir) if os.path.exists(synthetic_lesions_dir) else 'Directory not found'}")
        
        # Split into train/val
        if mode == 'train':
            self.patients = self.valid_patients[:int(0.8 * len(self.valid_patients))]
        else:
            self.patients = self.valid_patients[int(0.8 * len(self.valid_patients)):]
        
        print(f"Using {len(self.patients)} patients for {mode} mode")
        
        # Create patient-synthetic lesion pairs
        self.samples = []
        for patient in self.patients:
            print(f"\nProcessing patient: {patient}")
            
            # Check for ADC files 
            adc_pattern = os.path.join(adc_dir, f"*{patient}*.mha")
            adc_files = glob.glob(adc_pattern)
            if not adc_files:
                print(f"WARNING: No ADC files found for pattern: {adc_pattern}")
                continue
            adc_file = adc_files[0]
            print(f"Found ADC file: {os.path.basename(adc_file)}")
            
            # Check for label files (format: MGHNICU_xxx-VISIT_xx_lesion.mha)
            label_pattern = os.path.join(label_dir, f"{patient}_lesion.mha")
            label_files = glob.glob(label_pattern)
            if not label_files:
                print(f"WARNING: No label files found for pattern: {label_pattern}")
                # Try alternative pattern with wildcard
                alt_label_pattern = os.path.join(label_dir, f"{patient}*lesion.mha")
                label_files = glob.glob(alt_label_pattern)
                if not label_files:
                    print(f"WARNING: No label files found for alternative pattern: {alt_label_pattern}")
                    continue
            label_file = label_files[0]
            print(f"Found label file: {os.path.basename(label_file)}")
            
            # Check for synthetic lesions (format: .../MGHNICU_xxx-VISIT_xx/registered_lesion_sampleXX.mha)
            syn_pattern = os.path.join(synthetic_lesions_dir, patient, "registered_lesion_*.mha")
            synthetic_lesions = glob.glob(syn_pattern)
            if not synthetic_lesions:
                print(f"WARNING: No synthetic lesions found for pattern: {syn_pattern}")
                print(f"Does directory exist? {os.path.exists(os.path.join(synthetic_lesions_dir, patient))}")
                if os.path.exists(os.path.join(synthetic_lesions_dir, patient)):
                    print(f"Contents of patient directory: {os.listdir(os.path.join(synthetic_lesions_dir, patient))}")
                continue
            
            print(f"Found {len(synthetic_lesions)} synthetic lesions")
            
            # Create sample pairs
            for syn_lesion in synthetic_lesions:
                self.samples.append({
                    'adc': adc_file,
                    'label': label_file,
                    'synthetic_lesion': syn_lesion
                })
        
        print(f"Created {len(self.samples)} samples for {mode} mode")
        
        if len(self.samples) == 0:
            print(f"ERROR: No valid samples could be created for {mode} mode!")
            print("Please check data paths and ensure the directory structure matches expected patterns")
        
        print(f"----- End Dataset Initialization ({mode}) -----\n")
    
    def register_atlas_to_subject(self, subject_img, subject_array, label_array):
        """
        Register the ADC atlas to the subject's brain, avoiding the lesion area.
        
        Args:
            subject_img: SimpleITK image of the subject's brain
            subject_array: Numpy array of the subject's brain
            label_array: Lesion mask array
            
        Returns:
            registered_atlas_array: Registered atlas as numpy array
            registered_std_atlas_array: Registered standard deviation atlas as numpy array
        """
        # If no atlas is available, return None
        if not self.has_atlas:
            return None, None
            
        try:
            # Získat velikost dat subjektu
            subject_size = subject_array.shape
            atlas_size = self.adc_mean_atlas_array.shape
            
            print(f"Subject size: {subject_size}, Atlas size: {atlas_size}")
            
            # Metoda 1: Zkusíme SimpleITK registraci
            try:
                import SimpleITK as sitk
                print("Používám SimpleITK pro registraci atlasu...")
                
                # Pokud rozměry atlasu a subjektu neodpovídají, použijeme resample
                if subject_size != atlas_size:
                    print(f"Atlas and subject have different sizes. Resampling atlas to match subject...")
                    
                    # Vytvořit identickou transformaci
                    identity = sitk.Transform(3, sitk.sitkIdentity)
                    
                    # Resample atlasu na velikost subjektu
                    reference_image = subject_img
                    
                    # Resample mean atlas
                    resampled_mean_atlas = sitk.Resample(
                        self.adc_mean_atlas, 
                        reference_image, 
                        identity, 
                        sitk.sitkLinear,
                        0.0,
                        self.adc_mean_atlas.GetPixelID()
                    )
                    
                    registered_atlas_array = sitk.GetArrayFromImage(resampled_mean_atlas)
                    print(f"Atlas resampled to size: {registered_atlas_array.shape}")
                    
                    # Resample std atlas if available
                    if self.adc_std_atlas is not None:
                        resampled_std_atlas = sitk.Resample(
                            self.adc_std_atlas, 
                            reference_image, 
                            identity, 
                            sitk.sitkLinear,
                            0.0,
                            self.adc_std_atlas.GetPixelID()
                        )
                        registered_std_atlas_array = sitk.GetArrayFromImage(resampled_std_atlas)
                    else:
                        registered_std_atlas_array = None
                    
                    # Kontrola velikosti
                    if registered_atlas_array.shape != subject_size:
                        print(f"VAROVÁNÍ: Velikost registrovaného atlasu {registered_atlas_array.shape} se neshoduje s velikostí subjektu {subject_size}")
                        raise ValueError("Nekonzistentní velikosti po registraci")
                    
                    # Kontrola NaN hodnot
                    if np.any(np.isnan(registered_atlas_array)):
                        print("VAROVÁNÍ: NaN hodnoty v registrovaném atlasu, nahrazuji mediánem")
                        nan_mask = np.isnan(registered_atlas_array)
                        non_nan_values = registered_atlas_array[~nan_mask]
                        if len(non_nan_values) > 0:
                            median_value = np.median(non_nan_values)
                            registered_atlas_array[nan_mask] = median_value
                    
                    print("Successfully resampled atlas to match subject size")
                    return registered_atlas_array, registered_std_atlas_array
                else:
                    # Pokud mají stejnou velikost, není třeba resample
                    print("Atlas and subject have the same size. No resampling needed.")
                    return self.adc_mean_atlas_array, self.adc_std_atlas_array
                
            except Exception as sitk_error:
                print(f"SimpleITK registration failed with error: {sitk_error}")
                print("Falling back to scipy ndimage method...")
                
            # Metoda 2: Pokud SimpleITK selže, použijeme scipy ndimage
            from scipy import ndimage
            
            # Výpočet faktorů pro zoom
            zoom_factors = (subject_size[0] / atlas_size[0], 
                            subject_size[1] / atlas_size[1],
                            subject_size[2] / atlas_size[2])
            
            # Kontrola extrémních hodnot
            if max(zoom_factors) > 10 or min(zoom_factors) < 0.1:
                print(f"WARNING: Extreme zoom factors: {zoom_factors}")
                print("Adjusting factors to reasonable range...")
                zoom_factors = tuple(min(max(factor, 0.1), 10.0) for factor in zoom_factors)
                print(f"Adjusted factors: {zoom_factors}")
            
            print(f"Using scipy zoom with factors: {zoom_factors}")
            
            # Resize mean atlas
            try:
                resized_mean_atlas = ndimage.zoom(self.adc_mean_atlas_array, zoom_factors, order=1, mode='nearest')
                
                # Kontrola, zda výsledek má správnou velikost
                if resized_mean_atlas.shape != subject_size:
                    print(f"VAROVÁNÍ: Velikost výsledku {resized_mean_atlas.shape} se neshoduje s velikostí subjektu {subject_size}")
                    # Další ořezání nebo doplnění pro zajištění správné velikosti
                    pad_width = [(max(0, target - current), max(0, target - current)) 
                                for target, current in zip(subject_size, resized_mean_atlas.shape)]
                    resized_mean_atlas = np.pad(resized_mean_atlas, pad_width, mode='constant')
                    resized_mean_atlas = resized_mean_atlas[:subject_size[0], :subject_size[1], :subject_size[2]]
                
                # Kontrola NaN hodnot
                if np.any(np.isnan(resized_mean_atlas)):
                    print("VAROVÁNÍ: NaN hodnoty v registrovaném atlasu, nahrazuji mediánem")
                    nan_mask = np.isnan(resized_mean_atlas)
                    non_nan_values = resized_mean_atlas[~nan_mask]
                    if len(non_nan_values) > 0:
                        median_value = np.median(non_nan_values)
                        resized_mean_atlas[nan_mask] = median_value
                
                # Resize std atlas if available
                if self.adc_std_atlas_array is not None:
                    resized_std_atlas = ndimage.zoom(self.adc_std_atlas_array, zoom_factors, order=1, mode='nearest')
                    # Upravení velikosti, pokud je potřeba
                    if resized_std_atlas.shape != subject_size:
                        pad_width = [(max(0, target - current), max(0, target - current)) 
                                    for target, current in zip(subject_size, resized_std_atlas.shape)]
                        resized_std_atlas = np.pad(resized_std_atlas, pad_width, mode='constant')
                        resized_std_atlas = resized_std_atlas[:subject_size[0], :subject_size[1], :subject_size[2]]
                        
                    # Kontrola NaN hodnot
                    if np.any(np.isnan(resized_std_atlas)):
                        print("VAROVÁNÍ: NaN hodnoty v registrovaném atlasu std, nahrazuji mediánem")
                        nan_mask = np.isnan(resized_std_atlas)
                        non_nan_values = resized_std_atlas[~nan_mask]
                        if len(non_nan_values) > 0:
                            median_value = np.median(non_nan_values)
                            resized_std_atlas[nan_mask] = median_value
                else:
                    resized_std_atlas = None
                
                print(f"Successfully resized atlas to size: {resized_mean_atlas.shape}")
                return resized_mean_atlas, resized_std_atlas
                
            except Exception as resize_error:
                print(f"Error during scipy resize: {resize_error}")
                print("Unable to register atlas properly. Returning original atlas as fallback.")
                return self.adc_mean_atlas_array, self.adc_std_atlas_array
                
        except Exception as e:
            print(f"Error registering atlas to subject: {e}")
            print("Using unregistered atlas as fallback")
            return self.adc_mean_atlas_array, self.adc_std_atlas_array
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with input, target, and metadata
        """
        # Get patient ID and synthetic lesion ID
        patient_id, synth_lesion_id = self.sample_pairs[idx]
        
        # Construct file paths
        adc_path = self.adc_files[patient_id]
        label_path = self.label_files[patient_id]
        synth_lesion_path = os.path.join(self.synthetic_lesions_dir, patient_id, f"{synth_lesion_id}.nii.gz")
        
        # Load ADC and label files
        try:
            adc_img = sitk.ReadImage(adc_path)
            adc_array = sitk.GetArrayFromImage(adc_img)
            
            label_img = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label_img).astype(bool)
            
            synthetic_lesion_img = sitk.ReadImage(synth_lesion_path)
            synthetic_lesion_array = sitk.GetArrayFromImage(synthetic_lesion_img).astype(bool)
        except Exception as e:
            print(f"Error loading data for patient {patient_id}, lesion {synth_lesion_id}: {e}")
            # Return empty sample in case of error
            return {
                "input": torch.zeros((2, 32, 32, 32)),
                "target": torch.zeros((1, 32, 32, 32)),
                "metadata": {
                    "patient_id": patient_id,
                    "synthetic_lesion_id": synth_lesion_id,
                    "error": str(e)
                }
            }
            
        # Naměřit statistiky o obrazu
        adc_mean = np.mean(adc_array[adc_array > 0])  # Průměr nenulových hodnot
        adc_std = np.std(adc_array[adc_array > 0])    # Směrodatná odchylka nenulových hodnot
        
        # Vypočítat percentily pro detekci outlierů
        non_zero = adc_array[adc_array > 0]
        p01 = np.percentile(non_zero, 1) if len(non_zero) > 0 else 0
        p99 = np.percentile(non_zero, 99) if len(non_zero) > 0 else 1
        
        print(f"ADC stats - mean: {adc_mean:.4f}, std: {adc_std:.4f}, p01: {p01:.4f}, p99: {p99:.4f}")
        
        # Normalizace ADC obrazu pomocí percentilů
        adc_normalized = np.copy(adc_array)
        adc_normalized[adc_normalized < p01] = p01  # Oříznout spodní outliers
        adc_normalized[adc_normalized > p99] = p99  # Oříznout horní outliers
        adc_normalized = (adc_normalized - p01) / (p99 - p01)  # Normalizace na [0,1]
        
        # Vytvořit "skutečně zdravý mozek" pomocí atlasu ADC nebo jednoduchého nahrazení hodnot
        if self.has_atlas:
            print("Creating healthy brain using registered ADC atlas...")
            
            # Registrovat atlas k subjektu
            registered_atlas, registered_std_atlas = self.register_atlas_to_subject(adc_img, adc_array, label_array)
            
            # Kontrola registrace
            if registered_atlas is not None:
                # Normalizace registrovaného atlasu do stejného rozsahu jako ADC
                atlas_normalized = np.copy(registered_atlas)
                atlas_normalized[atlas_normalized < p01] = p01
                atlas_normalized[atlas_normalized > p99] = p99
                atlas_normalized = (atlas_normalized - p01) / (p99 - p01)
                
                # Vytvořit masku dilací existujících lézí
                kernel = np.ones((3,3,3))
                dilated_mask = ndimage.binary_dilation(label_array, structure=kernel, iterations=2)
                
                # Vytvořit hranici okolo léze (dilated - original)
                border_mask = dilated_mask & np.logical_not(label_array)
                
                # Vypočítat poměr mezi hodnotami subjektu a atlasu v hranici
                if np.sum(border_mask) > 0:
                    ratio = np.mean(adc_normalized[border_mask]) / np.mean(atlas_normalized[border_mask] + 1e-6)
                    print(f"Border ratio: {ratio:.4f}")
                    # Omezit extrémní hodnoty poměru
                    ratio = np.clip(ratio, 0.5, 2.0)
                else:
                    ratio = 1.0
                    print("No border found, using default ratio 1.0")
                
                # Upravit atlas podle poměru a vytvořit zdravý mozek
                healthy_brain = adc_normalized.copy()
                
                # Použít atlas pouze v oblasti léze
                healthy_brain[label_array] = atlas_normalized[label_array] * ratio
                
                # Vytvořit diagnostickou vizualizaci procesu
                create_registration_visualization(
                    adc_normalized, 
                    atlas_normalized, 
                    label_array, 
                    healthy_brain,
                    ratio,
                    os.path.join(self.output_dir, 'registration_vis', f"{patient_id}_{synth_lesion_id}_registration.png")
                )
                
                print("Healthy brain created using atlas registration")
            else:
                print("Atlas registration failed, falling back to basic method")
                # Fallback metoda - jednoduchá náhrada lézí
                healthy_brain = self.create_basic_healthy_brain(adc_normalized, label_array)
        else:
            # Pokud nemáme atlas, použijeme základní metodu
            print("No atlas available, using basic healthy brain creation")
            healthy_brain = self.create_basic_healthy_brain(adc_normalized, label_array)
        
        # Vypočítat překryv mezi existující a syntetickou lézí
        overlap = np.logical_and(label_array, synthetic_lesion_array)
        overlap_percentage = np.sum(overlap) / np.sum(synthetic_lesion_array) if np.sum(synthetic_lesion_array) > 0 else 0
        print(f"Overlap percentage: {overlap_percentage:.4f}")
        
        # Určit cílový výstup
        if overlap_percentage > 0.5:
            # Pokud je překryv významný, použijeme původní ADC jako cíl (obsahující reálnou lézi)
            target = adc_normalized
            print("Using original ADC as target (high overlap)")
        else:
            # Jinak vytvoříme hybridní cíl
            target = self.create_target_with_synthetic_lesion(healthy_brain, synthetic_lesion_array, adc_normalized)
            print("Using hybrid target with synthetic lesion")
        
        # Spojit zdravý mozek a syntetickou lézi jako vstup pro model
        model_input = np.stack([healthy_brain, synthetic_lesion_array.astype(np.float32)], axis=0)
        model_target = target[np.newaxis, ...]  # Add channel dimension
        
        # Konvertovat na tensory
        input_tensor = torch.from_numpy(model_input.astype(np.float32))
        target_tensor = torch.from_numpy(model_target.astype(np.float32))
        
        return {
            "input": input_tensor,
            "target": target_tensor,
            "metadata": {
                "patient_id": patient_id,
                "synthetic_lesion_id": synth_lesion_id,
                "overlap_percentage": overlap_percentage
            }
        }

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
         num_epochs=200, batch_size=4, save_interval=5):
    """
    Train the HIE lesion inpainting GAN.
    
    Args:
        adc_dir: Directory containing ADC maps
        label_dir: Directory containing lesion labels
        synthetic_lesions_dir: Directory containing synthetic lesions
        output_dir: Directory to save model checkpoints and results
        adc_mean_atlas_path: Path to the mean ADC atlas
        adc_std_atlas_path: Path to the standard deviation ADC atlas
        num_epochs: Number of epochs to train
        batch_size: Batch size
        save_interval: Interval (in epochs) for saving model checkpoints
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n===== VALIDATING DATASET PATHS =====")
    print(f"ADC directory: {adc_dir} - Exists: {os.path.exists(adc_dir)}")
    print(f"Label directory: {label_dir} - Exists: {os.path.exists(label_dir)}")
    print(f"Synthetic lesions directory: {synthetic_lesions_dir} - Exists: {os.path.exists(synthetic_lesions_dir)}")
    
    if adc_mean_atlas_path:
        print(f"ADC mean atlas: {adc_mean_atlas_path} - Exists: {os.path.exists(adc_mean_atlas_path)}")
    if adc_std_atlas_path:
        print(f"ADC std atlas: {adc_std_atlas_path} - Exists: {os.path.exists(adc_std_atlas_path)}")
    
    if os.path.exists(synthetic_lesions_dir):
        print(f"Contents of synthetic_lesions_dir: {os.listdir(synthetic_lesions_dir)[:10]}")
    print("=====================================\n")
    
    # Create datasets
    train_dataset = LesionInpaintingDataset(
        adc_dir=adc_dir,
        label_dir=label_dir,
        synthetic_lesions_dir=synthetic_lesions_dir,
        adc_mean_atlas_path=adc_mean_atlas_path,
        adc_std_atlas_path=adc_std_atlas_path,
        patch_size=(96, 96, 96),
        mode='train'
    )
    
    val_dataset = LesionInpaintingDataset(
        adc_dir=adc_dir,
        label_dir=label_dir,
        synthetic_lesions_dir=synthetic_lesions_dir,
        adc_mean_atlas_path=adc_mean_atlas_path,
        adc_std_atlas_path=adc_std_atlas_path,
        patch_size=(96, 96, 96),
        mode='val'
    )
    
    # Check if datasets have samples
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty! Cannot continue with training.\n"
            "Please check the following:\n"
            "1. The directories exist and contain the expected files\n"
            "2. The file naming conventions match those expected in the code\n"
            "3. The synthetic_lesions_dir has subdirectories named after patients\n"
            "4. Each patient directory contains synthetic lesion MHA files"
        )
    
    if len(val_dataset) == 0:
        print("WARNING: Validation dataset is empty. Training will continue without validation.")
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Initialize model
    gan_model = LesionInpaintingGAN(img_shape=(96, 96, 96))
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        epoch_losses = {
            'g_loss': 0,
            'adv_loss': 0,
            'l1_loss': 0,
            'context_loss': 0,
            'd_loss': 0
        }
        
        pbar = tqdm(train_loader)
        for batch in pbar:
            losses = gan_model.train_step(batch)
            
            # Update progress bar
            pbar.set_description(f"G: {losses['g_loss']:.4f}, D: {losses['d_loss']:.4f}")
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        train_losses.append(epoch_losses)
        
        # Validation
        if val_loader is not None:
            print("Validating...")
            val_epoch_losses = gan_model.validate(val_loader)
            val_losses.append(val_epoch_losses)
            print(f"Validation - G: {val_epoch_losses['g_loss']:.4f}, D: {val_epoch_losses['d_loss']:.4f}")
        
        # Update learning rates
        gan_model.scheduler_G.step()
        gan_model.scheduler_D.step()
        
        # Save checkpoint based on specified save_interval
        if (epoch + 1) % save_interval == 0:
            gan_model.save_models(output_dir, epoch + 1)
        
        # Generate visualizations after each epoch
        if val_loader is not None:
            with torch.no_grad():
                # Get a sample from validation set
                val_sample = next(iter(val_loader))
                input_data = val_sample['input'].to(gan_model.device)
                real_brain = val_sample['target'].to(gan_model.device)
                synthetic_mask = val_sample['synthetic_mask'].to(gan_model.device)
                
                # Generate inpainted image
                fake_brain = gan_model.generator(input_data)
                
                # Create detailed visualizations for each sample
                for i in range(min(2, input_data.shape[0])):
                    # Get current sample data
                    input_vol = input_data[i, 0].cpu().numpy()
                    mask_vol = synthetic_mask[i, 0].cpu().numpy()
                    real_vol = real_brain[i, 0].cpu().numpy()
                    fake_vol = fake_brain[i, 0].cpu().numpy()
                    
                    # Find slices containing the lesion
                    lesion_slices = []
                    for z in range(mask_vol.shape[0]):
                        if np.any(mask_vol[z] > 0):
                            lesion_slices.append(z)
                    
                    if not lesion_slices:
                        # If no lesion found, use middle slices
                        mid_slice = mask_vol.shape[0] // 2
                        lesion_slices = list(range(max(0, mid_slice-5), min(mask_vol.shape[0], mid_slice+6)))
                    
                    # We want ALL slices with lesion content
                    print(f"Visualizing all {len(lesion_slices)} slices with lesion content")
                    
                    # Calculate overall change in lesion area
                    lesion_area_values_before = input_vol[mask_vol > 0]
                    lesion_area_values_after = fake_vol[mask_vol > 0]
                    mean_value_before = np.mean(lesion_area_values_before) if len(lesion_area_values_before) > 0 else 0
                    mean_value_after = np.mean(lesion_area_values_after) if len(lesion_area_values_after) > 0 else 0
                    mean_abs_change = np.mean(np.abs(lesion_area_values_after - lesion_area_values_before)) if len(lesion_area_values_before) > 0 else 0
                    
                    # Create multi-page PDF to store all slices if there are too many
                    use_pdf = len(lesion_slices) > 15
                    if use_pdf:
                        pdf_filename = os.path.join(output_dir, f'epoch_{epoch+1:03d}_sample_{i}_all_slices.pdf')
                        pdf = PdfPages(pdf_filename)
                        
                        # Create multiple figures with max 15 slices per figure
                        for slice_batch_idx in range(0, len(lesion_slices), 15):
                            batch_slices = lesion_slices[slice_batch_idx:slice_batch_idx+15]
                            fig, axes = plt.subplots(4, len(batch_slices), figsize=(4*len(batch_slices), 16))
                            
                            # If only one slice, reshape axes for indexing
                            if len(batch_slices) == 1:
                                axes = axes.reshape(4, 1)
                                
                            for j, slice_idx in enumerate(batch_slices):
                                # Process slice as before...
                                # Create overlay of input with lesion mask for visualization
                                input_with_mask = np.stack([input_vol[slice_idx], input_vol[slice_idx], input_vol[slice_idx]], axis=2)
                                # Add red overlay for lesion
                                mask_overlay = mask_vol[slice_idx] > 0
                                input_with_mask[mask_overlay, 0] = 1.0  # Red channel
                                input_with_mask[mask_overlay, 1] = 0.0  # Green channel
                                input_with_mask[mask_overlay, 2] = 0.0  # Blue channel
                                
                                # Calculate changes for this particular slice
                                slice_mask = mask_vol[slice_idx] > 0
                                if np.any(slice_mask):
                                    slice_before = input_vol[slice_idx][slice_mask]
                                    slice_after = fake_vol[slice_idx][slice_mask]
                                    slice_mean_change = np.mean(np.abs(slice_after - slice_before))
                                    slice_title = f'Slice {slice_idx} (Δ={slice_mean_change:.4f})'
                                else:
                                    slice_title = f'Slice {slice_idx}'
                                
                                # Create difference map between fake and input to show changes
                                diff_map = np.abs(fake_vol[slice_idx] - input_vol[slice_idx])
                                
                                # Plot each slice
                                axes[0, j].imshow(input_vol[slice_idx], cmap='gray')
                                axes[0, j].set_title(slice_title)
                                axes[0, j].axis('off')
                                
                                axes[1, j].imshow(input_with_mask)
                                axes[1, j].set_title(f'Lesion Overlay')
                                axes[1, j].axis('off')
                                
                                axes[2, j].imshow(fake_vol[slice_idx], cmap='gray')
                                axes[2, j].set_title(f'Generated')
                                axes[2, j].axis('off')
                                
                                # Show difference map - where changes were made
                                axes[3, j].imshow(diff_map, cmap='hot')
                                axes[3, j].set_title(f'Change Map')
                                axes[3, j].axis('off')
                            
                            # Add row labels
                            axes[0, 0].set_ylabel('Input Volume')
                            axes[1, 0].set_ylabel('Lesion Location')
                            axes[2, 0].set_ylabel('Generated Result')
                            axes[3, 0].set_ylabel('Change Heatmap')
                            
                            # Add overall statistics to the figure
                            plt.suptitle(f'Epoch {epoch+1} - G:{epoch_losses["g_loss"]:.4f}, D:{epoch_losses["d_loss"]:.4f}\n'
                                         f'Mean value in lesion area: Before={mean_value_before:.4f}, After={mean_value_after:.4f}, Change={mean_abs_change:.4f}')
                            
                            plt.tight_layout()
                            pdf.savefig(fig)
                            plt.close()
                            
                        pdf.close()
                        print(f"Saved all {len(lesion_slices)} slices to {pdf_filename}")
                    else:
                        # If few enough slices, just create one image
                        num_slices = len(lesion_slices)
                        fig, axes = plt.subplots(4, num_slices, figsize=(4*num_slices, 16))
                        
                        # If only one slice, reshape axes for indexing
                        if num_slices == 1:
                            axes = axes.reshape(4, 1)
                        
                        for j, slice_idx in enumerate(lesion_slices):
                            # Create overlay of input with lesion mask for visualization
                            input_with_mask = np.stack([input_vol[slice_idx], input_vol[slice_idx], input_vol[slice_idx]], axis=2)
                            # Add red overlay for lesion
                            mask_overlay = mask_vol[slice_idx] > 0
                            input_with_mask[mask_overlay, 0] = 1.0  # Red channel
                            input_with_mask[mask_overlay, 1] = 0.0  # Green channel
                            input_with_mask[mask_overlay, 2] = 0.0  # Blue channel
                            
                            # Calculate changes for this particular slice
                            slice_mask = mask_vol[slice_idx] > 0
                            if np.any(slice_mask):
                                slice_before = input_vol[slice_idx][slice_mask]
                                slice_after = fake_vol[slice_idx][slice_mask]
                                slice_mean_change = np.mean(np.abs(slice_after - slice_before))
                                slice_title = f'Slice {slice_idx} (Δ={slice_mean_change:.4f})'
                            else:
                                slice_title = f'Slice {slice_idx}'
                            
                            # Create difference map between fake and input to show changes
                            diff_map = np.abs(fake_vol[slice_idx] - input_vol[slice_idx])
                            
                            # Plot each slice
                            axes[0, j].imshow(input_vol[slice_idx], cmap='gray')
                            axes[0, j].set_title(slice_title)
                            axes[0, j].axis('off')
                            
                            axes[1, j].imshow(input_with_mask)
                            axes[1, j].set_title(f'Lesion Overlay')
                            axes[1, j].axis('off')
                            
                            axes[2, j].imshow(fake_vol[slice_idx], cmap='gray')
                            axes[2, j].set_title(f'Generated')
                            axes[2, j].axis('off')
                            
                            # Show difference map - where changes were made
                            axes[3, j].imshow(diff_map, cmap='hot')
                            axes[3, j].set_title(f'Change Map')
                            axes[3, j].axis('off')
                        
                        # Add row labels
                        if num_slices > 0:
                            axes[0, 0].set_ylabel('Input Volume')
                            axes[1, 0].set_ylabel('Lesion Location')
                            axes[2, 0].set_ylabel('Generated Result')
                            axes[3, 0].set_ylabel('Change Heatmap')
                        
                        # Add overall statistics to the figure
                        plt.suptitle(f'Epoch {epoch+1} - G:{epoch_losses["g_loss"]:.4f}, D:{epoch_losses["d_loss"]:.4f}\n'
                                     f'Mean value in lesion area: Before={mean_value_before:.4f}, After={mean_value_after:.4f}, Change={mean_abs_change:.4f}')
                        
                        # Save with epoch number and sample number
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'epoch_{epoch+1:03d}_sample_{i}_full_volume.png'), dpi=150)
                        plt.close()
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    # Generator losses
    plt.subplot(2, 1, 1)
    plt.plot([loss['g_loss'] for loss in train_losses], label='G Total Loss (Train)')
    plt.plot([loss['adv_loss'] for loss in train_losses], label='G Adversarial Loss (Train)')
    plt.plot([loss['l1_loss'] for loss in train_losses], label='G L1 Loss (Train)')
    plt.plot([loss['context_loss'] for loss in train_losses], label='G Context Loss (Train)')
    
    if val_losses:
        plt.plot([loss['g_loss'] for loss in val_losses], '--', label='G Total Loss (Val)')
    
    plt.legend()
    plt.title('Generator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Discriminator losses
    plt.subplot(2, 1, 2)
    plt.plot([loss['d_loss'] for loss in train_losses], label='D Loss (Train)')
    
    if val_losses:
        plt.plot([loss['d_loss'] for loss in val_losses], '--', label='D Loss (Val)')
    
    plt.legend()
    plt.title('Discriminator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    return gan_model


def apply_inpainting(gan_model, adc_file, synthetic_lesion_file, output_file):
    """
    Apply the trained model to inpaint a lesion into a brain scan.
    
    Args:
        gan_model: Trained LesionInpaintingGAN model
        adc_file: Path to the ADC file (.mha)
        synthetic_lesion_file: Path to the synthetic lesion file (.mha)
        output_file: Path to save the output (.mha)
    """
    # Load ADC map
    adc_img = sitk.ReadImage(adc_file)
    adc_array = sitk.GetArrayFromImage(adc_img)
    
    # Load synthetic lesion
    syn_lesion_img = sitk.ReadImage(synthetic_lesion_file)
    syn_lesion_array = sitk.GetArrayFromImage(syn_lesion_img)
    
    # Normalize ADC to [0, 1]
    adc_min = adc_array.min()
    adc_max = adc_array.max()
    adc_array_norm = (adc_array - adc_min) / (adc_max - adc_min + 1e-8)
    
    # Convert synthetic lesion to binary
    syn_lesion_array = (syn_lesion_array > 0).astype(np.float32)
    
    # Convert to tensors
    adc_tensor = torch.from_numpy(adc_array_norm).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    syn_lesion_tensor = torch.from_numpy(syn_lesion_array).float().unsqueeze(0).unsqueeze(0)
    
    # Generate inpainted image
    inpainted_brain = gan_model.inference(adc_tensor, syn_lesion_tensor)
    
    # Convert back to numpy and denormalize
    inpainted_array = inpainted_brain.squeeze().cpu().numpy()
    inpainted_array = inpainted_array * (adc_max - adc_min) + adc_min
    
    # Create SimpleITK image (using original image as reference)
    output_img = sitk.GetImageFromArray(inpainted_array)
    output_img.CopyInformation(adc_img)  # Copy metadata
    
    # Save the output
    sitk.WriteImage(output_img, output_file)


def visualize_healthy_brain_creation(adc_array, label_array, healthy_brain, output_path, title=None):
    """
    Vizualizuje proces vytvoření zdravého mozku z původních dat s lézí.
    
    Args:
        adc_array: Původní ADC data (normalizovaná 0-1)
        label_array: Maska léze
        healthy_brain: Vytvořený zdravý mozek
        output_path: Cesta k uložení výstupu
        title: Titulek pro vizualizaci
    """
    # Najít řezy obsahující lézi
    lesion_slices = []
    for z in range(label_array.shape[0]):
        if np.any(label_array[z] > 0):
            lesion_slices.append(z)
    
    if not lesion_slices:
        print("Varování: Nenalezeny žádné řezy s lézí pro vizualizaci.")
        return
    
    # Výpočet statistik změn v oblasti léze
    mask_3d = label_array > 0
    original_values = adc_array[mask_3d]
    healthy_values = healthy_brain[mask_3d]
    
    mean_original = np.mean(original_values) if len(original_values) > 0 else 0
    mean_healthy = np.mean(healthy_values) if len(healthy_values) > 0 else 0
    mean_abs_diff = np.mean(np.abs(healthy_values - original_values)) if len(original_values) > 0 else 0
    
    stats_text = (f"Statistika oblasti léze:\n"
                 f"Průměrná hodnota původní: {mean_original:.4f}\n"
                 f"Průměrná hodnota zdravá: {mean_healthy:.4f}\n"
                 f"Průměrná absolutní změna: {mean_abs_diff:.4f}")
    
    use_pdf = len(lesion_slices) > 15
    
    if use_pdf:
        with PdfPages(output_path) as pdf:
            for slice_batch_idx in range(0, len(lesion_slices), 15):
                batch_slices = lesion_slices[slice_batch_idx:slice_batch_idx+15]
                fig, axs = plt.subplots(4, len(batch_slices), figsize=(4*len(batch_slices), 16))
                
                # Pro případ jednoho řezu
                if len(batch_slices) == 1:
                    axs = axs.reshape(4, 1)
                
                for j, slice_idx in enumerate(batch_slices):
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
    Vytvoří a vizualizuje proces tvorby zdravého mozku z ADC mapy s lézí za použití atlasu.
    
    Args:
        adc_file: Cesta k ADC souboru (.mha)
        label_file: Cesta k souboru s maskou léze (.mha)
        adc_mean_atlas_path: Cesta k průměrnému ADC atlasu
        adc_std_atlas_path: Cesta k atlasu směrodatných odchylek (volitelné)
        output_file: Cesta k výstupnímu souboru
    """
    print("Načítání dat...")
    
    # Načtení ADC mapy
    adc_img = sitk.ReadImage(adc_file)
    adc_array = sitk.GetArrayFromImage(adc_img)
    
    # Načtení masky léze
    label_img = sitk.ReadImage(label_file)
    label_array = sitk.GetArrayFromImage(label_img)
    
    # Normalizace ADC do [0, 1]
    adc_min = adc_array.min()
    adc_max = adc_array.max()
    adc_array_norm = (adc_array - adc_min) / (adc_max - adc_min + 1e-8)
    
    # Převod masky léze na binární
    label_array = (label_array > 0).astype(np.float32)
    
    # Načtení atlasů
    print("Načítání atlasů...")
    if adc_mean_atlas_path and os.path.exists(adc_mean_atlas_path):
        print(f"Načítání průměrného ADC atlasu: {adc_mean_atlas_path}")
        try:
            adc_mean_atlas = sitk.ReadImage(adc_mean_atlas_path)
            adc_mean_atlas_array = sitk.GetArrayFromImage(adc_mean_atlas)
            
            if adc_std_atlas_path and os.path.exists(adc_std_atlas_path):
                print(f"Načítání atlasu směrodatných odchylek: {adc_std_atlas_path}")
                adc_std_atlas = sitk.ReadImage(adc_std_atlas_path)
                adc_std_atlas_array = sitk.GetArrayFromImage(adc_std_atlas)
            else:
                print("VAROVÁNÍ: Atlas směrodatných odchylek nenalezen.")
                adc_std_atlas = None
                adc_std_atlas_array = None
                
            has_atlas = True
        except Exception as e:
            print(f"Chyba při načítání atlasů: {e}")
            has_atlas = False
    else:
        print("VAROVÁNÍ: Průměrný ADC atlas nenalezen.")
        return
    
    try:
        # Transformace atlasu na velikost pacienta
        print("Registrace atlasu k subjektu...")
        subject_size = adc_array.shape
        atlas_size = adc_mean_atlas_array.shape
        
        print(f"Velikost ADC subjektu: {subject_size}, Velikost atlasu: {atlas_size}")
        
        # Použití vylepšené metody převzorkování s kontrolou poměrů
        try:
            from scipy import ndimage
            
            # Vypočítáme poměry rozměrů pro převzorkování
            zoom_factors = (subject_size[0] / atlas_size[0], 
                           subject_size[1] / atlas_size[1],
                           subject_size[2] / atlas_size[2])
            
            print(f"Použití scipy zoom s faktory: {zoom_factors}")
            
            # Kontrola, zda nejsou faktory zoom příliš extrémní
            if max(zoom_factors) > 10 or min(zoom_factors) < 0.1:
                print(f"VAROVÁNÍ: Extrémní faktory pro zoom: {zoom_factors}")
                print("Upravuji faktory, aby byly v rozumném rozsahu...")
                zoom_factors = tuple(min(max(factor, 0.1), 10.0) for factor in zoom_factors)
                print(f"Upravené faktory: {zoom_factors}")
            
            # Registrace průměrného atlasu
            registered_atlas_array = ndimage.zoom(adc_mean_atlas_array, zoom_factors, order=1, mode='nearest')
            
            # Kontrola, zda výsledek má správnou velikost
            if registered_atlas_array.shape != subject_size:
                print(f"VAROVÁNÍ: Velikost registrovaného atlasu {registered_atlas_array.shape} se neshoduje s velikostí subjektu {subject_size}")
                # Další ořezání nebo doplnění pro zajištění správné velikosti
                pad_width = [(max(0, target - current), max(0, target - current)) 
                            for target, current in zip(subject_size, registered_atlas_array.shape)]
                registered_atlas_array = np.pad(registered_atlas_array, pad_width, mode='constant')
                registered_atlas_array = registered_atlas_array[:subject_size[0], :subject_size[1], :subject_size[2]]
            
            # Registrace atlasu standardních odchylek, pokud existuje
            if adc_std_atlas_array is not None:
                registered_std_atlas_array = ndimage.zoom(adc_std_atlas_array, zoom_factors, order=1, mode='nearest')
                # Upravení velikosti, pokud je potřeba
                if registered_std_atlas_array.shape != subject_size:
                    pad_width = [(max(0, target - current), max(0, target - current)) 
                                for target, current in zip(subject_size, registered_std_atlas_array.shape)]
                    registered_std_atlas_array = np.pad(registered_std_atlas_array, pad_width, mode='constant')
                    registered_std_atlas_array = registered_std_atlas_array[:subject_size[0], :subject_size[1], :subject_size[2]]
            else:
                registered_std_atlas_array = None
            
            print(f"Atlas úspěšně registrován na velikost: {registered_atlas_array.shape}")
        except Exception as e:
            print(f"Chyba při registraci atlasu: {e}")
            print("Není možné pokračovat bez správně registrovaného atlasu.")
            return
        
        # Vytvoření zdravého mozku
        print("Vytváření zdravého mozku...")
        healthy_brain = adc_array_norm.copy()
        
        # Normalizace atlasu
        atlas_min = np.nanmin(registered_atlas_array)
        atlas_max = np.nanmax(registered_atlas_array)
        registered_atlas_norm = (registered_atlas_array - atlas_min) / (atlas_max - atlas_min + 1e-8)
        
        # Vytvoření masky okraje léze
        from scipy import ndimage
        dilated_mask = ndimage.binary_dilation(label_array, iterations=2)
        border_mask = dilated_mask & np.logical_not(label_array)
        
        # Výpočet škálovacího faktoru pro atlas
        if np.any(border_mask):
            # Získání hodnot na hranici léze
            original_border_values = adc_array_norm[border_mask]
            atlas_border_values = registered_atlas_norm[border_mask]
            
            # Odstranění případných NaN hodnot
            original_border_values = original_border_values[~np.isnan(original_border_values)]
            atlas_border_values = atlas_border_values[~np.isnan(atlas_border_values)]
            
            if len(original_border_values) > 0 and len(atlas_border_values) > 0:
                # Výpočet mediánu pro robustnost
                orig_median = np.median(original_border_values)
                atlas_median = np.median(atlas_border_values)
                
                # Výpočet škálovacího faktoru, ošetření dělení nulou
                if atlas_median > 1e-6:
                    scaling_factor = orig_median / atlas_median
                else:
                    scaling_factor = 1.0
                    print("VAROVÁNÍ: Mediánová hodnota atlasu blízká nule, používám scaling_factor=1.0")
                
                print(f"Škálovací faktor pro atlas: {scaling_factor:.4f}")
                
                # Aplikace atlasových hodnot do oblasti léze
                mask_indices = np.where(label_array > 0)
                if len(mask_indices[0]) > 0:
                    # Kontrola, zda nejsou v atlasu NaN hodnoty v oblasti léze
                    atlas_values = registered_atlas_norm[mask_indices]
                    nan_indices = np.isnan(atlas_values)
                    
                    if np.any(nan_indices):
                        print(f"VAROVÁNÍ: {np.sum(nan_indices)} NaN hodnot v atlasu v oblasti léze, nahrazuji mediánem")
                        atlas_values[nan_indices] = atlas_median
                    
                    # Aplikace škálovaných hodnot
                    healthy_brain[mask_indices] = atlas_values * scaling_factor
                    
                    # Kontrola, zda nejsou ve výsledku NaN hodnoty
                    if np.any(np.isnan(healthy_brain)):
                        print("VAROVÁNÍ: NaN hodnoty ve výsledném 'zdravém mozku', nahrazuji mediánem")
                        nan_mask = np.isnan(healthy_brain)
                        healthy_brain[nan_mask] = orig_median
                    
                    # Zajištění hodnot v platném rozsahu
                    healthy_brain = np.clip(healthy_brain, 0, 1)
                    
                    print("Úspěšně vytvořen zdravý mozek s použitím atlasu")
                else:
                    print("VAROVÁNÍ: Žádné voxely v masce léze")
            else:
                print("VAROVÁNÍ: Prázdné hodnoty na hranici léze, použiji interpolaci")
                # Fallback na interpolaci
                temp_array = adc_array_norm.copy()
                temp_array[label_array > 0] = np.nan
                filled_array = ndimage.median_filter(np.nan_to_num(temp_array), size=5)
                healthy_brain[label_array > 0] = filled_array[label_array > 0]
        else:
            print("VAROVÁNÍ: Žádná hranice léze nenalezena, použiji interpolaci")
            # Fallback na interpolaci
            temp_array = adc_array_norm.copy()
            temp_array[label_array > 0] = np.nan
            filled_array = ndimage.median_filter(np.nan_to_num(temp_array), size=5)
            healthy_brain[label_array > 0] = filled_array[label_array > 0]
        
        # Výpočet statistik změn v oblasti léze
        mask_3d = label_array > 0
        if np.any(mask_3d):
            original_values = adc_array_norm[mask_3d]
            healthy_values = healthy_brain[mask_3d]
            
            # Odstranění případných NaN hodnot
            valid_mask = ~np.isnan(original_values) & ~np.isnan(healthy_values)
            original_values = original_values[valid_mask]
            healthy_values = healthy_values[valid_mask]
            
            if len(original_values) > 0:
                mean_original = np.mean(original_values)
                mean_healthy = np.mean(healthy_values)
                mean_abs_diff = np.mean(np.abs(healthy_values - original_values))
                median_abs_diff = np.median(np.abs(healthy_values - original_values))
                
                stats_text = (f"Statistika oblasti léze:\n"
                              f"Průměrná hodnota původní: {mean_original:.4f}\n"
                              f"Průměrná hodnota zdravá: {mean_healthy:.4f}\n"
                              f"Průměrná absolutní změna: {mean_abs_diff:.4f}\n"
                              f"Mediánová absolutní změna: {median_abs_diff:.4f}")
            else:
                stats_text = "Statistika oblasti léze: Nedostatek platných dat"
        else:
            stats_text = "Statistika oblasti léze: Žádné voxely v masce léze"
        
        # Vizualizace
        print("Vytvářím vizualizaci...")
        
        # Najít řezy obsahující lézi
        lesion_slices = []
        for z in range(label_array.shape[0]):
            if np.any(label_array[z] > 0):
                lesion_slices.append(z)
        
        if not lesion_slices:
            print("VAROVÁNÍ: Nenalezeny žádné řezy s lézí pro vizualizaci.")
            # Použijeme prostřední řez jako zálohu
            lesion_slices = [label_array.shape[0] // 2]
        
        # Středový řez s lézí
        mid_slice_idx = lesion_slices[len(lesion_slices) // 2]
        
        # Detailní vizualizace jednoho řezu
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            
        # První řádek - ADC řez, atlas řez, rozdíl
        axs[0, 0].imshow(adc_array_norm[mid_slice_idx], cmap='gray', vmin=0, vmax=1)
        axs[0, 0].set_title(f'Původní ADC (řez {mid_slice_idx})')
        axs[0, 0].axis('off')
        
        # Zobrazení registrovaného atlasu s ošetřením NaN hodnot
        atlas_display = np.copy(registered_atlas_norm[mid_slice_idx])
        if np.any(np.isnan(atlas_display)):
            print(f"VAROVÁNÍ: NaN hodnoty v zobrazení atlasu, nahrazuji nulami")
            atlas_display = np.nan_to_num(atlas_display)
        
        axs[0, 1].imshow(atlas_display, cmap='gray', vmin=0, vmax=1)
        axs[0, 1].set_title('Registrovaný atlas')
        axs[0, 1].axis('off')
        
        # Rozdíl také s ošetřením NaN hodnot
        diff_atlas = np.abs(adc_array_norm[mid_slice_idx] - atlas_display)
        if np.any(np.isnan(diff_atlas)):
            diff_atlas = np.nan_to_num(diff_atlas)
        
        axs[0, 2].imshow(diff_atlas, cmap='hot', vmin=0, vmax=1)
        axs[0, 2].set_title('Rozdíl ADC vs. atlas')
        axs[0, 2].axis('off')
        
        # Druhý řádek - maska léze, zdravý mozek, rozdíl
        # Maska léze jako překryv
        mask = label_array[mid_slice_idx] > 0
        input_with_mask = np.stack([adc_array_norm[mid_slice_idx], 
                                   adc_array_norm[mid_slice_idx], 
                                   adc_array_norm[mid_slice_idx]], axis=2)
        input_with_mask[mask, 0] = 1.0  # Červený kanál
        input_with_mask[mask, 1] = 0.0  # Zelený kanál
        input_with_mask[mask, 2] = 0.0  # Modrý kanál
        
        axs[1, 0].imshow(input_with_mask)
        axs[1, 0].set_title('ADC s označením léze')
        axs[1, 0].axis('off')
        
        # Zdravý mozek - zajištění, že nemá NaN hodnoty
        healthy_display = np.copy(healthy_brain[mid_slice_idx])
        if np.any(np.isnan(healthy_display)):
            healthy_display = np.nan_to_num(healthy_display)
        
        axs[1, 1].imshow(healthy_display, cmap='gray', vmin=0, vmax=1)
        axs[1, 1].set_title('Zdravý mozek (bez léze)')
        axs[1, 1].axis('off')
        
        # Mapa změn - rozdíl mezi původním a zdravým mozkem
        diff_healthy = np.abs(adc_array_norm[mid_slice_idx] - healthy_display)
        if np.any(np.isnan(diff_healthy)):
            diff_healthy = np.nan_to_num(diff_healthy)
        
        im = axs[1, 2].imshow(diff_healthy, cmap='hot', vmin=0, vmax=np.max(diff_healthy) if np.max(diff_healthy) > 0 else 1)
        axs[1, 2].set_title('Mapa změn')
        axs[1, 2].axis('off')
        
        # Přidat colorbar pro mapu změn
        cbar = fig.colorbar(im, ax=axs[1, 2], fraction=0.046, pad=0.04)
        cbar.set_label('Absolutní rozdíl intenzity')
        
        plt.suptitle(f"Detailní srovnání atlasové metody pro řez {mid_slice_idx}\n{stats_text}")
        plt.tight_layout()
        
        # Uložit detail jako druhou vizualizaci
        detail_output = os.path.splitext(output_file)[0] + "_detail.png"
        plt.savefig(detail_output, dpi=150)
        plt.close()
        
        print(f"Detailní vizualizace uložena jako: {detail_output}")
        
        # Vytvořit vizualizaci pro všechny řezy s lézí
        use_pdf = len(lesion_slices) > 15
        
        if use_pdf:
            with PdfPages(output_file) as pdf:
                for slice_batch_idx in range(0, len(lesion_slices), 15):
                    batch_slices = lesion_slices[slice_batch_idx:slice_batch_idx+15]
                    fig, axs = plt.subplots(4, len(batch_slices), figsize=(4*len(batch_slices), 16))
                    
                    # Pro případ jednoho řezu
                    if len(batch_slices) == 1:
                        axs = axs.reshape(4, 1)
                    
                    for j, slice_idx in enumerate(batch_slices):
                        # Původní ADC řez
                        axs[0, j].imshow(adc_array_norm[slice_idx], cmap='gray', vmin=0, vmax=1)
                        axs[0, j].set_title(f'Řez {slice_idx}')
                        axs[0, j].axis('off')
                        
                        # Maska léze jako překryv
                        mask = label_array[slice_idx] > 0
                        input_with_mask = np.stack([adc_array_norm[slice_idx], 
                                                  adc_array_norm[slice_idx], 
                                                  adc_array_norm[slice_idx]], axis=2)
                        input_with_mask[mask, 0] = 1.0  # Červený kanál
                        input_with_mask[mask, 1] = 0.0  # Zelený kanál
                        input_with_mask[mask, 2] = 0.0  # Modrý kanál
                        
                        axs[1, j].imshow(input_with_mask)
                        axs[1, j].set_title('Označení léze')
                        axs[1, j].axis('off')
                        
                        # Zobrazit zdravý mozek
                        healthy_display = np.copy(healthy_brain[slice_idx])
                        if np.any(np.isnan(healthy_display)):
                            healthy_display = np.nan_to_num(healthy_display)
                        
                        axs[2, j].imshow(healthy_display, cmap='gray', vmin=0, vmax=1)
                        axs[2, j].set_title('Zdravý mozek')
                        axs[2, j].axis('off')
                        
                        # Mapa změn
                        diff_healthy = np.abs(adc_array_norm[slice_idx] - healthy_display)
                        if np.any(np.isnan(diff_healthy)):
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
                    
                    # Přidat popisky řad
                    axs[0, 0].set_ylabel('Původní ADC')
                    axs[1, 0].set_ylabel('Léze')
                    axs[2, 0].set_ylabel('Zdravý mozek')
                    axs[3, 0].set_ylabel('Změny')
                    
                    # Celkový titulek
                    plt.suptitle(f"Vizualizace vytvoření zdravého mozku pomocí atlasu\n{stats_text}")
                        
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=150)
                    plt.close()
            
            print(f"Uložena kompletní vizualizace tvorby zdravého mozku do PDF: {output_file}")
        else:
            # Jeden obrázek pro méně než 15 řezů
            fig, axs = plt.subplots(4, len(lesion_slices), figsize=(4*len(lesion_slices), 16))
            
            # Pro případ jednoho řezu
            if len(lesion_slices) == 1:
                axs = axs.reshape(4, 1)
            
            for j, slice_idx in enumerate(lesion_slices):
                # Původní ADC řez
                axs[0, j].imshow(adc_array_norm[slice_idx], cmap='gray', vmin=0, vmax=1)
                axs[0, j].set_title(f'Řez {slice_idx}')
                axs[0, j].axis('off')
                
                # Maska léze jako překryv
                mask = label_array[slice_idx] > 0
                input_with_mask = np.stack([adc_array_norm[slice_idx], 
                                          adc_array_norm[slice_idx], 
                                          adc_array_norm[slice_idx]], axis=2)
                input_with_mask[mask, 0] = 1.0  # Červený kanál
                input_with_mask[mask, 1] = 0.0  # Zelený kanál
                input_with_mask[mask, 2] = 0.0  # Modrý kanál
                
                axs[1, j].imshow(input_with_mask)
                axs[1, j].set_title('Označení léze')
                axs[1, j].axis('off')
                
                # Zobrazit zdravý mozek
                healthy_display = np.copy(healthy_brain[slice_idx])
                if np.any(np.isnan(healthy_display)):
                    healthy_display = np.nan_to_num(healthy_display)
                
                axs[2, j].imshow(healthy_display, cmap='gray', vmin=0, vmax=1)
                axs[2, j].set_title('Zdravý mozek')
                axs[2, j].axis('off')
                
                # Mapa změn
                diff_healthy = np.abs(adc_array_norm[slice_idx] - healthy_display)
                if np.any(np.isnan(diff_healthy)):
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
            
            # Přidat popisky řad
            axs[0, 0].set_ylabel('Původní ADC')
            axs[1, 0].set_ylabel('Léze')
            axs[2, 0].set_ylabel('Zdravý mozek')
            axs[3, 0].set_ylabel('Změny')
            
            # Celkový titulek
            plt.suptitle(f"Vizualizace vytvoření zdravého mozku pomocí atlasu\n{stats_text}")
                
            plt.tight_layout()
            plt.savefig(output_file, dpi=150)
            plt.close()
            
            print(f"Uložena kompletní vizualizace tvorby zdravého mozku: {output_file}")
        
    except Exception as e:
        print(f"Chyba při vytváření vizualizace: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    # Vytvoříme hlavní parser pro příkazovou řádku
    parser = argparse.ArgumentParser(description='HIE Lesion Inpainting GAN')
    
    # Vytvoříme podparsery pro různé příkazy (train, generate)
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parser pro příkaz "train"
    train_parser = subparsers.add_parser('train', help='Train the HIE lesion inpainting GAN model')
    train_parser.add_argument('--adc_dir', type=str, default="data/BONBID2023_Train/2Z_ADC",
                       help='Directory containing ADC maps')
    train_parser.add_argument('--label_dir', type=str, default="data/BONBID2023_Train/3LABEL",
                       help='Directory containing lesion label masks')
    train_parser.add_argument('--synthetic_lesions_dir', type=str, default="data/registered_lesions",
                       help='Directory containing synthetic lesions')
    train_parser.add_argument('--output_dir', type=str, default="output/lesion_inpainting",
                       help='Directory to save model checkpoints and results')
    train_parser.add_argument('--adc_mean_atlas_path', type=str, default=None,
                       help='Path to the mean ADC atlas for healthy brain approximation')
    train_parser.add_argument('--adc_std_atlas_path', type=str, default=None,
                       help='Path to the standard deviation ADC atlas')
    train_parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of epochs to train')
    train_parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    train_parser.add_argument('--save_interval', type=int, default=5,
                       help='Interval (in epochs) for saving model checkpoints')
    train_parser.add_argument('--test_patient', type=str, default="MGHNICU_445-VISIT_01",
                       help='Patient ID to use for test inference after training')
    train_parser.add_argument('--test_lesion', type=str, default="registered_lesion_sample33.mha",
                       help='Synthetic lesion file to use for test inference')
    train_parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training (cuda or cpu). Uses cuda if available by default.')
    train_parser.add_argument('--visualize_healthy_brain', action='store_true',
                       help='Generate detailed visualizations of healthy brain creation process')
    
    # Parser pro příkaz "generate"
    generate_parser = subparsers.add_parser('generate', help='Generate inpainted brain images using a trained model')
    generate_parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    generate_parser.add_argument('--adc_file', type=str, required=True,
                       help='Path to ADC MHA file to inpaint')
    generate_parser.add_argument('--lesion_file', type=str, required=True,
                       help='Path to synthetic lesion mask MHA file')
    generate_parser.add_argument('--output_file', type=str, required=True,
                       help='Path where to save output inpainted MHA file')
    generate_parser.add_argument('--device', type=str, default=None,
                       help='Device to use for inference (cuda or cpu). Uses cuda if available by default.')
    
    # Také přidám novou komandu pro samostatnou vizualizaci
    visualize_parser = subparsers.add_parser('visualize_healthy_brain', help='Visualize the process of creating healthy brain from ADC map using atlas')
    visualize_parser.add_argument('--adc_file', type=str, required=True,
                       help='Path to ADC MHA file (with lesion)')
    visualize_parser.add_argument('--label_file', type=str, required=True,
                       help='Path to lesion mask MHA file')
    visualize_parser.add_argument('--adc_mean_atlas_path', type=str, required=True,
                       help='Path to the mean ADC atlas')
    visualize_parser.add_argument('--adc_std_atlas_path', type=str, default=None,
                       help='Path to the standard deviation ADC atlas')
    visualize_parser.add_argument('--output_file', type=str, required=True,
                       help='Path where to save visualization')
    
    # Zpracujeme argumenty
    args = parser.parse_args()
    
    # Zpracování příkazu "train"
    if args.command == 'train':
        print(f"Starting training with the following parameters:")
        print(f"ADC directory: {args.adc_dir}")
        print(f"Label directory: {args.label_dir}")
        print(f"Synthetic lesions directory: {args.synthetic_lesions_dir}")
        print(f"Output directory: {args.output_dir}")
        
        if args.adc_mean_atlas_path:
            print(f"ADC mean atlas: {args.adc_mean_atlas_path}")
        else:
            print("No ADC mean atlas provided, will use interpolation")
            
        if args.adc_std_atlas_path:
            print(f"ADC std atlas: {args.adc_std_atlas_path}")
        else:
            print("No ADC std atlas provided")
            
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Save interval: Every {args.save_interval} epochs")
        
        # Trénujeme model s argumenty z příkazové řádky
        gan_model = train(
            adc_dir=args.adc_dir,
            label_dir=args.label_dir,
            synthetic_lesions_dir=args.synthetic_lesions_dir,
            output_dir=args.output_dir,
            adc_mean_atlas_path=args.adc_mean_atlas_path,
            adc_std_atlas_path=args.adc_std_atlas_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            save_interval=args.save_interval
        )
        
        print(f"Training completed. Running inference on test patient {args.test_patient}")
        
        # Provádění inference s natrénovaným modelem na testovacím pacientovi
        adc_file = os.path.join(args.adc_dir, f"Zmap_{args.test_patient}-ADC_smooth2mm_clipped10.mha")
        synthetic_lesion_file = os.path.join(args.synthetic_lesions_dir, args.test_patient, args.test_lesion)
        output_file = os.path.join(args.output_dir, f"{args.test_patient}_inpainted.mha")
        
        apply_inpainting(gan_model, adc_file, synthetic_lesion_file, output_file)
        print(f"Inference completed. Output saved to {output_file}")
    
    # Zpracování příkazu "generate"
    elif args.command == 'generate':
        print(f"Running inference with the following parameters:")
        print(f"Model checkpoint: {args.model_path}")
        print(f"ADC file: {args.adc_file}")
        print(f"Synthetic lesion file: {args.lesion_file}")
        print(f"Output file: {args.output_file}")
        
        # Inicializace modelu
        device = args.device
        model = LesionInpaintingGAN(device=device)
        
        # Načtení modelu z checkpointu
        print(f"Loading model from checkpoint: {args.model_path}")
        epoch = model.load_models(args.model_path)
        print(f"Loaded model from epoch {epoch}")
        
        # Inference - aplikace inpainting
        print(f"Running inference...")
        apply_inpainting(model, args.adc_file, args.lesion_file, args.output_file)
        print(f"Inference completed. Output saved to {args.output_file}")
    
    # Zpracování příkazu "visualize_healthy_brain"
    elif args.command == 'visualize_healthy_brain':
        print(f"Visualizing healthy brain creation process for:")
        print(f"ADC file: {args.adc_file}")
        print(f"Lesion mask file: {args.label_file}")
        print(f"Output file: {args.output_file}")
        
        # Vytvoření a vizualizace zdravého mozku
        create_and_visualize_healthy_brain(args.adc_file, args.label_file, args.adc_mean_atlas_path, args.adc_std_atlas_path, args.output_file)
    
    else:
        parser.print_help()
