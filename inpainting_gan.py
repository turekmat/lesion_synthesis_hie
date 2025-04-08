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
                 output_dir,
                 adc_mean_atlas_path=None,
                 adc_std_atlas_path=None,
                 patch_size=(96, 96, 96),
                 mode='train',
                 transform=None):
        """
        Inicializuje dataset pro inpainting lézí.
        
        Args:
            adc_dir: Adresář s ADC mapami
            label_dir: Adresář s maskami reálných lézí
            synthetic_lesions_dir: Adresář se syntetickými lézemi
            output_dir: Adresář pro ukládání výsledků
            adc_mean_atlas_path: Cesta k průměrnému ADC atlasu (volitelné)
            adc_std_atlas_path: Cesta k atlasu směrodatných odchylek ADC (volitelné)
            patch_size: Velikost výstupního patche (pro trénink)
            mode: 'train' nebo 'validation'
            transform: Transformace aplikované na data
        """
        self.adc_dir = adc_dir
        self.label_dir = label_dir
        self.synthetic_lesions_dir = synthetic_lesions_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.mode = mode
        self.transform = transform
        
        # Načíst atlasy, pokud jsou k dispozici
        self.has_atlas = False
        self.adc_mean_atlas = None
        self.adc_std_atlas = None
        
        if adc_mean_atlas_path is not None and os.path.exists(adc_mean_atlas_path):
            try:
                # Načíst atlas průměrných hodnot ADC
                print(f"Načítám průměrný ADC atlas: {adc_mean_atlas_path}")
                atlas_img = sitk.ReadImage(adc_mean_atlas_path)
                self.adc_mean_atlas = sitk.GetArrayFromImage(atlas_img)
                
                # Kontrola, zda atlas obsahuje NaN hodnoty
                if np.isnan(self.adc_mean_atlas).any():
                    print("VAROVÁNÍ: Průměrný atlas obsahuje NaN hodnoty, nahrazuji nulami")
                    self.adc_mean_atlas = np.nan_to_num(self.adc_mean_atlas)
                
                self.has_atlas = True
                print(f"Atlas načten, velikost: {self.adc_mean_atlas.shape}")
                
                # Vytvořit adresář pro vizualizace registrace
                os.makedirs(os.path.join(self.output_dir, 'registration_vis'), exist_ok=True)
                
                # Načíst atlas směrodatných odchylek, pokud existuje
                if adc_std_atlas_path is not None and os.path.exists(adc_std_atlas_path):
                    print(f"Načítám atlas směrodatných odchylek ADC: {adc_std_atlas_path}")
                    std_atlas_img = sitk.ReadImage(adc_std_atlas_path)
                    self.adc_std_atlas = sitk.GetArrayFromImage(std_atlas_img)
                    
                    # Kontrola, zda atlas směrodatných odchylek obsahuje NaN hodnoty
                    if np.isnan(self.adc_std_atlas).any():
                        print("VAROVÁNÍ: Atlas směrodatných odchylek obsahuje NaN hodnoty, nahrazuji nulami")
                        self.adc_std_atlas = np.nan_to_num(self.adc_std_atlas)
                        
                    print(f"Atlas směrodatných odchylek načten, velikost: {self.adc_std_atlas.shape}")
                else:
                    print("Atlas směrodatných odchylek není k dispozici")
            except Exception as e:
                print(f"Chyba při načítání atlasů: {e}")
                self.has_atlas = False
        else:
            print("Atlas ADC není k dispozici, bude použita základní metoda vytváření zdravého mozku")
        
        # Najít všechny soubory ADC map a masek lézí
        adc_pattern = os.path.join(adc_dir, "*.nii.gz")
        label_pattern = os.path.join(label_dir, "*.nii.gz")
        
        adc_files = sorted(glob.glob(adc_pattern))
        label_files = sorted(glob.glob(label_pattern))
        
        # Extrahovat ID pacientů z názvů souborů ADC
        self.adc_files = {}
        for f in adc_files:
            patient_id = os.path.basename(f).split(".")[0]  # Předpokládáme formát ID.nii.gz
            self.adc_files[patient_id] = f
        
        # Přiřadit soubory masek k ID pacientů
        self.label_files = {}
        for f in label_files:
            patient_id = os.path.basename(f).split(".")[0]
            if patient_id in self.adc_files:  # Jen pacienti, kteří mají ADC
                self.label_files[patient_id] = f
        
        # Zkontrolovat, kteří pacienti mají syntetické léze
        valid_patient_ids = []
        for patient_id in self.adc_files.keys():
            if patient_id in self.label_files:
                synth_dir = os.path.join(synthetic_lesions_dir, patient_id)
                if os.path.exists(synth_dir):
                    lesion_files = glob.glob(os.path.join(synth_dir, "*.nii.gz"))
                    if len(lesion_files) > 0:
                        valid_patient_ids.append(patient_id)
        
        print(f"Nalezeno {len(valid_patient_ids)} platných pacientů s ADC, maskami lézí a syntetickými lézemi")
        
        # Rozdělit data na trénovací a validační množinu
        np.random.seed(42)  # Pro reprodukovatelné rozdělení
        np.random.shuffle(valid_patient_ids)
        split_idx = int(len(valid_patient_ids) * 0.8)  # 80% trénovací, 20% validační
        
        if mode == 'train':
            self.patient_ids = valid_patient_ids[:split_idx]
        else:  # validation
            self.patient_ids = valid_patient_ids[split_idx:]
        
        print(f"Používám {len(self.patient_ids)} pacientů pro {mode}")
        
        # Vytvořit páry vzorků (pacient_id, synth_lesion_id)
        self.sample_pairs = []
        for patient_id in self.patient_ids:
            synth_dir = os.path.join(synthetic_lesions_dir, patient_id)
            lesion_files = glob.glob(os.path.join(synth_dir, "*.nii.gz"))
            for lesion_file in lesion_files:
                # Extrahovat ID syntetické léze z názvu souboru
                lesion_id = os.path.basename(lesion_file).split(".")[0]
                self.sample_pairs.append((patient_id, lesion_id))
        
        print(f"Vytvořeno {len(self.sample_pairs)} trénovacích/validačních párů")
    
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
    Trénuje model pro inpainting lézí.
    
    Args:
        adc_dir: Adresář s ADC mapami
        label_dir: Adresář s maskami lézí
        synthetic_lesions_dir: Adresář se syntetickými lézemi
        output_dir: Adresář pro ukládání výsledků
        adc_mean_atlas_path: Cesta k průměrnému ADC atlasu
        adc_std_atlas_path: Cesta k atlasu směrodatných odchylek ADC
        num_epochs: Počet trénovacích epoch
        batch_size: Velikost dávky
        save_interval: Interval (v epochách) pro ukládání kontrolních bodů
    """
    print(f"Spouštím trénink s parametry:")
    print(f"- ADC dir: {adc_dir}")
    print(f"- Label dir: {label_dir}")
    print(f"- Synthetic lesions dir: {synthetic_lesions_dir}")
    print(f"- Output dir: {output_dir}")
    print(f"- ADC mean atlas: {adc_mean_atlas_path}")
    print(f"- ADC std atlas: {adc_std_atlas_path}")
    print(f"- Počet epoch: {num_epochs}")
    print(f"- Velikost dávky: {batch_size}")
    
    # Kontrola existence adresářů
    for dir_path in [adc_dir, label_dir, synthetic_lesions_dir]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Adresář neexistuje: {dir_path}")
    
    # Kontrola atlasů
    if adc_mean_atlas_path is not None and not os.path.exists(adc_mean_atlas_path):
        raise ValueError(f"ADC atlas neexistuje: {adc_mean_atlas_path}")
    
    if adc_std_atlas_path is not None and not os.path.exists(adc_std_atlas_path):
        raise ValueError(f"ADC atlas směrodatných odchylek neexistuje: {adc_std_atlas_path}")
    
    # Vytvořit výstupní adresář, pokud neexistuje
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Vytvořen výstupní adresář: {output_dir}")
    
    # Vytvořit adresáře pro checkpoint a vizualizace
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    vis_dir = os.path.join(output_dir, 'visualizations')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Nastavit zařízení (GPU, pokud je k dispozici)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Používám zařízení: {device}")
    
    # Vytvořit datové sady
    train_dataset = LesionInpaintingDataset(
        adc_dir=adc_dir,
        label_dir=label_dir,
        synthetic_lesions_dir=synthetic_lesions_dir,
        output_dir=output_dir,
        adc_mean_atlas_path=adc_mean_atlas_path,
        adc_std_atlas_path=adc_std_atlas_path,
        mode='train'
    )
    
    val_dataset = LesionInpaintingDataset(
        adc_dir=adc_dir,
        label_dir=label_dir,
        synthetic_lesions_dir=synthetic_lesions_dir,
        output_dir=output_dir,
        adc_mean_atlas_path=adc_mean_atlas_path,
        adc_std_atlas_path=adc_std_atlas_path,
        mode='val'
    )
    
    # Vytvořit datové loadery
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Trénovací dataset: {len(train_dataset)} vzorků")
    print(f"Validační dataset: {len(val_dataset)} vzorků")
    
    # Inicializovat GAN model
    gan_model = LesionInpaintingGAN(device=device)
    
    # Trénovat model
    print("Zahajuji trénink modelu...")
    
    # Metriky pro monitorování tréninku
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpocha {epoch}/{num_epochs}")
        
        # Trénink
        gan_model.train_on_loader(train_loader)
        
        # Validace
        val_loss = gan_model.validate(val_loader)
        
        # Ukládat metriky
        train_losses.append(gan_model.last_g_loss)
        val_losses.append(val_loss)
        
        # Reportovat metody
        print(f"G loss: {gan_model.last_g_loss:.4f}, D loss: {gan_model.last_d_loss:.4f}, Val loss: {val_loss:.4f}")
        
        # Ukládat model na intervalu a také nejlepší model
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"gan_epoch_{epoch}.pt")
            gan_model.save_models(checkpoint_path)
            print(f"Model uložen do: {checkpoint_path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "gan_best.pt")
            gan_model.save_models(best_model_path)
            print(f"Nový nejlepší model! Uložen jako: {best_model_path}")
        
        # Generovat vizualizace pro monitorování pokroku
        if epoch % 5 == 0 or epoch == 1:
            # Vybrat náhodný batch pro vizualizaci
            for i, batch in enumerate(val_loader):
                if i > 0:  # Jen první batch
                    break
                    
                inputs, targets = batch['input'].to(device), batch['target'].to(device)
                generated = gan_model.generator(inputs)
                
                # Denormalizovat a převést na numpy
                input_healthy = inputs[:, 0].detach().cpu().numpy()  # Kanál 0 - zdravý mozek
                input_mask = inputs[:, 1].detach().cpu().numpy()     # Kanál 1 - maska léze
                target_imgs = targets.detach().cpu().numpy()
                generated_imgs = generated.detach().cpu().numpy()
                
                # Vizualizovat jeden vzorek z batche
                for j in range(min(3, inputs.size(0))):  # Maximálně 3 vzorky
                    vis_path = os.path.join(vis_dir, f"epoch_{epoch}_sample_{j}.png")
                    
                    # Vypočítat rozdíl - mapa změn
                    diff_map = np.abs(generated_imgs[j, 0] - input_healthy[j])
                    
                    # Vytvořit vizualizaci
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    axes = axes.flat
                    
                    # Najít střední řez s nejvíce lézemi
                    lesion_sum = np.sum(input_mask[j], axis=(1, 2))
                    middle_slice = np.argmax(lesion_sum) if np.max(lesion_sum) > 0 else input_mask[j].shape[0] // 2
                    
                    # Zobrazit zdravý mozek
                    axes[0].imshow(input_healthy[j, middle_slice], cmap='gray')
                    axes[0].set_title('Zdravý mozek (vstup)')
                    axes[0].axis('off')
                    
                    # Zobrazit masku léze
                    axes[1].imshow(input_mask[j, middle_slice], cmap='hot')
                    axes[1].set_title('Maska léze (vstup)')
                    axes[1].axis('off')
                    
                    # Zobrazit cílový výstup
                    axes[2].imshow(target_imgs[j, 0, middle_slice], cmap='gray')
                    axes[2].set_title('Cílový výstup')
                    axes[2].axis('off')
                    
                    # Zobrazit generovaný výstup
                    axes[3].imshow(generated_imgs[j, 0, middle_slice], cmap='gray')
                    axes[3].set_title('Generovaný výstup')
                    axes[3].axis('off')
                    
                    # Zobrazit rozdíl
                    axes[4].imshow(diff_map[middle_slice], cmap='hot')
                    axes[4].set_title('Mapa změn')
                    axes[4].axis('off')
                    
                    # Zobrazit kombinaci (generovaný + kontura léze)
                    axes[5].imshow(generated_imgs[j, 0, middle_slice], cmap='gray')
                    
                    # Přidat kontury léze
                    from scipy import ndimage
                    contours = ndimage.binary_dilation(input_mask[j, middle_slice] > 0.5) ^ (input_mask[j, middle_slice] > 0.5)
                    masked_data = np.ma.masked_where(~contours, np.ones_like(contours))
                    axes[5].imshow(masked_data, cmap='autumn', alpha=1.0)
                    axes[5].set_title('Generovaný + kontura léze')
                    axes[5].axis('off')
                    
                    plt.suptitle(f"Epoch {epoch} - Vzorek {j} - Řez {middle_slice}", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(vis_path, dpi=150)
                    plt.close(fig)
                    
                    print(f"Uložena vizualizace: {vis_path}")
    
    # Uložit finální model
    final_model_path = os.path.join(checkpoint_dir, "gan_final.pt")
    gan_model.save_models(final_model_path)
    print(f"Trénink dokončen. Finální model uložen jako: {final_model_path}")
    
    # Vytvořit a uložit graf vývoje loss funkcí
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Generator Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()
    
    return final_model_path


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
            max_slices = 8  # Maximální počet řezů pro přehlednost
            if len(lesion_slices) > max_slices:
                # Vybrat rovnoměrně rozložené řezy
                indices = np.round(np.linspace(0, len(lesion_slices) - 1, max_slices)).astype(int)
                lesion_slices = [lesion_slices[i] for i in indices]
                print(f"### DEBUG: Limited visualization to {max_slices} slices: {lesion_slices}")
            
            n_rows = 4  # 4 řádky (orig, label, healthy, diff)
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
                
                # Zdravý mozek
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
            
            # Přidat popisky řad
            axs[0, 0].set_ylabel('Původní ADC')
            axs[1, 0].set_ylabel('Léze')
            axs[2, 0].set_ylabel('Zdravý mozek')
            axs[3, 0].set_ylabel('Změny')
            
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
