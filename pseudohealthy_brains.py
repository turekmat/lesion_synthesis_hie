import os
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from monai.transforms import GaussianSmooth
from tqdm import tqdm
import math
from scipy import ndimage
from skimage.transform import resize
import torch
from scipy.ndimage import gaussian_filter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import ants  # Přidání importu ANTs


def load_mha_file(file_path):
    """
    Load an MHA file and return as a numpy array
    """
    print(f"Loading {file_path}")
    img = sitk.ReadImage(str(file_path))
    data = sitk.GetArrayFromImage(img)
    return data, img


def save_mha_file(data, reference_image, output_path):
    """
    Save a numpy array as an MHA file using the metadata from reference_image
    """
    print(f"Saving to {output_path}")
    out_img = sitk.GetImageFromArray(data)
    out_img.CopyInformation(reference_image)
    sitk.WriteImage(out_img, str(output_path))


def calculate_lesion_volumes(label_dir, adc_dir):
    """
    Calculate lesion volumes for all patients and return them sorted.
    Ignores cases with empty label maps (zero lesion volume).
    """
    volumes = []
    
    # List all label files
    label_files = [f for f in os.listdir(label_dir) if f.endswith('_lesion.mha')]
    
    non_empty_count = 0
    empty_count = 0
    
    for label_file in label_files:
        # Extract patient ID
        patient_id = label_file.replace('_lesion.mha', '')
        
        # Check if ADC file exists
        adc_file = f"{patient_id}-ADC_ss.mha"
        if not os.path.exists(os.path.join(adc_dir, adc_file)):
            print(f"Warning: ADC file not found for {patient_id}")
            continue
        
        # Load label map
        label_path = os.path.join(label_dir, label_file)
        label_data, _ = load_mha_file(label_path)
        
        # Calculate lesion volume (number of voxels with value > 0)
        lesion_volume = np.sum(label_data > 0)
        
        # Skip cases with empty label maps (no lesions)
        if lesion_volume == 0:
            print(f"Skipping {patient_id} - empty label map (no lesions)")
            empty_count += 1
            continue
        
        volumes.append({
            'patient_id': patient_id,
            'volume': lesion_volume,
            'label_file': label_file,
            'adc_file': adc_file
        })
        non_empty_count += 1
    
    print(f"Found {non_empty_count} cases with lesions and skipped {empty_count} cases with empty label maps")
    
    # Sort by volume
    volumes.sort(key=lambda x: x['volume'])
    
    return volumes


def find_symmetric_region(coords, shape):
    """
    Find the symmetric coordinates in the opposite hemisphere
    Assumes the mid-sagittal plane is at the center of the x-axis
    """
    x, y, z = coords
    mid_x = shape[0] // 2
    symmetric_x = 2 * mid_x - x  # Reflect across mid-sagittal plane
    
    # Make sure the coordinates are within bounds
    symmetric_x = max(0, min(shape[0]-1, symmetric_x))
    
    return (symmetric_x, y, z)


def generate_perlin_noise(shape, scale=10.0, octaves=6):
    """
    Generate 3D Perlin noise
    """
    def perlin(x, y, z, seed=0):
        # These constants are used for the permutation table
        p = np.arange(256, dtype=int)
        np.random.seed(seed)
        np.random.shuffle(p)
        p = np.stack([p, p]).flatten()
        
        # Coordinates of the unit cube
        xi, yi, zi = int(x) & 255, int(y) & 255, int(z) & 255
        
        # Internal coordinates of the cube
        xf, yf, zf = x - int(x), y - int(y), z - int(z)
        
        # Fade curves
        u, v, w = fade(xf), fade(yf), fade(zf)
        
        # Hash coordinates of the 8 cube corners
        perm_x, perm_y, perm_z = p[xi], p[yi], p[zi]
        a = p[perm_x] + perm_y
        aa = p[a] + perm_z
        ab = p[a + 1] + perm_z
        b = p[perm_x + 1] + perm_y
        ba = p[b] + perm_z
        bb = p[b + 1] + perm_z
        
        # Blend gradients from the 8 corners of the cube
        g1 = grad(p[aa], xf, yf, zf)
        g2 = grad(p[ba], xf - 1, yf, zf)
        g3 = grad(p[ab], xf, yf - 1, zf)
        g4 = grad(p[bb], xf - 1, yf - 1, zf)
        g5 = grad(p[aa + 1], xf, yf, zf - 1)
        g6 = grad(p[ba + 1], xf - 1, yf, zf - 1)
        g7 = grad(p[ab + 1], xf, yf - 1, zf - 1)
        g8 = grad(p[bb + 1], xf - 1, yf - 1, zf - 1)
        
        # Interpolate gradients
        v1 = lerp(u, g1, g2)
        v2 = lerp(u, g3, g4)
        v3 = lerp(u, g5, g6)
        v4 = lerp(u, g7, g8)
        
        v5 = lerp(v, v1, v2)
        v6 = lerp(v, v3, v4)
        
        return lerp(w, v5, v6)

    def fade(t):
        # 6t^5 - 15t^4 + 10t^3
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(t, a, b):
        return a + t * (b - a)

    def grad(hash_val, x, y, z):
        h = hash_val & 15
        if h < 8:
            u = x
        else:
            u = y
        if h < 4:
            v = y
        elif h == 12 or h == 14:
            v = x
        else:
            v = z
        if h & 1:
            u = -u
        if h & 2:
            v = -v
        return u + v
    
    # Create a 3D noise array
    noise = np.zeros(shape)
    
    # Generate the noise
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                noise[i, j, k] = 0
                amplitude = 1.0
                frequency = 1.0
                for o in range(octaves):
                    v = perlin(i/scale*frequency, j/scale*frequency, k/scale*frequency, seed=o)
                    noise[i, j, k] += v * amplitude
                    amplitude *= 0.5
                    frequency *= 2.0
    
    # Normalize to range [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    return noise


def create_smooth_transition_mask(binary_mask, sigma=2.0):
    """
    Create a smooth transition mask from a binary mask using Gaussian smoothing
    """
    # Create distance map (positive inside the mask, negative outside)
    dist_map = ndimage.distance_transform_edt(binary_mask) - ndimage.distance_transform_edt(~binary_mask)
    
    # Apply Gaussian smoothing to the distance map
    smooth_dist = gaussian_filter(dist_map, sigma=sigma)
    
    # Convert to transition mask in range [0, 1]
    transition_mask = 1.0 / (1.0 + np.exp(-smooth_dist))
    
    return transition_mask


def create_pseudo_healthy_brain(adc_data, label_data):
    """
    Create a pseudo-healthy brain from ADC and label data by replacing lesions
    """
    # Create a copy of the ADC data for the result
    pseudo_healthy = adc_data.copy()
    
    # Find voxels that belong to the lesion
    lesion_mask = label_data > 0
    
    if not np.any(lesion_mask):
        print("No lesion found in the label data")
        return pseudo_healthy, None  # Vrátíme None jako referenční hodnoty
    
    # Get shape of the data
    shape = adc_data.shape
    
    # Find coordinates of lesion voxels
    lesion_coords = np.where(lesion_mask)
    lesion_coords = list(zip(lesion_coords[0], lesion_coords[1], lesion_coords[2]))
    
    # Create a map to track which regions have symmetric lesions
    has_symmetric_lesion = np.zeros_like(lesion_mask, dtype=bool)
    
    # Check for each lesion voxel if there's a lesion in the symmetric region
    for x, y, z in lesion_coords:
        sym_x, sym_y, sym_z = find_symmetric_region((x, y, z), shape)
        if lesion_mask[sym_x, sym_y, sym_z]:
            has_symmetric_lesion[x, y, z] = True
    
    # Create smooth transition mask
    transition_mask = create_smooth_transition_mask(lesion_mask, sigma=3.0)
    
    # Get connected components in the lesion
    labeled_lesions, num_lesions = ndimage.label(lesion_mask)
    
    # Ukládáme referenční hodnoty pro každou lézi
    reference_values = np.zeros_like(lesion_mask, dtype=np.float32)
    
    # Process each connected component (lesion) separately
    for lesion_idx in range(1, num_lesions + 1):
        current_lesion = labeled_lesions == lesion_idx
        
        # Check if this lesion has symmetric lesions
        has_sym = np.any(has_symmetric_lesion & current_lesion)
        
        if not has_sym:
            # Case 1: No symmetric lesion - use symmetric region values
            
            # Get the coordinates for this lesion
            current_coords = np.where(current_lesion)
            current_coords = list(zip(current_coords[0], current_coords[1], current_coords[2]))
            
            # Get values from symmetric regions
            symmetric_values = []
            for x, y, z in current_coords:
                sym_x, sym_y, sym_z = find_symmetric_region((x, y, z), shape)
                if not lesion_mask[sym_x, sym_y, sym_z]:  # Only if symmetric region is healthy
                    symmetric_values.append(adc_data[sym_x, sym_y, sym_z])
            
            if symmetric_values:
                # Calculate average value from symmetric healthy regions
                avg_value = np.mean(symmetric_values)
                print(f"Average value from symmetric regions: {avg_value:.2f}")
                
                # Uložíme tuto hodnotu jako referenční pro celou lézi
                reference_values[current_lesion] = avg_value
                
                # Generate Perlin noise with a smaller scale for local variations
                noise = generate_perlin_noise(shape, scale=5.0, octaves=3)
                
                # Apply the average value with noise to the lesion area with smooth transition
                for x, y, z in current_coords:
                    # Apply transition weight
                    weight = transition_mask[x, y, z]
                    
                    # Add noise scaled to about 10% of the avg_value
                    noise_scale = 0.1 * avg_value
                    noisy_value = avg_value + (noise[x, y, z] - 0.5) * noise_scale
                    
                    # Blend between original and new value based on the transition mask
                    pseudo_healthy[x, y, z] = (1 - weight) * adc_data[x, y, z] + weight * noisy_value
            
        else:
            # Case 2: Symmetric lesion present - use normative approach
            # This is a simplified approach without proper atlas registration
            
            # Use the mean of healthy tissue around the lesion as an estimate
            # Create a dilated mask of the lesion
            dilated_lesion = ndimage.binary_dilation(current_lesion, iterations=5)
            
            # Create a ring around the lesion (dilated minus original)
            ring_mask = dilated_lesion & ~current_lesion & ~lesion_mask
            
            if np.any(ring_mask):
                # Calculate the mean value in the ring
                ring_values = adc_data[ring_mask]
                avg_ring_value = np.mean(ring_values)
                print(f"Average value from dilated ring: {avg_ring_value:.2f}")
                
                # Uložíme tuto hodnotu jako referenční pro celou lézi
                reference_values[current_lesion] = avg_ring_value
                
                # Generate noise for the lesion area
                noise = generate_perlin_noise(shape, scale=5.0, octaves=3)
                
                # Apply the average value with noise to the lesion area
                current_coords = np.where(current_lesion)
                current_coords = list(zip(current_coords[0], current_coords[1], current_coords[2]))
                
                for x, y, z in current_coords:
                    # Apply transition weight
                    weight = transition_mask[x, y, z]
                    
                    # Add noise scaled to about 10% of the avg_value
                    noise_scale = 0.1 * avg_ring_value
                    noisy_value = avg_ring_value + (noise[x, y, z] - 0.5) * noise_scale
                    
                    # Blend between original and new value based on the transition mask
                    pseudo_healthy[x, y, z] = (1 - weight) * adc_data[x, y, z] + weight * noisy_value
    
    return pseudo_healthy, reference_values


def create_pseudo_healthy_brain_with_atlas(adc_data, label_data, atlas_path, std_atlas_path=None):
    """
    Create a pseudo-healthy brain using normative atlas registration with ANTs.
    
    Args:
        adc_data: Original ADC map data (numpy array)
        label_data: Lesion mask data (numpy array)
        atlas_path: Path to normative atlas (mean values)
        std_atlas_path: Path to normative atlas of standard deviations (optional)
    
    Returns:
        Pseudo-healthy brain data with lesions replaced using atlas values
        Reference values (from atlas) used for replacement
    """
    # Create a copy of the ADC data for the result
    pseudo_healthy = adc_data.copy()
    
    # Find voxels that belong to the lesion
    lesion_mask = label_data > 0
    
    if not np.any(lesion_mask):
        print("No lesion found in the label data")
        return pseudo_healthy, None
    
    print("Performing atlas-based lesion replacement with ANTs registration...")
    
    # Convert numpy arrays to ANTs images for registration
    # Create temporary files for the ANTs registration
    temp_dir = Path("temp_registration")
    temp_dir.mkdir(exist_ok=True)
    
    # Save the ADC and label data as temporary files for ANTs processing
    temp_adc_path = temp_dir / "temp_adc.nii.gz"
    temp_label_path = temp_dir / "temp_label.nii.gz"
    temp_result_path = temp_dir / "temp_result.nii.gz"
    
    # Use SimpleITK to create and save temporary files with proper orientation
    temp_adc_sitk = sitk.GetImageFromArray(adc_data)
    sitk.WriteImage(temp_adc_sitk, str(temp_adc_path))
    
    temp_label_sitk = sitk.GetImageFromArray(label_data.astype(np.uint8))
    sitk.WriteImage(temp_label_sitk, str(temp_label_path))
    
    # Vytvoříme pole pro referenční hodnoty
    reference_values = np.zeros_like(adc_data, dtype=np.float32)
    
    # Load ADC and atlas into ANTs format
    adc_ant = ants.image_read(str(temp_adc_path))
    atlas_ant = ants.image_read(str(atlas_path))
    label_ant = ants.image_read(str(temp_label_path))
    
    # Pokud je k dispozici std atlas, načteme ho také
    std_atlas_ant = None
    if std_atlas_path:
        print(f"Loading standard deviation atlas from {std_atlas_path}")
        std_atlas_ant = ants.image_read(str(std_atlas_path))
    
    # Normalize intensities for better registration
    print("Normalizing images for registration...")
    atlas_ant = ants.iMath(atlas_ant, "Normalize")
    
    # For ADC: normalize to [0,1] range
    adc_np = adc_ant.numpy()
    adc_min, adc_max = adc_np.min(), adc_np.max()
    normalized_adc_np = (adc_np - adc_min) / (adc_max - adc_min)
    adc_ant_norm = ants.from_numpy(
        normalized_adc_np,
        origin=adc_ant.origin,
        spacing=adc_ant.spacing,
        direction=adc_ant.direction
    )
    
    try:
        # SPRÁVNÝ POSTUP: Registrovat PACIENTA NA ATLAS (opačně než předtím)
        print("Performing affine registration of patient to atlas...")
        init_reg = ants.registration(
            fixed=atlas_ant,    # Atlas je nyní fixed image
            moving=adc_ant_norm,  # Pacientský mozek je moving image 
            type_of_transform='Affine',
            verbose=False
        )
        
        print("Performing SyN registration...")
        reg = ants.registration(
            fixed=atlas_ant, 
            moving=adc_ant_norm,
            type_of_transform='SyN',
            initial_transform=init_reg['fwdtransforms'][0],
            reg_iterations=[50, 30, 20],
            verbose=False
        )
        
        # Aplikace transformace na pacientský ADC
        print("Applying transformation to patient ADC...")
        registered_adc = ants.apply_transforms(
            fixed=atlas_ant,
            moving=adc_ant_norm,
            transformlist=reg['fwdtransforms'],
            interpolator='linear',
            verbose=False
        )
        
        # Transformace léze do prostoru atlasu
        print("Transforming lesion mask to atlas space...")
        registered_lesion = ants.apply_transforms(
            fixed=atlas_ant,
            moving=label_ant,
            transformlist=reg['fwdtransforms'],
            interpolator='nearestNeighbor',  # Důležité pro binární masku!
            verbose=False
        )
        
        # Převod do numpy pro další zpracování
        registered_lesion_np = registered_lesion.numpy() > 0
        atlas_np = atlas_ant.numpy()
        
        # Pokud je k dispozici std atlas, získáme ho jako numpy pole
        std_atlas_np = None
        if std_atlas_ant is not None:
            std_atlas_np = std_atlas_ant.numpy()
        
        # Nyní v prostoru atlasu extrahuji hodnoty pro léze
        print("Extracting normative values from atlas in lesion area...")
        
        # Získání hodnot z atlasu pro všechny voxely v transformované lézi
        # NOVÁ ÚPRAVA: Filtruji nulové hodnoty z atlasu v oblasti léze
        valid_atlas_mask = (atlas_np > 0) & registered_lesion_np
        if np.any(valid_atlas_mask):
            # Použijeme pouze nenulové hodnoty z atlasu pro nahrazování
            atlas_values_in_lesion = atlas_np[valid_atlas_mask]
            print(f"Found {len(atlas_values_in_lesion)} valid non-zero atlas values in the registered lesion area")
        else:
            # Pokud nejsou k dispozici žádné nenulové hodnoty, musíme použít jinou metodu
            print("Warning: No non-zero atlas values found in the registered lesion area.")
            print("Falling back to symmetric replacement method...")
            return create_pseudo_healthy_brain(adc_data, label_data)
        
        if len(atlas_values_in_lesion) > 0:
            # Výpočet statistik (průměrná hodnota v atlasu pro oblast léze)
            atlas_mean = np.mean(atlas_values_in_lesion)
            
            # Pokud máme std atlas, použijeme hodnoty z něj přímo pro každý voxel
            # jinak vypočítáme směrodatnou odchylku z hodnot v lézi
            if std_atlas_np is not None:
                print("Using standard deviation values from provided std atlas")
                # NOVÁ ÚPRAVA: Použijeme pouze nenulové hodnoty std atlasu
                valid_std_mask = (std_atlas_np > 0) & registered_lesion_np
                if np.any(valid_std_mask):
                    std_values_in_lesion = std_atlas_np[valid_std_mask]
                    atlas_std_global = np.mean(std_values_in_lesion)  # Průměrná hodnota směrodatné odchylky
                else:
                    # Pokud nejsou k dispozici žádné nenulové hodnoty std atlasu, použijeme základní odhad
                    atlas_std_global = 0.1 * atlas_mean  # Výchozí 10% odhad
                    print("Warning: No non-zero standard deviation values found. Using 10% of mean as default.")
                print(f"Atlas normative values in lesion area: mean={atlas_mean:.2f}, mean std={atlas_std_global:.2f}")
            else:
                # Pokud nemáme std atlas, spočítáme směrodatnou odchylku z hodnot v lézi
                atlas_std_global = np.std(atlas_values_in_lesion)
                print(f"Atlas normative values in lesion area: mean={atlas_mean:.2f}, std={atlas_std_global:.2f}")
            
            # Vytvoření přechodové masky pro původní lézi
            transition_mask = create_smooth_transition_mask(lesion_mask, sigma=3.0)
            
            # Komponenty léze v původním prostoru pacienta
            labeled_lesions, num_lesions = ndimage.label(lesion_mask)
            
            # Generování šumu pro přirozený vzhled
            noise = generate_perlin_noise(adc_data.shape, scale=5.0, octaves=3)
            
            # Zpracování každé komponenty léze zvlášť
            for lesion_idx in range(1, num_lesions + 1):
                current_lesion = labeled_lesions == lesion_idx
                
                # Transformace masky aktuální léze do prostoru atlasu
                current_lesion_ant = ants.from_numpy(
                    current_lesion.astype(np.float32),
                    origin=label_ant.origin,
                    spacing=label_ant.spacing,
                    direction=label_ant.direction
                )
                
                registered_current_lesion = ants.apply_transforms(
                    fixed=atlas_ant,
                    moving=current_lesion_ant,
                    transformlist=reg['fwdtransforms'],
                    interpolator='nearestNeighbor',
                    verbose=False
                )
                
                registered_current_lesion_np = registered_current_lesion.numpy() > 0.5
                
                # NOVÁ ÚPRAVA: Kontrola, jestli jsou v atlasu nenulové hodnoty pro tuto lézi
                valid_lesion_atlas_mask = (atlas_np > 0) & registered_current_lesion_np
                if not np.any(valid_lesion_atlas_mask):
                    print(f"Warning: No non-zero atlas values for lesion component {lesion_idx}.")
                    print("Using alternative method for this lesion component...")
                    
                    # Pro tuto lézi použijeme metodu symetrie nebo perimetru
                    # Vytvoříme dilatovanou masku léze
                    dilated_lesion = ndimage.binary_dilation(current_lesion, iterations=5)
                    
                    # Vytvoříme prstenec kolem léze (dilatovaná bez původní)
                    ring_mask = dilated_lesion & ~current_lesion & ~lesion_mask
                    
                    if np.any(ring_mask):
                        # Vypočítáme průměrnou hodnotu v prstenci
                        ring_values = adc_data[ring_mask]
                        avg_ring_value = np.mean(ring_values)
                        
                        # Aplikujeme průměrnou hodnotu prstence s šumem na oblast léze
                        current_coords = np.where(current_lesion)
                        current_coords = list(zip(current_coords[0], current_coords[1], current_coords[2]))
                        
                        for x, y, z in current_coords:
                            # Váha přechodu
                            weight = transition_mask[x, y, z]
                            
                            # Přidáme šum škálovaný na cca 10% průměrné hodnoty
                            noise_scale = 0.1 * avg_ring_value
                            noisy_value = avg_ring_value + (noise[x, y, z] - 0.5) * 2 * noise_scale
                            
                            # Plynulý přechod mezi původní a novou hodnotou
                            pseudo_healthy[x, y, z] = (1 - weight) * adc_data[x, y, z] + weight * noisy_value
                    
                    # Uložíme referenční hodnotu z prstence
                    if np.any(ring_mask):
                        reference_values[current_lesion] = avg_ring_value
                    
                    continue  # Přeskočíme normální zpracování pomocí atlasu pro tuto komponentu
                
                # Souřadnice voxelů v aktuální lézi
                current_coords = np.where(current_lesion)
                current_coords = list(zip(current_coords[0], current_coords[1], current_coords[2]))
                
                # Uložíme referenční hodnotu z atlasu
                reference_values[current_lesion] = atlas_mean
                
                # Aplikace normativních hodnot z atlasu s přechodem
                for x, y, z in current_coords:
                    # Váha přechodu
                    weight = transition_mask[x, y, z]
                    
                    # ZMĚNA: Přidání šumu škálovaného podle směrodatné odchylky z atlasu
                    if std_atlas_np is not None:
                        # Najít odpovídající voxel v registrované lézi
                        idx = np.argwhere(registered_current_lesion_np)
                        if len(idx) > 0:
                            # Použít lokální směrodatnou odchylku z atlasu std pro tento voxel léze
                            # Pro jednodušší implementaci použijeme průměrnou hodnotu std z transformované oblasti léze
                            noise_scale = atlas_std_global
                        else:
                            # Fallback, pokud není voxel v registrované lézi
                            noise_scale = atlas_std_global
                    else:
                        # Pokud nemáme std atlas, použijeme globální směrodatnou odchylku
                        noise_scale = atlas_std_global
                    
                    # Škálování šumu směrodatnou odchylkou
                    noisy_value = atlas_mean + (noise[x, y, z] - 0.5) * 2 * noise_scale  # *2 pro využití celého rozsahu
                    
                    # Škálování zpět do původního rozsahu hodnot pacientského obrazu
                    scaled_value = noisy_value * (adc_max - adc_min) + adc_min
                    
                    # Plynulý přechod mezi původní a novou hodnotou
                    pseudo_healthy[x, y, z] = (1 - weight) * adc_data[x, y, z] + weight * scaled_value
        
        else:
            print("Warning: No valid lesion voxels found in registered space. Using original method.")
            return create_pseudo_healthy_brain(adc_data, label_data)
        
        print("Atlas-based lesion replacement completed")
        
    except Exception as e:
        print(f"Error during atlas registration: {e}")
        print("Falling back to symmetric replacement method...")
        return create_pseudo_healthy_brain(adc_data, label_data)
    
    finally:
        # Clean up temporary files
        try:
            if temp_adc_path.exists():
                temp_adc_path.unlink()
            if temp_label_path.exists():
                temp_label_path.unlink()
            if temp_result_path.exists():
                temp_result_path.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")
    
    return pseudo_healthy, reference_values


def visualize_results(adc_data, pseudo_healthy, label_data, patient_id, output_dir, atlas_path=None, std_atlas_path=None, reference_values=None):
    """
    Create a PDF visualization showing original ADC, pseudo-healthy ADC, lesion outline,
    and the normative atlas slice used for replacing the lesion.
    
    Args:
        adc_data: Original ADC map data
        pseudo_healthy: Pseudo-healthy ADC map data
        label_data: Lesion mask data
        patient_id: Patient identifier for naming the output
        output_dir: Directory to save the PDF visualization
        atlas_path: Path to the normative atlas (optional, for showing where values came from)
        std_atlas_path: Path to the standard deviation atlas (optional)
        reference_values: Reference values used for replacing lesions (optional)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{patient_id}_visualization.pdf")
    print(f"Creating visualization PDF: {output_path}")
    
    # Inicializace proměnných, které se používají později
    atlas_data_axial = None
    registered_lesion_axial = None
    
    # Prepare atlas data if path is provided
    if atlas_path is not None and np.any(label_data > 0):
        print("Preparing atlas visualization data...")
        # We need to register the lesion to the atlas space similarly as in create_pseudo_healthy_brain_with_atlas
        # Create temporary directory for registration
        temp_dir = Path("temp_visualization")
        temp_dir.mkdir(exist_ok=True)
        
        # Save temporary files for ANTs processing
        temp_adc_path = temp_dir / "temp_viz_adc.nii.gz"
        temp_label_path = temp_dir / "temp_viz_label.nii.gz"
        
        # Use SimpleITK to save temporary files
        temp_adc_sitk = sitk.GetImageFromArray(adc_data)
        sitk.WriteImage(temp_adc_sitk, str(temp_adc_path))
        
        temp_label_sitk = sitk.GetImageFromArray(label_data.astype(np.uint8))
        sitk.WriteImage(temp_label_sitk, str(temp_label_path))
        
        try:
            # Load images in ANTs format
            adc_ant = ants.image_read(str(temp_adc_path))
            atlas_ant = ants.image_read(str(atlas_path))
            label_ant = ants.image_read(str(temp_label_path))
            
            # Normalize images for better registration
            atlas_ant = ants.iMath(atlas_ant, "Normalize")
            
            # Normalize ADC to [0,1] range
            adc_np = adc_ant.numpy()
            adc_min, adc_max = adc_np.min(), adc_np.max()
            normalized_adc_np = (adc_np - adc_min) / (adc_max - adc_min)
            adc_ant_norm = ants.from_numpy(
                normalized_adc_np,
                origin=adc_ant.origin,
                spacing=adc_ant.spacing,
                direction=adc_ant.direction
            )
            
            # Perform registration in the same way as in create_pseudo_healthy_brain_with_atlas
            print("Performing registration for visualization...")
            # Affine registration
            init_reg = ants.registration(
                fixed=atlas_ant,
                moving=adc_ant_norm,
                type_of_transform='Affine',
                verbose=False
            )
            
            # SyN registration
            reg = ants.registration(
                fixed=atlas_ant,
                moving=adc_ant_norm,
                type_of_transform='SyN',
                initial_transform=init_reg['fwdtransforms'][0],
                reg_iterations=[50, 30, 20],
                verbose=False
            )
            
            # Transform lesion mask to atlas space
            registered_lesion = ants.apply_transforms(
                fixed=atlas_ant,
                moving=label_ant,
                transformlist=reg['fwdtransforms'],
                interpolator='nearestNeighbor',
                verbose=False
            )
            
            # Convert atlas and registered lesion to numpy arrays
            atlas_np = atlas_ant.numpy()
            registered_lesion_np = registered_lesion.numpy() > 0
            
            # Transpose atlas data for correct orientation in visualization
            atlas_data_axial = np.transpose(atlas_np, (1, 2, 0))
            registered_lesion_axial = np.transpose(registered_lesion_np, (1, 2, 0))
            
        except Exception as e:
            print(f"Error during registration for visualization: {e}")
            atlas_data_axial = None
            registered_lesion_axial = None
            
        finally:
            # Clean up temporary files
            try:
                if temp_adc_path.exists():
                    temp_adc_path.unlink()
                if temp_label_path.exists():
                    temp_label_path.unlink()
                # Remove the temporary directory if it's empty
                if len(list(temp_dir.iterdir())) == 0:
                    temp_dir.rmdir()
            except Exception as e:
                print(f"Warning: Could not clean up temporary visualization files: {e}")
    
    # SimpleITK returns data in [z,y,x] order, we need to transpose for correct orientation
    # Transpose to get proper axial view (top-down view)
    adc_data_axial = np.transpose(adc_data, (1, 2, 0))
    pseudo_healthy_axial = np.transpose(pseudo_healthy, (1, 2, 0))
    label_data_axial = np.transpose(label_data, (1, 2, 0))
    
    # Transponujeme také referenční hodnoty, pokud existují
    reference_values_axial = None
    if reference_values is not None:
        reference_values_axial = np.transpose(reference_values, (1, 2, 0))
    
    # Calculate value limits for consistent display
    vmin = np.percentile(adc_data_axial[adc_data_axial > 0], 1) if np.any(adc_data_axial > 0) else 0
    vmax = np.percentile(adc_data_axial[adc_data_axial > 0], 99) if np.any(adc_data_axial > 0) else 1
    
    # Create a binary mask for the lesion
    lesion_mask = label_data_axial > 0
    
    # Find which slices actually contain brain tissue (non-zero values)
    non_zero_slices = []
    for z in range(adc_data_axial.shape[2]):
        if np.any(adc_data_axial[:, :, z] > 0):
            non_zero_slices.append(z)
    
    if not non_zero_slices:
        print("Warning: No non-zero slices found in the data")
        return
    
    # Generate PDF with all slices
    with PdfPages(output_path) as pdf:
        for z in non_zero_slices:
            # Create a figure with equal subplot sizes
            fig = plt.figure(figsize=(15, 7))  # Increased height to accommodate text
            
            # Calculate statistics for this slice
            if np.any(lesion_mask[:, :, z]):
                # Calculate average intensities in the lesion area
                lesion_voxels_count = np.sum(lesion_mask[:, :, z])
                orig_mean = np.mean(adc_data_axial[:, :, z][lesion_mask[:, :, z]])
                pseudo_mean = np.mean(pseudo_healthy_axial[:, :, z][lesion_mask[:, :, z]])
                diff = pseudo_mean - orig_mean
                percent_change = (diff / orig_mean * 100) if orig_mean != 0 else 0
                
                # Reference value statistics
                reference_value_str = ""
                if reference_values_axial is not None:
                    # Filtrujeme nenulové referenční hodnoty v oblasti léze
                    ref_values_in_lesion = reference_values_axial[:, :, z][lesion_mask[:, :, z]]
                    ref_values_in_lesion = ref_values_in_lesion[ref_values_in_lesion > 0]
                    if len(ref_values_in_lesion) > 0:
                        ref_mean = np.mean(ref_values_in_lesion)
                        reference_value_str = f"\nReference value (target): {ref_mean:.2f}"
                
                # Atlas statistics if available
                atlas_mean_str = ""
                atlas_voxels_count_str = ""
                if atlas_data_axial is not None and registered_lesion_axial is not None:
                    if np.any(registered_lesion_axial[:, :, z]):
                        atlas_voxels_count = np.sum(registered_lesion_axial[:, :, z])
                        atlas_mean = np.mean(atlas_data_axial[:, :, z][registered_lesion_axial[:, :, z]])
                        atlas_mean_str = f"\nAtlas mean intensity in lesion area: {atlas_mean:.2f}"
                        atlas_voxels_count_str = f"\nAtlas lesion area size: {atlas_voxels_count} voxels"
                
                # Generate statistics text
                stats_text = (
                    f"Lesion area statistics (Slice {z}):\n"
                    f"Lesion area size: {lesion_voxels_count} voxels\n"
                    f"Original mean intensity: {orig_mean:.2f}\n"
                    f"Pseudo-healthy mean intensity: {pseudo_mean:.2f}{reference_value_str}\n"
                    f"Difference: {diff:.2f} ({percent_change:.1f}%){atlas_mean_str}{atlas_voxels_count_str}"
                )
            else:
                stats_text = f"No lesion found in this slice ({z})"
            
            # Original ADC
            ax1 = fig.add_subplot(1, 3, 1)
            im1 = ax1.imshow(adc_data_axial[:, :, z], cmap='gray', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
            ax1.set_title(f'Original ADC (Axial Slice {z})')
            ax1.set_axis_off()
            
            # Pseudo-healthy ADC
            ax2 = fig.add_subplot(1, 3, 2)
            im2 = ax2.imshow(pseudo_healthy_axial[:, :, z], cmap='gray', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
            ax2.set_title(f'Pseudo-healthy ADC (Axial Slice {z})')
            ax2.set_axis_off()
            
            # Original ADC with full lesion highlighted in red
            ax3 = fig.add_subplot(1, 3, 3)
            im3 = ax3.imshow(adc_data_axial[:, :, z], cmap='gray', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
            
            # Overlay the complete lesion in red
            if np.any(lesion_mask[:, :, z]):
                # Create a mask for overlay - highlight the entire lesion
                mask = np.zeros((*adc_data_axial.shape[:2], 4))  # RGBA
                mask[:, :, 0] = 1  # Red channel
                # Use the full lesion mask (not just the outline)
                mask[:, :, 3] = lesion_mask[:, :, z] * 0.7  # Alpha channel, slightly transparent
                ax3.imshow(mask, alpha=0.7, aspect='equal', origin='lower')
            
            ax3.set_title(f'Original ADC with Lesion Highlighted (Axial Slice {z})')
            ax3.set_axis_off()
            
            # Add statistics text at the bottom of the figure
            plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10, 
                        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
            # Ensure proper layout and spacing
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for the text
            
            # Save the current figure to PDF
            pdf.savefig(fig)
            plt.close(fig)
    
    # Calculate overall statistics for the whole volume
    if np.any(lesion_mask):
        total_lesion_voxels = np.sum(label_data > 0)
        overall_orig_mean = np.mean(adc_data[label_data > 0])
        overall_pseudo_mean = np.mean(pseudo_healthy[label_data > 0])
        overall_diff = overall_pseudo_mean - overall_orig_mean
        overall_percent = (overall_diff / overall_orig_mean * 100) if overall_orig_mean != 0 else 0
        
        # Reference value statistics for whole volume
        overall_reference_str = ""
        if reference_values is not None:
            # Filtrujeme nenulové referenční hodnoty v lézi
            ref_values_in_lesion = reference_values[label_data > 0]
            ref_values_in_lesion = ref_values_in_lesion[ref_values_in_lesion > 0]
            if len(ref_values_in_lesion) > 0:
                overall_ref_mean = np.mean(ref_values_in_lesion)
                overall_reference_str = f"\nReference mean (target): {overall_ref_mean:.2f}"
        
        print(f"\nOverall statistics for {patient_id}:")
        print(f"Total lesion size: {total_lesion_voxels} voxels")
        print(f"Original mean in lesion area: {overall_orig_mean:.2f}")
        print(f"Pseudo-healthy mean in same area: {overall_pseudo_mean:.2f}{overall_reference_str}")
        print(f"Difference: {overall_diff:.2f} ({overall_percent:.1f}%)")
        
        # Atlas statistics if available
        if atlas_path is not None and registered_lesion_np is not None:
            if np.any(registered_lesion_np):
                atlas_total_voxels = np.sum(registered_lesion_np)
                atlas_np = atlas_ant.numpy()
                atlas_mean_in_lesion = np.mean(atlas_np[registered_lesion_np])
                print(f"Atlas mean in lesion area: {atlas_mean_in_lesion:.2f}")
                print(f"Atlas registered lesion size: {atlas_total_voxels} voxels")
    
    print(f"Visualization saved to {output_path}")


def process_dataset(adc_dir, label_dir, output_dir, atlas_path=None, std_atlas_path=None, percentage=50, visualize=False):
    """
    Process the entire dataset, selecting the lower X% by lesion volume
    but processing them in random order for better comparisons
    """
    # Create output directories
    pseudohealthy_dir = os.path.join(output_dir, "pseudohealthy")
    os.makedirs(pseudohealthy_dir, exist_ok=True)
    
    # If visualization is enabled, create visualization directory
    visualization_dir = None
    if visualize:
        visualization_dir = os.path.join(output_dir, "visualization")
        os.makedirs(visualization_dir, exist_ok=True)
    
    # Calculate lesion volumes and select the lower X%
    volumes = calculate_lesion_volumes(label_dir, adc_dir)
    
    # Calculate how many patients to include (lower X%)
    num_patients = len(volumes)
    num_to_include = int(num_patients * percentage / 100)
    
    # Select only the bottom X% by volume (smallest lesions)
    selected_volumes = volumes[:num_to_include]
    
    # Randomly shuffle the selected volumes to process them in random order
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(selected_volumes)
    
    print(f"Total patients: {num_patients}")
    print(f"Including {num_to_include} patients with the smallest lesion volumes ({percentage}%)")
    print(f"Processing in random order for better comparison")
    print(f"Saving pseudo-healthy data to: {pseudohealthy_dir}")
    if visualize:
        print(f"Saving visualizations to: {visualization_dir}")
    
    # Process selected patients in random order
    for patient in tqdm(selected_volumes):
        patient_id = patient['patient_id']
        
        print(f"\nProcessing {patient_id} (lesion volume: {patient['volume']} voxels)")
        
        # Load ADC and label data
        adc_path = os.path.join(adc_dir, patient['adc_file'])
        label_path = os.path.join(label_dir, patient['label_file'])
        
        adc_data, adc_img = load_mha_file(adc_path)
        label_data, _ = load_mha_file(label_path)
        
        # Create pseudo-healthy brain using atlas if available, otherwise use symmetry
        if atlas_path:
            pseudo_healthy, reference_values = create_pseudo_healthy_brain_with_atlas(adc_data, label_data, atlas_path, std_atlas_path)
        else:
            pseudo_healthy, reference_values = create_pseudo_healthy_brain(adc_data, label_data)
        
        # Always save the pseudo-healthy and difference map files
        output_path = os.path.join(pseudohealthy_dir, f"{patient_id}-PSEUDO_HEALTHY.mha")
        save_mha_file(pseudo_healthy, adc_img, output_path)
        
        
        # If visualize is enabled, create PDF visualization
        if visualize:
            visualize_results(adc_data, pseudo_healthy, label_data, patient_id, visualization_dir, atlas_path, std_atlas_path, reference_values)


def main():
    parser = argparse.ArgumentParser(description='Create pseudo-healthy brains from ADC maps with HIE lesions')
    parser.add_argument('--adc_dir', type=str, default='data/BONBID2023_Train/1ADC_ss', 
                        help='Directory containing ADC maps')
    parser.add_argument('--label_dir', type=str, default='data/BONBID2023_Train/3LABEL', 
                        help='Directory containing label maps')
    parser.add_argument('--output_dir', type=str, default='data/pseudo_healthy', 
                        help='Directory to save results')
    parser.add_argument('--atlas_path', type=str, default=None,
                        help='Path to normative atlas file with mean values')
    parser.add_argument('--std_atlas_path', type=str, default=None,
                        help='Path to normative atlas file with standard deviation values (must be in same space as mean atlas)')
    parser.add_argument('--percentage', type=int, default=50, 
                        help='Process lower X% of cases by lesion volume')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate PDF visualizations instead of saving files')
    
    args = parser.parse_args()
    
    process_dataset(args.adc_dir, args.label_dir, args.output_dir, args.atlas_path, args.std_atlas_path, args.percentage, args.visualize)


if __name__ == "__main__":
    main()
