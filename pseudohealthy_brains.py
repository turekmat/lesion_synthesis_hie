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
    Calculate lesion volumes for all patients and return them sorted
    """
    volumes = []
    
    # List all label files
    label_files = [f for f in os.listdir(label_dir) if f.endswith('_lesion.mha')]
    
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
        
        volumes.append({
            'patient_id': patient_id,
            'volume': lesion_volume,
            'label_file': label_file,
            'adc_file': adc_file
        })
    
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
        return pseudo_healthy
    
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
    
    return pseudo_healthy


def visualize_results(adc_data, pseudo_healthy, label_data, patient_id, output_dir):
    """
    Create a PDF visualization showing original ADC, pseudo-healthy ADC, and lesion outline
    for all slices of the brain.
    
    Args:
        adc_data: Original ADC map data
        pseudo_healthy: Pseudo-healthy ADC map data
        label_data: Lesion mask data
        patient_id: Patient identifier for naming the output
        output_dir: Directory to save the PDF visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{patient_id}_visualization.pdf")
    print(f"Creating visualization PDF: {output_path}")
    
    # Calculate value limits for consistent display
    vmin = np.percentile(adc_data[adc_data > 0], 1) if np.any(adc_data > 0) else 0
    vmax = np.percentile(adc_data[adc_data > 0], 99) if np.any(adc_data > 0) else 1
    
    # Create a binary mask for the lesion outline
    lesion_mask = label_data > 0
    
    # Create lesion outline using binary erosion
    if np.any(lesion_mask):
        eroded = ndimage.binary_erosion(lesion_mask, iterations=1)
        lesion_outline = lesion_mask & ~eroded
    else:
        lesion_outline = np.zeros_like(lesion_mask)
    
    # Find which slices actually contain brain tissue (non-zero values)
    non_zero_slices = []
    for z in range(adc_data.shape[2]):
        if np.any(adc_data[:, :, z] > 0):
            non_zero_slices.append(z)
    
    if not non_zero_slices:
        print("Warning: No non-zero slices found in the data")
        return
    
    # Generate PDF with all slices
    with PdfPages(output_path) as pdf:
        for z in non_zero_slices:
            plt.figure(figsize=(15, 5))
            
            # Original ADC
            plt.subplot(1, 3, 1)
            plt.imshow(adc_data[:, :, z], cmap='gray', vmin=vmin, vmax=vmax)
            plt.title(f'Original ADC (Slice {z})')
            plt.axis('off')
            
            # Pseudo-healthy ADC
            plt.subplot(1, 3, 2)
            plt.imshow(pseudo_healthy[:, :, z], cmap='gray', vmin=vmin, vmax=vmax)
            plt.title(f'Pseudo-healthy ADC (Slice {z})')
            plt.axis('off')
            
            # Difference with lesion outline
            plt.subplot(1, 3, 3)
            diff = np.abs(pseudo_healthy[:, :, z] - adc_data[:, :, z])
            # Normalize the difference for better visualization
            if np.any(diff > 0):
                diff = diff / np.max(diff)
            
            plt.imshow(adc_data[:, :, z], cmap='gray', vmin=vmin, vmax=vmax)
            
            # Overlay the lesion outline in red
            if np.any(lesion_outline[:, :, z]):
                # Create a mask for overlay
                mask = np.zeros((*adc_data.shape[:2], 4))  # RGBA
                mask[:, :, 0] = 1  # Red channel
                mask[:, :, 3] = lesion_outline[:, :, z] * 1.0  # Alpha channel
                plt.imshow(mask, alpha=0.5)
            
            plt.title(f'Original ADC with Lesion Outline (Slice {z})')
            plt.axis('off')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    
    print(f"Visualization saved to {output_path}")


def process_dataset(adc_dir, label_dir, output_dir, percentage=50, visualize=False):
    """
    Process the entire dataset, selecting the lower X% by lesion volume
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate lesion volumes and select the lower X%
    volumes = calculate_lesion_volumes(label_dir, adc_dir)
    
    # Calculate how many patients to include (lower X%)
    num_patients = len(volumes)
    num_to_include = int(num_patients * percentage / 100)
    
    print(f"Total patients: {num_patients}")
    print(f"Including {num_to_include} patients with the smallest lesion volumes ({percentage}%)")
    
    # Process selected patients
    for i in tqdm(range(num_to_include)):
        patient = volumes[i]
        patient_id = patient['patient_id']
        
        print(f"\nProcessing {patient_id} (lesion volume: {patient['volume']} voxels)")
        
        # Load ADC and label data
        adc_path = os.path.join(adc_dir, patient['adc_file'])
        label_path = os.path.join(label_dir, patient['label_file'])
        
        adc_data, adc_img = load_mha_file(adc_path)
        label_data, _ = load_mha_file(label_path)
        
        # Create pseudo-healthy brain
        pseudo_healthy = create_pseudo_healthy_brain(adc_data, label_data)
        
        if visualize:
            # Generate visualization instead of saving files
            visualize_results(adc_data, pseudo_healthy, label_data, patient_id, output_dir)
        else:
            # Save result files
            output_path = os.path.join(output_dir, f"{patient_id}-PSEUDO_HEALTHY.mha")
            save_mha_file(pseudo_healthy, adc_img, output_path)
            
            # Optional: Save a difference map to visualize changes
            diff_map = np.abs(pseudo_healthy - adc_data)
            diff_path = os.path.join(output_dir, f"{patient_id}-DIFF_MAP.mha")
            save_mha_file(diff_map, adc_img, diff_path)


def main():
    parser = argparse.ArgumentParser(description='Create pseudo-healthy brains from ADC maps with HIE lesions')
    parser.add_argument('--adc_dir', type=str, default='data/BONBID2023_Train/1ADC_ss', 
                        help='Directory containing ADC maps')
    parser.add_argument('--label_dir', type=str, default='data/BONBID2023_Train/3LABEL', 
                        help='Directory containing label maps')
    parser.add_argument('--output_dir', type=str, default='data/pseudo_healthy', 
                        help='Directory to save results')
    parser.add_argument('--percentage', type=int, default=50, 
                        help='Process lower X% of cases by lesion volume')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate PDF visualizations instead of saving files')
    
    args = parser.parse_args()
    
    process_dataset(args.adc_dir, args.label_dir, args.output_dir, args.percentage, args.visualize)


if __name__ == "__main__":
    main()
