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
import glob


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


def find_synthetic_lesions(patient_id, synthetic_lesions_dir):
    """
    Find all synthetic lesion files for a given patient ID
    
    Args:
        patient_id: ID of the patient
        synthetic_lesions_dir: Directory containing synthetic lesions
    
    Returns:
        List of paths to synthetic lesion files
    """
    # Patient folder should be in the synthetic_lesions_dir
    patient_folder = os.path.join(synthetic_lesions_dir, patient_id)
    
    if not os.path.exists(patient_folder):
        print(f"Warning: No synthetic lesions folder found for {patient_id}")
        return []
    
    # Find all registered_lesion*.mha files in the patient folder
    lesion_files = glob.glob(os.path.join(patient_folder, "registered_lesion_*.mha"))
    
    if not lesion_files:
        print(f"Warning: No synthetic lesion files found for {patient_id}")
        return []
    
    print(f"Found {len(lesion_files)} synthetic lesion files for {patient_id}")
    return lesion_files


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


def apply_smoothing_postprocessing(lesion_adc, synthetic_lesion_mask, sigma_spatial=1.5, iterations=2, blur_margin=5):
    """
    Applies advanced smoothing to the lesion area and its boundary for more natural transitions.
    
    Args:
        lesion_adc: The ADC data after synthetic lesion insertion
        synthetic_lesion_mask: Mask of the synthetic lesion
        sigma_spatial: Gaussian blur sigma parameter
        iterations: Number of iterations for the progressive smoothing
        blur_margin: How far outside the lesion to apply blurring (in voxels)
        
    Returns:
        Smoothed ADC data with synthetic lesion
    """
    # Make a copy to avoid modifying the original
    result = lesion_adc.copy()
    
    # Create a dilated mask to include the margin around the lesion
    dilated_mask = ndimage.binary_dilation(synthetic_lesion_mask, iterations=blur_margin)
    
    # Create a distance map from lesion boundary
    dist_map = ndimage.distance_transform_edt(synthetic_lesion_mask) - ndimage.distance_transform_edt(~synthetic_lesion_mask)
    
    # Apply Gaussian smoothing to the distance map to create a smooth transition mask
    smooth_dist = gaussian_filter(dist_map, sigma=3.5)
    
    # Create a transition mask in range [0, 1]
    transition_mask = 1.0 / (1.0 + np.exp(-smooth_dist / 1.8))
    
    # Progressive smoothing with decreasing impact
    for i in range(iterations):
        # Calculate current smoothing parameters
        current_sigma = sigma_spatial * (iterations - i) / iterations
        
        # Apply Gaussian filter to the whole image
        smoothed = gaussian_filter(result, sigma=current_sigma)
        
        # Blend smoothed with original using the transition mask and iteration-dependent weighting
        blend_factor = 0.85 * (iterations - i) / iterations
        
        # Apply the blend only in the dilated mask region with gradient weighting
        idx = dilated_mask
        result[idx] = result[idx] * (1 - blend_factor * transition_mask[idx]) + \
                     smoothed[idx] * (blend_factor * transition_mask[idx])
    
    # Final smoothing pass with reduced weight
    final_smoothed = gaussian_filter(result, sigma=0.8)
    result[dilated_mask] = result[dilated_mask] * 0.85 + final_smoothed[dilated_mask] * 0.15
    
    return result


def calculate_lesion_offset(adc_data, pseudo_healthy_data, lesion_mask):
    """
    Calculate the offset value for synthetic lesions based on real lesions
    
    Args:
        adc_data: Original ADC map with lesions
        pseudo_healthy_data: Pseudo-healthy ADC map (lesions removed)
        lesion_mask: Mask of the real lesions
    
    Returns:
        Offset value to apply to create synthetic lesions
    """
    if not np.any(lesion_mask):
        print("Warning: No lesions found in the mask")
        return 0
    
    # Get the average value in the original lesion area
    lesion_values = adc_data[lesion_mask]
    avg_lesion_value = np.mean(lesion_values)
    
    # Get the average value in the same area in the pseudo-healthy image
    healthy_values = pseudo_healthy_data[lesion_mask]
    avg_healthy_value = np.mean(healthy_values)
    
    # Calculate the offset (we want to decrease values, so it should be negative)
    offset = avg_lesion_value - avg_healthy_value
    
    # Ensure the offset is negative (to decrease values)
    if offset >= 0:
        print(f"Warning: Calculated offset is non-negative ({offset:.2f}), using negative value")
        offset = -abs(offset)
    
    print(f"Calculated offset: {offset:.2f} (Avg lesion value: {avg_lesion_value:.2f}, Avg healthy value: {avg_healthy_value:.2f})")
    
    # Adjust offset to create stronger effect (make it more negative)
    adjusted_offset = offset * 1.50
    print(f"Adjusted offset (80% stronger): {adjusted_offset:.2f}")
    
    return adjusted_offset


def add_synthetic_lesion(adc_data, original_label_data, synthetic_lesion_data, pseudo_healthy_data=None, offset=None):
    """
    Add a synthetic lesion to an ADC map
    
    Args:
        adc_data: Original ADC map
        original_label_data: Original lesion mask
        synthetic_lesion_data: Synthetic lesion mask to add
        pseudo_healthy_data: Pseudo-healthy ADC map (optional, for offset calculation)
        offset: Pre-calculated offset (optional, calculated from data if not provided)
    
    Returns:
        Modified ADC map with synthetic lesion added
        Combined lesion mask (original + synthetic)
    """
    # Create a copy of the ADC data for the result
    modified_adc = adc_data.copy()
    
    # Create mask of the original lesions
    original_lesion_mask = original_label_data > 0
    
    # Create mask of the synthetic lesion
    synthetic_lesion_mask = synthetic_lesion_data > 0
    
    # Ensure we only apply changes where there isn't already a lesion
    # This creates a mask where the synthetic lesion is BUT the original lesion isn't
    valid_synthetic_lesion_mask = synthetic_lesion_mask & ~original_lesion_mask
    
    if not np.any(valid_synthetic_lesion_mask):
        print("Warning: Synthetic lesion completely overlaps with original lesion, no changes made")
        # Return combined mask anyway
        combined_mask = np.logical_or(original_lesion_mask, synthetic_lesion_mask).astype(np.uint8)
        return modified_adc, combined_mask
    
    # Calculate offset if not provided and pseudo-healthy data is available
    if offset is None and pseudo_healthy_data is not None:
        offset = calculate_lesion_offset(adc_data, pseudo_healthy_data, original_lesion_mask)
    elif offset is None:
        # Default offset if neither offset nor pseudo-healthy data is provided
        # Use a conservative negative value
        offset = -200
        print(f"No offset or pseudo-healthy data provided, using default offset: {offset}")
    
    # Create a smooth transition mask for the synthetic lesion
    edge_transition_mask = create_smooth_transition_mask(valid_synthetic_lesion_mask, sigma=2.5)
    
    # Get connected components in the synthetic lesion
    labeled_synthetic_lesions, num_synthetic_lesions = ndimage.label(valid_synthetic_lesion_mask)
    
    # Parameters for lesion application
    TARGET_RATIO = 0.95  # How much of the target value to apply (increased from 0.8)
    EDGE_THRESHOLD = 0.08  # Boundary between lesion interior and edge
    EDGE_BLEND_FACTOR = 0.85  # Factor for edge blending (increased from 0.7)
    BOOST_FACTOR = 1.1  # Factor to adjust offset strength (increased from 0.85)
    
    # Generate Perlin noise for the entire volume
    noise = generate_perlin_noise(adc_data.shape, scale=5.0, octaves=3)
    
    # Process each connected component (synthetic lesion) separately
    for lesion_idx in range(1, num_synthetic_lesions + 1):
        current_lesion = labeled_synthetic_lesions == lesion_idx
        
        # Skip if empty
        if not np.any(current_lesion):
            continue
        
        # Calculate mean value in the current synthetic lesion area
        synthetic_values = adc_data[current_lesion]
        synthetic_mean = np.mean(synthetic_values)
        
        # Apply offset (adjusted with BOOST_FACTOR)
        effective_offset = offset * BOOST_FACTOR
        print(f"Synthetic lesion {lesion_idx}: Mean value: {synthetic_mean:.2f}, Effective offset: {effective_offset:.2f}")
        
        # Create distance map from the edge of the lesion
        distance_map = ndimage.distance_transform_edt(current_lesion)
        if distance_map.max() > 0:
            normalized_dist = distance_map / distance_map.max()
        else:
            normalized_dist = distance_map
        
        # Get coordinates of voxels in the current lesion
        current_coords = np.where(current_lesion)
        current_coords = list(zip(current_coords[0], current_coords[1], current_coords[2]))
        
        # Apply the offset to each voxel with smooth transitions
        for x, y, z in current_coords:
            # Get transition weight
            weight = edge_transition_mask[x, y, z]
            
            # Add noise scaled to ~5% of mean value
            noise_scale = 0.05 * synthetic_mean
            noise_value = (noise[x, y, z] - 0.5) * 2 * noise_scale
            
            # Calculate local deviation to preserve texture
            local_deviation = adc_data[x, y, z] - synthetic_mean
            
            if weight > EDGE_THRESHOLD:
                # Interior of the lesion
                normalized_weight = (weight - EDGE_THRESHOLD) / (1 - EDGE_THRESHOLD)
                blend_ratio = TARGET_RATIO * (1 - 0.3 * (1 - normalized_weight) ** 3)
                dist_factor = 0.7 + 0.3 * normalized_dist[x, y, z] ** 0.33
                
                # Apply offset while preserving local texture variation
                modified_adc[x, y, z] = synthetic_mean + local_deviation + blend_ratio * dist_factor * effective_offset + noise_value
            else:
                # Edge of the lesion
                normalized_weight = weight / EDGE_THRESHOLD
                edge_ratio = TARGET_RATIO * EDGE_BLEND_FACTOR * normalized_weight ** 3
                
                # Apply reduced offset at the edges for smooth transition
                modified_adc[x, y, z] = adc_data[x, y, z] + edge_ratio * effective_offset + noise_value
    
    # Apply post-processing smoothing for more natural transitions
    print("Applying post-processing smoothing for more natural transitions...")
    modified_adc = apply_smoothing_postprocessing(
        modified_adc, 
        valid_synthetic_lesion_mask,
        sigma_spatial=1.8, 
        iterations=4, 
        blur_margin=7
    )
    
    # Create combined lesion mask (original + synthetic)
    combined_mask = np.logical_or(original_lesion_mask, synthetic_lesion_mask).astype(np.uint8)
    
    return modified_adc, combined_mask


def visualize_results(orig_adc_data, mod_adc_data, orig_label_data, comb_label_data, 
                      patient_id, output_dir, synthetic_lesion_masks=None):
    """
    Create a PDF visualization showing original ADC, modified ADC with synthetic lesions,
    original lesion outlines, and combined lesion outlines.
    
    Args:
        orig_adc_data: Original ADC map data
        mod_adc_data: Modified ADC map data with synthetic lesions
        orig_label_data: Original lesion mask data
        comb_label_data: Combined lesion mask data (original + synthetic)
        patient_id: Patient identifier for naming the output
        output_dir: Directory to save the PDF visualization
        synthetic_lesion_masks: List of synthetic lesion masks (optional, for visualization)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{patient_id}_visualization.pdf")
    print(f"Creating visualization PDF: {output_path}")
    
    # SimpleITK returns data in [z,y,x] order, we need to transpose for correct orientation
    # Transpose to get proper axial view (top-down view)
    orig_adc_axial = np.transpose(orig_adc_data, (1, 2, 0))
    mod_adc_axial = np.transpose(mod_adc_data, (1, 2, 0))
    orig_label_axial = np.transpose(orig_label_data, (1, 2, 0))
    comb_label_axial = np.transpose(comb_label_data, (1, 2, 0))
    
    # Create masks for visualization
    orig_lesion_mask = orig_label_axial > 0
    comb_lesion_mask = comb_label_axial > 0
    
    # Create synthetic-only mask (combined minus original)
    synth_only_mask = comb_lesion_mask & ~orig_lesion_mask
    
    # Calculate value limits for consistent display
    vmin = np.percentile(orig_adc_axial[orig_adc_axial > 0], 1) if np.any(orig_adc_axial > 0) else 0
    vmax = np.percentile(orig_adc_axial[orig_adc_axial > 0], 99) if np.any(orig_adc_axial > 0) else 1
    
    # Find which slices contain brain tissue (non-zero values)
    non_zero_slices = []
    for z in range(orig_adc_axial.shape[2]):
        if np.any(orig_adc_axial[:, :, z] > 0):
            non_zero_slices.append(z)
    
    if not non_zero_slices:
        print("Warning: No non-zero slices found in the data")
        return
    
    # Generate PDF with all slices
    with PdfPages(output_path) as pdf:
        for z in non_zero_slices:
            # Only include slices that have synthetic lesions
            if not np.any(synth_only_mask[:, :, z]):
                continue
                
            # Create a figure with equal subplot sizes
            fig = plt.figure(figsize=(15, 7))
            
            # Calculate statistics for this slice
            if np.any(synth_only_mask[:, :, z]):
                # Calculate average intensities in the lesion area
                synth_voxels_count = np.sum(synth_only_mask[:, :, z])
                orig_mean = np.mean(orig_adc_axial[:, :, z][synth_only_mask[:, :, z]])
                mod_mean = np.mean(mod_adc_axial[:, :, z][synth_only_mask[:, :, z]])
                diff = mod_mean - orig_mean
                percent_change = (diff / orig_mean * 100) if orig_mean != 0 else 0
                
                # Statistics text
                stats_text = (
                    f"Synthetic Lesion Statistics (Slice {z}):\n"
                    f"Synthetic lesion area size: {synth_voxels_count} voxels\n"
                    f"Original mean intensity: {orig_mean:.2f}\n"
                    f"Modified mean intensity: {mod_mean:.2f}\n"
                    f"Difference: {diff:.2f} ({percent_change:.1f}%)"
                )
            else:
                stats_text = f"No synthetic lesion in this slice ({z})"
            
            # Original ADC
            ax1 = fig.add_subplot(1, 3, 1)
            im1 = ax1.imshow(orig_adc_axial[:, :, z], cmap='gray', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
            # Overlay original lesion outline in blue
            if np.any(orig_lesion_mask[:, :, z]):
                # Create contour of original lesion
                contours = plt.contour(orig_lesion_mask[:, :, z], levels=[0.5], colors='blue', linewidths=1.5)
            ax1.set_title(f'Original ADC (Axial Slice {z})')
            ax1.set_axis_off()
            
            # Modified ADC with synthetic lesion
            ax2 = fig.add_subplot(1, 3, 2)
            im2 = ax2.imshow(mod_adc_axial[:, :, z], cmap='gray', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
            # Overlay combined lesion outline
            if np.any(comb_lesion_mask[:, :, z]):
                # Create contour of combined lesion
                contours = plt.contour(comb_lesion_mask[:, :, z], levels=[0.5], colors='red', linewidths=1.5)
            ax2.set_title(f'Modified ADC with Synthetic Lesion (Axial Slice {z})')
            ax2.set_axis_off()
            
            # Difference map (original - modified)
            ax3 = fig.add_subplot(1, 3, 3)
            diff_map = orig_adc_axial[:, :, z] - mod_adc_axial[:, :, z]
            # Show differences only in the synthetic lesion area
            masked_diff = np.zeros_like(diff_map)
            masked_diff[synth_only_mask[:, :, z]] = diff_map[synth_only_mask[:, :, z]]
            im3 = ax3.imshow(masked_diff, cmap='hot', aspect='equal', origin='lower')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            ax3.set_title(f'Difference Map (Original - Modified)')
            ax3.set_axis_off()
            
            # Add statistics text at the bottom of the figure
            plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10, 
                        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
            # Ensure proper layout and spacing
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            
            # Save the current figure to PDF
            pdf.savefig(fig)
            plt.close(fig)
    
    # Calculate overall statistics for the whole volume
    if np.any(synth_only_mask):
        total_synth_voxels = np.sum(synth_only_mask)
        overall_orig_mean = np.mean(orig_adc_data[np.transpose(synth_only_mask, (2, 0, 1))])
        overall_mod_mean = np.mean(mod_adc_data[np.transpose(synth_only_mask, (2, 0, 1))])
        overall_diff = overall_mod_mean - overall_orig_mean
        overall_percent = (overall_diff / overall_orig_mean * 100) if overall_orig_mean != 0 else 0
        
        print(f"\nOverall statistics for {patient_id}:")
        print(f"Total synthetic lesion size: {total_synth_voxels} voxels")
        print(f"Original mean in synthetic lesion area: {overall_orig_mean:.2f}")
        print(f"Modified mean in synthetic lesion area: {overall_mod_mean:.2f}")
        print(f"Difference: {overall_diff:.2f} ({overall_percent:.1f}%)")
    
    print(f"Visualization saved to {output_path}")


def visualize_comprehensive(orig_adc_data, mod_adc_data, orig_label_data, comb_label_data, 
                           synth_lesion_data, patient_id, output_dir, random_reference_id=None):
    """
    Create a comprehensive visualization for the first processed image showing:
    - Original ADC map with original lesion
    - Modified ADC map with combined lesions
    - Synthetic lesion mask only
    - Combined lesion mask
    - Difference map
    
    Args:
        orig_adc_data: Original ADC map data
        mod_adc_data: Modified ADC map data with synthetic lesions
        orig_label_data: Original lesion mask data
        comb_label_data: Combined lesion mask data (original + synthetic)
        synth_lesion_data: Synthetic lesion mask data
        patient_id: Patient identifier for naming the output
        output_dir: Directory to save the PDF visualization
        random_reference_id: ID of the random reference patient used for offset calculation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{patient_id}_comprehensive_visualization.pdf")
    print(f"Creating comprehensive visualization PDF: {output_path}")
    
    # SimpleITK returns data in [z,y,x] order, we need to transpose for correct orientation
    # Transpose to get proper axial view (top-down view)
    orig_adc_axial = np.transpose(orig_adc_data, (1, 2, 0))
    mod_adc_axial = np.transpose(mod_adc_data, (1, 2, 0))
    orig_label_axial = np.transpose(orig_label_data, (1, 2, 0))
    comb_label_axial = np.transpose(comb_label_data, (1, 2, 0))
    synth_lesion_axial = np.transpose(synth_lesion_data, (1, 2, 0))
    
    # Create masks for visualization
    orig_lesion_mask = orig_label_axial > 0
    comb_lesion_mask = comb_label_axial > 0
    synth_lesion_mask = synth_lesion_axial > 0
    
    # Create synthetic-only mask (combined minus original)
    synth_only_mask = comb_lesion_mask & ~orig_lesion_mask
    
    # Calculate value limits for consistent display
    vmin = np.percentile(orig_adc_axial[orig_adc_axial > 0], 1) if np.any(orig_adc_axial > 0) else 0
    vmax = np.percentile(orig_adc_axial[orig_adc_axial > 0], 99) if np.any(orig_adc_axial > 0) else 1
    
    # Find which slices contain brain tissue and synthetic lesions
    lesion_slices = []
    for z in range(orig_adc_axial.shape[2]):
        if np.any(orig_adc_axial[:, :, z] > 0) and np.any(synth_only_mask[:, :, z]):
            lesion_slices.append(z)
    
    if not lesion_slices:
        print("Warning: No slices with synthetic lesions found")
        return
    
    # Generate PDF with all relevant slices
    with PdfPages(output_path) as pdf:
        for z in lesion_slices:
            # Create a figure with 2 rows of 3 subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Calculate statistics for this slice
            if np.any(synth_only_mask[:, :, z]):
                synth_voxels_count = np.sum(synth_only_mask[:, :, z])
                orig_mean = np.mean(orig_adc_axial[:, :, z][synth_only_mask[:, :, z]])
                mod_mean = np.mean(mod_adc_axial[:, :, z][synth_only_mask[:, :, z]])
                diff = mod_mean - orig_mean
                percent_change = (diff / orig_mean * 100) if orig_mean != 0 else 0
                
                # Statistics text
                stats_text = (
                    f"Patient: {patient_id} (Slice {z})\n"
                    f"{'Random reference patient used: '+random_reference_id if random_reference_id else ''}\n"
                    f"Synthetic lesion area: {synth_voxels_count} voxels\n"
                    f"Original mean intensity: {orig_mean:.2f}\n"
                    f"Modified mean intensity: {mod_mean:.2f}\n"
                    f"Difference: {diff:.2f} ({percent_change:.1f}%)"
                )
            else:
                stats_text = f"No synthetic lesion in this slice ({z})"
            
            # Row 1, Col 1: Original ADC
            ax1 = fig.add_subplot(2, 3, 1)
            im1 = ax1.imshow(orig_adc_axial[:, :, z], cmap='gray', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
            if np.any(orig_lesion_mask[:, :, z]):
                # Overlay original lesion outline in blue
                contours = plt.contour(orig_lesion_mask[:, :, z], levels=[0.5], colors='blue', linewidths=1.5)
            ax1.set_title(f'Original ADC with Original Lesion')
            ax1.set_axis_off()
            
            # Row 1, Col 2: Modified ADC
            ax2 = fig.add_subplot(2, 3, 2)
            im2 = ax2.imshow(mod_adc_axial[:, :, z], cmap='gray', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
            if np.any(comb_lesion_mask[:, :, z]):
                # Overlay combined lesion outline in red
                contours = plt.contour(comb_lesion_mask[:, :, z], levels=[0.5], colors='red', linewidths=1.5)
            ax2.set_title(f'Modified ADC with Combined Lesion')
            ax2.set_axis_off()
            
            # Row 1, Col 3: Difference map
            ax3 = fig.add_subplot(2, 3, 3)
            diff_map = orig_adc_axial[:, :, z] - mod_adc_axial[:, :, z]
            # Show differences only in the synthetic lesion area
            masked_diff = np.zeros_like(diff_map)
            masked_diff[synth_only_mask[:, :, z]] = diff_map[synth_only_mask[:, :, z]]
            im3 = ax3.imshow(masked_diff, cmap='hot', aspect='equal', origin='lower')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            ax3.set_title(f'Difference Map (Original - Modified)')
            ax3.set_axis_off()
            
            # Row 2, Col 1: Original lesion mask
            ax4 = fig.add_subplot(2, 3, 4)
            im4 = ax4.imshow(orig_lesion_mask[:, :, z], cmap='Blues', aspect='equal', origin='lower')
            ax4.set_title(f'Original Lesion Mask')
            ax4.set_axis_off()
            
            # Row 2, Col 2: Synthetic lesion mask
            ax5 = fig.add_subplot(2, 3, 5)
            im5 = ax5.imshow(synth_only_mask[:, :, z], cmap='Reds', aspect='equal', origin='lower')
            ax5.set_title(f'Synthetic Lesion Mask')
            ax5.set_axis_off()
            
            # Row 2, Col 3: Combined lesion mask
            ax6 = fig.add_subplot(2, 3, 6)
            # Create a custom colormap for combined mask
            combined_viz = np.zeros((*comb_lesion_mask.shape[:2], 3))
            combined_viz[:, :, 0] = synth_only_mask[:, :, z]  # Red channel for synthetic
            combined_viz[:, :, 2] = orig_lesion_mask[:, :, z]  # Blue channel for original
            im6 = ax6.imshow(combined_viz, aspect='equal', origin='lower')
            ax6.set_title(f'Combined Lesion Mask')
            ax6.set_axis_off()
            
            # Add statistics text at the bottom of the figure
            plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10, 
                        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
            # Ensure proper layout and spacing
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            
            # Save the current figure to PDF
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"Comprehensive visualization saved to {output_path}")


def process_dataset(adc_dir, label_dir, pseudo_healthy_dir, synthetic_lesions_dir, output_dir, visualize=False, reverse_order=False):
    """
    Process the dataset by adding synthetic lesions to ADC maps
    
    Args:
        adc_dir: Directory containing original ADC maps
        label_dir: Directory containing original lesion label maps
        pseudo_healthy_dir: Directory containing pseudo-healthy ADC maps
        synthetic_lesions_dir: Directory containing synthetic lesion masks
        output_dir: Directory to save results
        visualize: Whether to generate visualizations
        reverse_order: Whether to process files in reverse order
    """
    # Create output directories
    inpainted_dir = os.path.join(output_dir, "lesioned_adc")
    combined_labels_dir = os.path.join(output_dir, "combined_labels")
    
    os.makedirs(inpainted_dir, exist_ok=True)
    os.makedirs(combined_labels_dir, exist_ok=True)
    
    # If visualization is enabled, create visualization directory
    visualization_dir = None
    if visualize:
        visualization_dir = os.path.join(output_dir, "visualization")
        os.makedirs(visualization_dir, exist_ok=True)
    
    # List all ADC files
    adc_files = [f for f in os.listdir(adc_dir) if f.endswith('_ss.mha')]
    
    # Apply reverse order if requested
    if reverse_order:
        print("Processing files in REVERSE order")
        adc_files = adc_files[::-1]
    else:
        print("Processing files in NORMAL order")
    
    # First, collect all valid patients with lesions for random selection
    valid_patients = []
    for adc_file in adc_files:
        patient_id = adc_file.replace('-ADC_ss.mha', '')
        
        # Check if corresponding label file exists
        label_file = f"{patient_id}_lesion.mha"
        label_path = os.path.join(label_dir, label_file)
        
        # Check if corresponding pseudo-healthy file exists
        pseudo_healthy_file = f"{patient_id}-PSEUDO_HEALTHY.mha"
        pseudo_healthy_path = os.path.join(pseudo_healthy_dir, pseudo_healthy_file)
        
        if os.path.exists(label_path) and os.path.exists(pseudo_healthy_path):
            # Quick check if there are lesions
            label_data, _ = load_mha_file(label_path)
            if np.any(label_data > 0):
                valid_patients.append({
                    'patient_id': patient_id,
                    'adc_file': adc_file,
                    'label_file': label_file,
                    'pseudo_healthy_file': pseudo_healthy_file
                })
    
    if not valid_patients:
        print("Error: No valid patients with lesions found for reference selection")
        return
    
    print(f"Found {len(valid_patients)} valid patients with lesions for reference selection")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Flag for first lesion processed
    first_lesion_processed = False
    
    # Process each ADC file
    for adc_file in tqdm(adc_files):
        # Extract patient ID
        patient_id = adc_file.replace('-ADC_ss.mha', '')
        print(f"\nProcessing {patient_id}")
        
        # Check if corresponding label file exists
        label_file = f"{patient_id}_lesion.mha"
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {patient_id}")
            continue
        
        # Check if corresponding pseudo-healthy file exists
        pseudo_healthy_file = f"{patient_id}-PSEUDO_HEALTHY.mha"
        pseudo_healthy_path = os.path.join(pseudo_healthy_dir, pseudo_healthy_file)
        
        if not os.path.exists(pseudo_healthy_path):
            print(f"Warning: Pseudo-healthy file not found for {patient_id}")
            continue
        
        # Find synthetic lesions for this patient
        synthetic_lesion_files = find_synthetic_lesions(patient_id, synthetic_lesions_dir)
        
        if not synthetic_lesion_files:
            print(f"Skipping {patient_id} - no synthetic lesions found")
            continue
        
        # Apply reverse order to synthetic lesions if requested
        if reverse_order:
            synthetic_lesion_files = synthetic_lesion_files[::-1]
        
        # Load ADC and label data for the current patient
        adc_path = os.path.join(adc_dir, adc_file)
        adc_data, adc_img = load_mha_file(adc_path)
        label_data, _ = load_mha_file(label_path)
        
        # Create list of valid reference patients (excluding current patient)
        valid_references = [p for p in valid_patients if p['patient_id'] != patient_id]
        if not valid_references:
            print(f"Warning: No valid reference patients available, using self-reference")
            valid_references = [next((p for p in valid_patients if p['patient_id'] == patient_id), None)]
            
        # Process each synthetic lesion INDIVIDUALLY
        for i, lesion_file in enumerate(synthetic_lesion_files):
            lesion_basename = os.path.basename(lesion_file).replace(".mha", "")
            print(f"Processing synthetic lesion: {lesion_basename}")
            
            # *** Random reference selection for offset calculation (UNIQUE FOR EACH LESION) ***
            # Randomly select a reference patient for this specific lesion
            reference_patient = np.random.choice(valid_references)
            
            # Load reference patient data for offset calculation
            reference_id = reference_patient['patient_id']
            print(f"Using random reference patient {reference_id} for offset calculation (lesion: {lesion_basename})")
            
            reference_adc_path = os.path.join(adc_dir, reference_patient['adc_file'])
            reference_label_path = os.path.join(label_dir, reference_patient['label_file'])
            reference_healthy_path = os.path.join(pseudo_healthy_dir, reference_patient['pseudo_healthy_file'])
            
            reference_adc_data, _ = load_mha_file(reference_adc_path)
            reference_label_data, _ = load_mha_file(reference_label_path)
            reference_healthy_data, _ = load_mha_file(reference_healthy_path)
            
            # Calculate lesion-specific offset value from reference patient's lesions
            lesion_specific_offset = None
            if np.any(reference_label_data > 0):
                lesion_specific_offset = calculate_lesion_offset(reference_adc_data, reference_healthy_data, reference_label_data > 0)
            else:
                print(f"Warning: No lesions found in reference patient, using default offset")
            
            # Load synthetic lesion mask
            synthetic_lesion_data, _ = load_mha_file(lesion_file)
            
            # Create a fresh copy of the ADC data for each synthetic lesion
            modified_adc = adc_data.copy()
            
            # Create a fresh copy of the label data for combining
            combined_label = label_data.copy()
            
            # Add synthetic lesion to the ADC map (just one synthetic lesion at a time)
            modified_adc, combined_label = add_synthetic_lesion(
                modified_adc, 
                combined_label,
                synthetic_lesion_data,
                None,  # We're not using the patient's own pseudo-healthy data anymore
                lesion_specific_offset  # Using the lesion-specific offset from random reference patient
            )
            
            # *** NOVÝ KROK: Vymaskování výsledné ADC mapy podle původní ADC mapy ***
            # Vytvoření masky nenulových hodnot z původní ADC mapy
            original_mask = adc_data > 0
            print("Applying original ADC mask to remove potential background artifacts...")
            
            # Aplikace masky na modifikovanou ADC mapu - zachováme pouze hodnoty v místech, kde byly nenulové hodnoty v původní mapě
            masked_modified_adc = np.zeros_like(modified_adc)
            masked_modified_adc[original_mask] = modified_adc[original_mask]
            
            # Použití vymaskované verze pro další zpracování a uložení
            modified_adc = masked_modified_adc
            
            # Save modified ADC and combined label with unique names for each synthetic lesion
            modified_adc_path = os.path.join(inpainted_dir, f"{patient_id}-{lesion_basename}-LESIONED_ADC.mha")
            combined_label_path = os.path.join(combined_labels_dir, f"{patient_id}-{lesion_basename}_combined_lesion.mha")
            
            save_mha_file(modified_adc, adc_img, modified_adc_path)
            save_mha_file(combined_label, adc_img, combined_label_path)
            
            # Create visualization for this specific lesion
            if visualize:
                # For the first lesion, create comprehensive visualization
                if not first_lesion_processed:
                    visualize_comprehensive(
                        adc_data,
                        modified_adc,
                        label_data,
                        combined_label,
                        synthetic_lesion_data,
                        f"{patient_id}-{lesion_basename}",
                        visualization_dir,
                        reference_id
                    )
                    first_lesion_processed = True
                
                # Standard visualization for all lesions
                visualize_results(
                    adc_data, 
                    modified_adc,
                    label_data,
                    combined_label,
                    f"{patient_id}-{lesion_basename}",
                    visualization_dir
                )


def main():
    parser = argparse.ArgumentParser(description='Add synthetic lesions to ADC maps')
    parser.add_argument('--adc_dir', type=str, required=True,
                        help='Directory containing original ADC maps')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Directory containing original lesion label maps')
    parser.add_argument('--pseudo_healthy_dir', type=str, required=True,
                        help='Directory containing pseudo-healthy ADC maps')
    parser.add_argument('--synthetic_lesions_dir', type=str, required=True,
                        help='Directory containing synthetic lesion masks')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate PDF visualizations')
    parser.add_argument('--reverse', action='store_true',
                        help='Process files in reverse order (useful for parallel processing)')
    
    args = parser.parse_args()
    
    process_dataset(
        args.adc_dir,
        args.label_dir,
        args.pseudo_healthy_dir,
        args.synthetic_lesions_dir,
        args.output_dir,
        args.visualize,
        args.reverse
    )


if __name__ == "__main__":
    main()
