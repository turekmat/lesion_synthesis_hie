import os
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
import re
from collections import defaultdict


def load_mha_file(file_path):
    """
    Load an MHA file and return as a numpy array
    """
    print(f"Loading {file_path}")
    img = sitk.ReadImage(str(file_path))
    data = sitk.GetArrayFromImage(img)
    return data, img


def load_nii_file(file_path):
    """
    Load a NIfTI file and return as a numpy array
    """
    print(f"Loading {file_path}")
    img = nib.load(file_path)
    data = img.get_fdata()
    # Convert to numpy array with the same orientation as SimpleITK
    data = np.transpose(data, (2, 1, 0))
    return data, img


def save_mha_file(data, reference_image, output_path):
    """
    Save a numpy array as an MHA file using the metadata from reference_image
    """
    print(f"Saving to {output_path}")
    out_img = sitk.GetImageFromArray(data)
    out_img.CopyInformation(reference_image)
    sitk.WriteImage(out_img, str(output_path))


def get_lesion_mask(adc_orig_data, adc_modified_data, threshold=1.0):
    """
    Create a mask of the synthetic lesion areas by comparing original and modified ADC maps
    
    Args:
        adc_orig_data: Original ADC map
        adc_modified_data: Modified ADC map with synthetic lesion
        threshold: Threshold for detecting changes (default: 1.0)
        
    Returns:
        Binary mask of synthetic lesion areas
    """
    # Calculate absolute difference
    diff = np.abs(adc_modified_data - adc_orig_data)
    
    # Create binary mask where difference exceeds threshold
    lesion_mask = diff > threshold
    
    return lesion_mask


def extract_lesion_name_from_file(filename):
    """
    Extract lesion name from modified ADC filename
    
    Args:
        filename: Modified ADC filename (e.g., MGHNICU_015-VISIT_01-registered_lesion_sample115-LESIONED_ADC.mha)
        
    Returns:
        Lesion name (e.g., lesion_sample115)
    """
    # Search for pattern "registered_<lesion_name>"
    match = re.search(r"registered_(lesion_sample\d+)", filename)
    if match:
        return match.group(1)
    else:
        # Try to find any pattern with "lesion_sample"
        match = re.search(r"(lesion_sample\d+)", filename)
        if match:
            return match.group(1)
        
    # If no match, return None
    return None


def precompute_lesion_statistics(lesions_dir, norm_atlas_path, stdev_atlas_path):
    """
    Precompute statistics for all lesions using normative and standard deviation atlases
    
    Args:
        lesions_dir: Directory containing lesion files
        norm_atlas_path: Path to normative atlas
        stdev_atlas_path: Path to standard deviation atlas
        
    Returns:
        Dictionary with lesion statistics
    """
    print("Precomputing lesion statistics...")
    
    # Load atlases
    norm_atlas_data, _ = load_nii_file(norm_atlas_path)
    stdev_atlas_data, _ = load_nii_file(stdev_atlas_path)
    
    print(f"Atlas dimensions: Norm={norm_atlas_data.shape}, StDev={stdev_atlas_data.shape}")
    
    # Dictionary to store statistics for each lesion
    lesion_stats = {}
    
    # List all lesion files
    lesion_files = [f for f in os.listdir(lesions_dir) if f.endswith('.mha') and 'lesion_sample' in f]
    print(f"Found {len(lesion_files)} lesion files")
    
    for lesion_file in lesion_files:
        lesion_name = lesion_file.replace(".mha", "")
        lesion_path = os.path.join(lesions_dir, lesion_file)
        
        # Load lesion mask
        lesion_data, _ = load_mha_file(lesion_path)
        
        # Create binary mask
        lesion_mask = lesion_data > 0
        
        # Skip if lesion mask is empty
        if not np.any(lesion_mask):
            print(f"Warning: Empty lesion mask for {lesion_name}, skipping")
            continue
        
        # Check if dimensions match atlas
        if lesion_mask.shape != norm_atlas_data.shape:
            print(f"Warning: Lesion {lesion_name} dimensions {lesion_mask.shape} don't match atlas dimensions {norm_atlas_data.shape}")
            print(f"Attempting to resample the lesion to match atlas dimensions")
            
            try:
                # Create SimpleITK image from the mask
                lesion_sitk = sitk.GetImageFromArray(lesion_mask.astype(np.uint8))
                
                # Create reference image with atlas dimensions
                ref_image = sitk.Image(norm_atlas_data.shape[::-1], sitk.sitkUInt8)
                
                # Resample the lesion mask to match atlas dimensions
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(ref_image)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for binary masks
                
                resampled_lesion_sitk = resampler.Execute(lesion_sitk)
                resampled_lesion_mask = sitk.GetArrayFromImage(resampled_lesion_sitk).astype(bool)
                
                print(f"Successfully resampled lesion to dimensions {resampled_lesion_mask.shape}")
                lesion_mask = resampled_lesion_mask
            except Exception as e:
                print(f"Error resampling lesion: {e}")
                print("Will try to extract statistics from the original lesion mask")
        
        # Extract values from atlases where the lesion mask is true
        try:
            # Make sure we're only accessing valid indices in the atlas
            valid_mask = lesion_mask
            if lesion_mask.shape != norm_atlas_data.shape:
                # Create a valid mask that fits within the atlas dimensions
                smaller_shape = [min(d1, d2) for d1, d2 in zip(lesion_mask.shape, norm_atlas_data.shape)]
                temp_mask = np.zeros_like(norm_atlas_data, dtype=bool)
                temp_mask[:smaller_shape[0], :smaller_shape[1], :smaller_shape[2]] = lesion_mask[:smaller_shape[0], :smaller_shape[1], :smaller_shape[2]]
                valid_mask = temp_mask
            
            # Extract values from atlases
            norm_values = norm_atlas_data[valid_mask]
            stdev_values = stdev_atlas_data[valid_mask]
            
            # Calculate statistics
            mean_norm = np.mean(norm_values)
            mean_stdev = np.mean(stdev_values)
            
            # Calculate sigma scaling factor - use a more aggressive factor to ensure ZADC changes appropriately
            # The standard deviation to mean ratio helps scale properly based on tissue characteristics
            sigma_scaling = (mean_stdev / mean_norm) * 0.1  # Increased from 0.05 to 0.1 for stronger effect
            
            # Store statistics
            lesion_stats[lesion_name] = {
                'mean_norm': mean_norm,
                'mean_stdev': mean_stdev,
                'sigma_scaling': sigma_scaling,
                'voxel_count': np.sum(valid_mask)
            }
            
            print(f"Lesion {lesion_name}: Mean norm={mean_norm:.2f}, Mean stdev={mean_stdev:.2f}, "
                  f"Sigma scaling={sigma_scaling:.4f}, Voxels={lesion_stats[lesion_name]['voxel_count']}")
        except Exception as e:
            print(f"Error processing lesion statistics for {lesion_name}: {e}")
            continue
    
    print(f"Processed {len(lesion_stats)} lesions with valid statistics")
    return lesion_stats


def generate_modified_zadc(zadc_orig_data, adc_orig_data, adc_modified_data, 
                          lesion_mean_norm, lesion_mean_stdev, sigma_scaling=1.0, provided_lesion_mask=None):
    """
    Generate modified ZADC map based on original ZADC, original ADC, and modified ADC
    
    Args:
        zadc_orig_data: Original ZADC map
        adc_orig_data: Original ADC map
        adc_modified_data: Modified ADC map with synthetic lesions
        lesion_mean_norm: Mean ADC value from normative atlas for this lesion
        lesion_mean_stdev: Standard deviation from normative atlas for this lesion
        sigma_scaling: Optional scaling factor for standard deviation (default: 1.0)
        provided_lesion_mask: Optional pre-computed lesion mask (default: None, will compute from difference)
    
    Returns:
        Modified ZADC map and lesion mask
    """
    # Use provided lesion mask or create one from the ADC difference
    if provided_lesion_mask is not None:
        original_lesion_mask = provided_lesion_mask.copy()
        print(f"Using provided lesion mask with {np.sum(original_lesion_mask)} voxels")
    else:
        # Create a mask of where changes were made (where ADC values differ)
        original_lesion_mask = get_lesion_mask(adc_orig_data, adc_modified_data)
        print(f"Computed difference-based lesion mask with {np.sum(original_lesion_mask)} voxels")
    
    # Simply dilate the mask by 2 voxels to expand the area
    from scipy import ndimage
    expanded_mask = ndimage.binary_dilation(original_lesion_mask, iterations=2)
    
    # Use the expanded mask for all operations
    lesion_mask = expanded_mask
    
    # Initialize the modified ZADC data as a copy of the original
    zadc_modified_data = zadc_orig_data.copy()
    
    if np.any(lesion_mask):
        # Get original values
        orig_adc_in_lesion = adc_orig_data[lesion_mask]
        orig_zadc_in_lesion = zadc_orig_data[lesion_mask]
        
        # Get modified ADC values in lesion area
        modified_adc_values = adc_modified_data[lesion_mask]
        
        # Compute ADC difference
        adc_diff = modified_adc_values - orig_adc_in_lesion
        
        # Apply the Z-score formula to get the new ZADC values
        stdev_adjusted = lesion_mean_stdev * sigma_scaling
        modified_zadc_values = (modified_adc_values - lesion_mean_norm) / stdev_adjusted
        
        # Ensure Z-scores are within reasonable range (usually between -10 and 10)
        modified_zadc_values = np.clip(modified_zadc_values, -10, 10)
        
        # Calculate the expected change in ZADC based on the change in ADC
        expected_zadc_diff = adc_diff / stdev_adjusted
        
        # Compute flag for voxels where the ZADC change has the wrong sign
        # If ADC decreases, ZADC should also change in the same direction
        inconsistent_voxels = ((adc_diff < 0) & (modified_zadc_values > orig_zadc_in_lesion))
        
        # For inconsistent voxels, adjust the ZADC values to ensure they decrease
        if np.any(inconsistent_voxels):
            # Count inconsistent voxels
            inconsistent_count = np.sum(inconsistent_voxels)
            total_voxels = inconsistent_voxels.size
            percent_inconsistent = (inconsistent_count / total_voxels) * 100
            
            print(f"Warning: {inconsistent_count}/{total_voxels} voxels ({percent_inconsistent:.2f}%) "
                  f"have inconsistent ZADC changes. Adjusting these values.")
            
            # For these voxels, ensure ZADC decreases by at least the amount of the expected change
            modified_zadc_values[inconsistent_voxels] = orig_zadc_in_lesion[inconsistent_voxels] + expected_zadc_diff[inconsistent_voxels]
        
        # Replace ZADC values in expanded lesion area with new Z-scores
        zadc_modified_data[lesion_mask] = modified_zadc_values
        
        # Print statistics for debugging
        print(f"Original ADC in expanded lesion - Mean: {np.mean(orig_adc_in_lesion):.2f}")
        print(f"Modified ADC in expanded lesion - Mean: {np.mean(modified_adc_values):.2f}")
        print(f"ADC difference - Mean: {np.mean(adc_diff):.2f}")
        print(f"Original ZADC in expanded lesion - Mean: {np.mean(orig_zadc_in_lesion):.4f}")
        print(f"Modified ZADC in expanded lesion - Mean: {np.mean(modified_zadc_values):.4f}")
        print(f"ZADC difference - Mean: {np.mean(modified_zadc_values - orig_zadc_in_lesion):.4f}")
        print(f"Normative atlas - Mean: {lesion_mean_norm:.2f}, StDev: {lesion_mean_stdev:.2f}")
        
        # Calculate percentage of voxels where ADC decreased
        adc_decreased = adc_diff < 0
        percent_adc_decreased = (np.sum(adc_decreased) / adc_decreased.size) * 100
        
        # Calculate percentage of voxels where ZADC decreased
        zadc_decreased = modified_zadc_values < orig_zadc_in_lesion
        percent_zadc_decreased = (np.sum(zadc_decreased) / zadc_decreased.size) * 100
        
        print(f"ADC decreased in {percent_adc_decreased:.2f}% of expanded lesion voxels")
        print(f"ZADC decreased in {percent_zadc_decreased:.2f}% of expanded lesion voxels")
        print(f"Expanded lesion mask from {np.sum(original_lesion_mask)} to {np.sum(lesion_mask)} voxels")
    
    return zadc_modified_data, lesion_mask


def find_matching_zadc_file(patient_id, zadc_dir):
    """
    Find the corresponding ZADC file for a patient ID
    
    Args:
        patient_id: Patient ID to match (e.g., MGHNICU_010-VISIT_01)
        zadc_dir: Directory containing ZADC files
        
    Returns:
        Path to the matching ZADC file or None if not found
    """
    # List all files in the ZADC directory
    zadc_files = [f for f in os.listdir(zadc_dir) if f.endswith('.mha')]
    
    # First, look for exact match with format: Zmap_MGHNICU_010-VISIT_01-ADC_...
    expected_prefix = f"Zmap_{patient_id}"
    for zadc_file in zadc_files:
        if zadc_file.startswith(expected_prefix):
            return os.path.join(zadc_dir, zadc_file)
    
    # If no exact match, try to extract core patient ID (assuming MGHNICU_010-VISIT_01 format)
    # Extract just the patient+visit part for matching
    parts = patient_id.split('-')
    if len(parts) >= 2:
        # Combine patient ID and visit ID
        patient_visit = f"{parts[0]}-{parts[1]}"
        for zadc_file in zadc_files:
            if patient_visit in zadc_file and zadc_file.startswith('Zmap_'):
                return os.path.join(zadc_dir, zadc_file)

    # Try just with patient ID if nothing found
    patient_only = parts[0]  # e.g., MGHNICU_010
    for zadc_file in zadc_files:
        if patient_only in zadc_file and zadc_file.startswith('Zmap_'):
            return os.path.join(zadc_dir, zadc_file)
    
    # No matching file found
    return None


def create_enhanced_visualization(orig_zadc_data, modified_zadc_data, adc_orig_data, adc_modified_data, 
                                 lesion_mask, slice_idx, patient_id, lesion_info, output_dir):
    """
    Create enhanced visualization of the ZADC modification
    
    Args:
        orig_zadc_data: Original ZADC data
        modified_zadc_data: Modified ZADC data
        adc_orig_data: Original ADC data
        adc_modified_data: Modified ADC data
        lesion_mask: Mask of the lesion area
        slice_idx: Slice index to visualize
        patient_id: Patient ID
        lesion_info: Lesion information
        output_dir: Output directory for visualization
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Calculate difference maps
    zadc_diff = modified_zadc_data - orig_zadc_data
    
    # Calculate detailed statistics in lesion area
    zadc_stats = {}
    adc_stats = {}
    
    if np.any(lesion_mask):
        # Original ZADC values in lesion area
        orig_zadc_in_lesion = orig_zadc_data[lesion_mask]
        zadc_stats['orig_mean'] = np.mean(orig_zadc_in_lesion)
        zadc_stats['orig_min'] = np.min(orig_zadc_in_lesion)
        zadc_stats['orig_max'] = np.max(orig_zadc_in_lesion)
        
        # Modified ZADC values in lesion area
        mod_zadc_in_lesion = modified_zadc_data[lesion_mask]
        zadc_stats['mod_mean'] = np.mean(mod_zadc_in_lesion)
        zadc_stats['mod_min'] = np.min(mod_zadc_in_lesion)
        zadc_stats['mod_max'] = np.max(mod_zadc_in_lesion)
        
        # ZADC Differences
        zadc_stats['change_mean'] = zadc_stats['mod_mean'] - zadc_stats['orig_mean']
        zadc_stats['change_min'] = zadc_stats['mod_min'] - zadc_stats['orig_min']
        zadc_stats['change_max'] = zadc_stats['mod_max'] - zadc_stats['orig_max']
        
        # ZADC Percentage changes (avoid division by zero)
        if zadc_stats['orig_mean'] != 0:
            zadc_stats['percent_change_mean'] = (zadc_stats['change_mean'] / np.abs(zadc_stats['orig_mean'])) * 100
        else:
            zadc_stats['percent_change_mean'] = float('nan')
            
        # ZADC Direction of change
        if zadc_stats['change_mean'] > 0:
            zadc_stats['direction'] = "increased"
        elif zadc_stats['change_mean'] < 0:
            zadc_stats['direction'] = "decreased"
        else:
            zadc_stats['direction'] = "unchanged"
            
        # Original ADC values in lesion area
        orig_adc_in_lesion = adc_orig_data[lesion_mask]
        adc_stats['orig_mean'] = np.mean(orig_adc_in_lesion)
        adc_stats['orig_min'] = np.min(orig_adc_in_lesion)
        adc_stats['orig_max'] = np.max(orig_adc_in_lesion)
        
        # Modified ADC values in lesion area
        mod_adc_in_lesion = adc_modified_data[lesion_mask]
        adc_stats['mod_mean'] = np.mean(mod_adc_in_lesion)
        adc_stats['mod_min'] = np.min(mod_adc_in_lesion)
        adc_stats['mod_max'] = np.max(mod_adc_in_lesion)
        
        # ADC Differences
        adc_stats['change_mean'] = adc_stats['mod_mean'] - adc_stats['orig_mean']
        adc_stats['change_min'] = adc_stats['mod_min'] - adc_stats['orig_min'] 
        adc_stats['change_max'] = adc_stats['mod_max'] - adc_stats['orig_max']
        
        # ADC Percentage changes (avoid division by zero)
        if adc_stats['orig_mean'] != 0:
            adc_stats['percent_change_mean'] = (adc_stats['change_mean'] / np.abs(adc_stats['orig_mean'])) * 100
        else:
            adc_stats['percent_change_mean'] = float('nan')
            
        # ADC Direction of change
        if adc_stats['change_mean'] > 0:
            adc_stats['direction'] = "increased"
        elif adc_stats['change_mean'] < 0:
            adc_stats['direction'] = "decreased"
        else:
            adc_stats['direction'] = "unchanged"
    
    # Create a multi-page PDF with both visualizations and statistics
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = os.path.join(viz_dir, f"{patient_id}-{lesion_info}-ZADC_analysis.pdf")
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Main visualizations
        plt.figure(figsize=(18, 10))
        
        # Row 1: ADC maps
        # Original ADC
        plt.subplot(231)
        plt.imshow(adc_orig_data[slice_idx, :, :], cmap='gray', origin='lower')
        plt.title(f'Original ADC (Axial Slice {slice_idx})')
        plt.colorbar()
        
        # Modified ADC
        plt.subplot(232)
        plt.imshow(adc_modified_data[slice_idx, :, :], cmap='gray', origin='lower')
        plt.title(f'Modified ADC (Axial Slice {slice_idx})')
        plt.colorbar()
        
        # ZADC Difference map
        plt.subplot(233)
        plt.imshow(zadc_diff[slice_idx, :, :], cmap='hot', vmin=-2, vmax=2, origin='lower')
        plt.title('ZADC Difference')
        plt.colorbar()
        
        # Row 2: ZADC maps and lesion mask
        # Original ZADC
        plt.subplot(234)
        plt.imshow(orig_zadc_data[slice_idx, :, :], cmap='gray', origin='lower')
        plt.title('Original ZADC')
        plt.colorbar()
        
        # Modified ZADC
        plt.subplot(235)
        plt.imshow(modified_zadc_data[slice_idx, :, :], cmap='gray', origin='lower')
        plt.title('Modified ZADC')
        plt.colorbar()
        
        # Binary Lesion Mask - čistá binární maska léze
        plt.subplot(236)
        plt.imshow(lesion_mask[slice_idx, :, :], cmap='binary', origin='lower')
        plt.title('Binary Lesion Mask')
        
        # Add title to the entire figure
        plt.suptitle(f"{patient_id} with lesion {lesion_info} - Axial Slice {slice_idx}", fontsize=16)
        
        # Save visualization to the PDF
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
        pdf.savefig()
        plt.close()
        
        # Page 2: Statistics page
        if np.any(lesion_mask):
            # Create a figure for statistics display
            plt.figure(figsize=(12, 8))
            
            # Format detailed statistics
            stats_text = (
                f"Statistics for {patient_id} with lesion {lesion_info}\n"
                f"=====================================================\n\n"
                f"Lesion area: {np.sum(lesion_mask)} voxels\n\n"
                f"ADC values in lesion region:\n"
                f"  Original: mean={adc_stats['orig_mean']:.2f}, min={adc_stats['orig_min']:.2f}, max={adc_stats['orig_max']:.2f}\n"
                f"  Modified: mean={adc_stats['mod_mean']:.2f}, min={adc_stats['mod_min']:.2f}, max={adc_stats['mod_max']:.2f}\n"
                f"  Change: mean={adc_stats['change_mean']:.2f} ({adc_stats['percent_change_mean']:.2f}%)\n"
                f"  ADC has {adc_stats['direction']} in the lesion region\n\n"
                f"ZADC values in lesion region:\n"
                f"  Original: mean={zadc_stats['orig_mean']:.4f}, min={zadc_stats['orig_min']:.4f}, max={zadc_stats['orig_max']:.4f}\n"
                f"  Modified: mean={zadc_stats['mod_mean']:.4f}, min={zadc_stats['mod_min']:.4f}, max={zadc_stats['mod_max']:.4f}\n"
                f"  Change: mean={zadc_stats['change_mean']:.4f} ({zadc_stats['percent_change_mean']:.2f}%)\n"
                f"  ZADC has {zadc_stats['direction']} in the lesion region"
            )
            
            # Display statistics on the empty figure
            plt.text(0.1, 0.5, stats_text, fontsize=14, family='monospace', va='center')
            plt.axis('off')  # Turn off axes
            plt.title(f"Statistics for {patient_id} with lesion {lesion_info}", fontsize=16)
            
            # Save statistics to the PDF
            pdf.savefig()
            plt.close()
            
            # Page 3: Multi-slice visualization if there are multiple slices with lesions
            lesion_slice_indices = np.where(np.any(np.any(lesion_mask, axis=1), axis=1))[0]
            
            if len(lesion_slice_indices) > 1:  # Only add this page if there's more than one slice
                # Take up to 4 slices through the lesion
                sample_indices = np.linspace(0, len(lesion_slice_indices)-1, min(4, len(lesion_slice_indices)), dtype=int)
                selected_slices = [lesion_slice_indices[i] for i in sample_indices]
                
                # Create multi-slice visualization
                plt.figure(figsize=(15, 4 * len(selected_slices)))
                
                for i, slice_i in enumerate(selected_slices):
                    # ZADC difference
                    plt.subplot(len(selected_slices), 3, 3*i + 1)
                    plt.imshow(zadc_diff[slice_i, :, :], cmap='hot', vmin=-2, vmax=2, origin='lower')
                    plt.title(f'ZADC Diff (Axial Slice {slice_i})')
                    plt.colorbar()
                    
                    # Modified ZADC
                    plt.subplot(len(selected_slices), 3, 3*i + 2)
                    plt.imshow(modified_zadc_data[slice_i, :, :], cmap='gray', origin='lower')
                    plt.title(f'Modified ZADC (Axial Slice {slice_i})')
                    plt.colorbar()
                    
                    # Binary lesion mask
                    plt.subplot(len(selected_slices), 3, 3*i + 3)
                    plt.imshow(lesion_mask[slice_i, :, :], cmap='binary', origin='lower')
                    plt.title(f'Binary Lesion Mask (Axial Slice {slice_i})')
                
                plt.suptitle(f"Multiple slices for {patient_id} with lesion {lesion_info}", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
                pdf.savefig()
                plt.close()
    
    print(f"Complete analysis saved to {pdf_path}")
    
    # Also create PNG version of the main visualization for quick preview
    plt.figure(figsize=(18, 10))
    
    # Row 1: ADC maps
    # Original ADC
    plt.subplot(231)
    plt.imshow(adc_orig_data[slice_idx, :, :], cmap='gray', origin='lower')
    plt.title(f'Original ADC (Axial Slice {slice_idx})')
    plt.colorbar()
    
    # Modified ADC
    plt.subplot(232)
    plt.imshow(adc_modified_data[slice_idx, :, :], cmap='gray', origin='lower')
    plt.title(f'Modified ADC (Axial Slice {slice_idx})')
    plt.colorbar()
    
    # ZADC Difference map
    plt.subplot(233)
    plt.imshow(zadc_diff[slice_idx, :, :], cmap='hot', vmin=-2, vmax=2, origin='lower')
    plt.title('ZADC Difference')
    plt.colorbar()
    
    # Row 2: ZADC maps and lesion mask
    # Original ZADC
    plt.subplot(234)
    plt.imshow(orig_zadc_data[slice_idx, :, :], cmap='gray', origin='lower')
    plt.title('Original ZADC')
    plt.colorbar()
    
    # Modified ZADC
    plt.subplot(235)
    plt.imshow(modified_zadc_data[slice_idx, :, :], cmap='gray', origin='lower')
    plt.title('Modified ZADC')
    plt.colorbar()
    
    # Binary Lesion Mask - čistá binární maska léze
    plt.subplot(236)
    plt.imshow(lesion_mask[slice_idx, :, :], cmap='binary', origin='lower')
    plt.title('Binary Lesion Mask')
    
    # Add title to the entire figure
    plt.suptitle(f"{patient_id} with lesion {lesion_info} - Axial Slice {slice_idx}", fontsize=16)
    
    # Save visualization as PNG
    viz_path = os.path.join(viz_dir, f"{patient_id}-{lesion_info}-ZADC_detailed_viz.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.savefig(viz_path, dpi=150)
    plt.close()
    
    print(f"Main visualization also saved to {viz_path}")
    
    # Save the statistics to a text file for reference
    if np.any(lesion_mask):
        stats_path = os.path.join(viz_dir, f"{patient_id}-{lesion_info}-statistics.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Statistics for {patient_id} with lesion {lesion_info}\n")
            f.write(f"=====================================================\n\n")
            f.write(f"Lesion size: {np.sum(lesion_mask)} voxels\n\n")
            
            f.write(f"ADC values in lesion region:\n")
            f.write(f"  Original: mean={adc_stats['orig_mean']:.2f}, min={adc_stats['orig_min']:.2f}, max={adc_stats['orig_max']:.2f}\n")
            f.write(f"  Modified: mean={adc_stats['mod_mean']:.2f}, min={adc_stats['mod_min']:.2f}, max={adc_stats['mod_max']:.2f}\n\n")
            f.write(f"ADC changes in lesion region:\n")
            f.write(f"  Mean change: {adc_stats['change_mean']:.2f} ({adc_stats['percent_change_mean']:.2f}%)\n")
            f.write(f"  Min change: {adc_stats['change_min']:.2f}\n")
            f.write(f"  Max change: {adc_stats['change_max']:.2f}\n")
            f.write(f"  Summary: ADC has {adc_stats['direction']} in the lesion region\n\n")
            
            f.write(f"ZADC values in lesion region:\n")
            f.write(f"  Original: mean={zadc_stats['orig_mean']:.4f}, min={zadc_stats['orig_min']:.4f}, max={zadc_stats['orig_max']:.4f}\n")
            f.write(f"  Modified: mean={zadc_stats['mod_mean']:.4f}, min={zadc_stats['mod_min']:.4f}, max={zadc_stats['mod_max']:.4f}\n\n")
            f.write(f"ZADC changes in lesion region:\n")
            f.write(f"  Mean change: {zadc_stats['change_mean']:.4f} ({zadc_stats['percent_change_mean']:.2f}%)\n")
            f.write(f"  Min change: {zadc_stats['change_min']:.4f}\n")
            f.write(f"  Max change: {zadc_stats['change_max']:.4f}\n")
            f.write(f"  Summary: ZADC has {zadc_stats['direction']} in the lesion region\n")
        
        print(f"Statistics text also saved to {stats_path}")


def process_dataset(orig_zadc_dir, orig_adc_dir, modified_adc_dir, output_dir, 
                  lesion_stats, registered_lesions_dir=None, default_sigma_scaling=1.0, visualize_all=False):
    """
    Process the dataset by generating modified ZADC maps
    
    Args:
        orig_zadc_dir: Directory containing original ZADC maps
        orig_adc_dir: Directory containing original ADC maps
        modified_adc_dir: Directory containing modified ADC maps with synthetic lesions
        output_dir: Directory to save the modified ZADC maps
        lesion_stats: Dictionary with precomputed lesion statistics
        registered_lesions_dir: Directory containing registered lesion masks organized by patient
        default_sigma_scaling: Default scaling factor for standard deviation (default: 1.0)
        visualize_all: Whether to create visualizations for all processed files (default: only first 5)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List all modified ADC files
    modified_adc_files = [f for f in os.listdir(modified_adc_dir) if f.endswith('-LESIONED_ADC.mha')]
    
    # Check for existing processed files to resume processing
    existing_files = set()
    if os.path.exists(output_dir):
        existing_files = set(os.listdir(output_dir))
        if existing_files:
            print(f"Found {len(existing_files)} existing processed files in {output_dir}")
            print("Will skip already processed files")
    
    # Count lesions used
    lesion_usage = defaultdict(int)
    
    # Process each modified ADC file
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for modified_adc_file in tqdm(modified_adc_files):
        try:
            # Extract patient and lesion info from filename
            parts = modified_adc_file.split('-')
            
            # Identify the lesion part by finding "LESIONED_ADC.mha" at the end
            lesion_end_idx = -1  # This is the index of "LESIONED_ADC.mha"
            
            # Check how many parts we have
            if len(parts) > 3:  # We have PATIENT-VISIT-LESION-LESIONED_ADC
                # First two parts are patient_id (includes visit)
                patient_id = f"{parts[0]}-{parts[1]}"
                # Everything between patient_id and "LESIONED_ADC" is the lesion info
                lesion_info = '-'.join(parts[2:lesion_end_idx])
            else:  # We have PATIENT-LESION-LESIONED_ADC
                patient_id = parts[0]
                lesion_info = '-'.join(parts[1:lesion_end_idx])
            
            # Extract lesion name from filename
            lesion_name = extract_lesion_name_from_file(modified_adc_file)
            
            # Print for debugging (reduce verbosity for large datasets)
            if processed_count < 5 or processed_count % 50 == 0:
                print(f"Parsed file: {modified_adc_file}")
                print(f"  Patient ID: {patient_id}")
                print(f"  Lesion Info: {lesion_info}")
                print(f"  Extracted Lesion Name: {lesion_name}")
            
            # Construct output filename
            output_filename = f"{patient_id}-{lesion_info}-LESIONED_ZADC.mha"
            output_path = os.path.join(output_dir, output_filename)
            
            # Skip if output file already exists
            if output_filename in existing_files:
                if processed_count < 5 or processed_count % 50 == 0:
                    print(f"Skipping {output_filename} - already processed")
                skipped_count += 1
                continue
            
            # Construct original ADC filename - should match the pattern MGHNICU_010-VISIT_01-ADC_ss.mha
            orig_adc_file = f"{patient_id}-ADC_ss.mha"
            orig_adc_path = os.path.join(orig_adc_dir, orig_adc_file)
            
            # Find matching ZADC file (they have different naming convention)
            orig_zadc_path = find_matching_zadc_file(patient_id, orig_zadc_dir)
            
            modified_adc_path = os.path.join(modified_adc_dir, modified_adc_file)
            
            # Check if all required files exist
            if not os.path.exists(orig_adc_path):
                print(f"Warning: Original ADC file not found: {orig_adc_path}")
                error_count += 1
                continue
            
            if not orig_zadc_path or not os.path.exists(orig_zadc_path):
                print(f"Warning: Original ZADC file not found for patient {patient_id}")
                error_count += 1
                continue
            
            # Load data
            orig_adc_data, orig_adc_img = load_mha_file(orig_adc_path)
            orig_zadc_data, orig_zadc_img = load_mha_file(orig_zadc_path)
            modified_adc_data, _ = load_mha_file(modified_adc_path)
            
            # Check dimensions
            if orig_adc_data.shape != orig_zadc_data.shape or orig_adc_data.shape != modified_adc_data.shape:
                print(f"Warning: Dimension mismatch for {patient_id}. Skipping.")
                print(f"  Original ADC shape: {orig_adc_data.shape}")
                print(f"  Original ZADC shape: {orig_zadc_data.shape}")
                print(f"  Modified ADC shape: {modified_adc_data.shape}")
                error_count += 1
                continue
            
            # Load the registered lesion mask instead of computing from the difference
            lesion_mask = None
            
            if registered_lesions_dir:
                # Construct path to registered lesion mask
                patient_lesion_dir = os.path.join(registered_lesions_dir, patient_id)
                
                if os.path.exists(patient_lesion_dir):
                    # We should find a file named "registered_lesion_sampleXXX.mha"
                    if lesion_name:
                        registered_lesion_file = f"registered_{lesion_name}.mha"
                        registered_lesion_path = os.path.join(patient_lesion_dir, registered_lesion_file)
                        
                        if os.path.exists(registered_lesion_path):
                            print(f"Using registered lesion mask: {registered_lesion_path}")
                            registered_lesion_data, _ = load_mha_file(registered_lesion_path)
                            
                            # Create binary mask
                            lesion_mask = registered_lesion_data > 0
                            
                            # Verify the mask has the same dimensions
                            if lesion_mask.shape != orig_adc_data.shape:
                                print(f"Warning: Registered lesion mask dimensions {lesion_mask.shape} "
                                      f"don't match ADC dimensions {orig_adc_data.shape}")
                                
                                if lesion_mask.shape[0] == 1 and lesion_mask.shape[1:] == orig_adc_data.shape[1:]:
                                    # Special case: if the mask is 2D but ADC is 3D with matching x,y dimensions
                                    # Find the axial slice with the maximum difference to determine where to place the 2D mask
                                    diff = np.abs(orig_adc_data - modified_adc_data)
                                    best_slice = np.argmax(np.sum(np.sum(diff, axis=1), axis=1))
                                    
                                    print(f"Detected 2D mask, placing at axial slice {best_slice}")
                                    
                                    # Create 3D mask with the 2D mask at the best slice
                                    full_mask = np.zeros_like(orig_adc_data, dtype=bool)
                                    full_mask[best_slice, :, :] = lesion_mask[0, :, :]
                                    lesion_mask = full_mask
                                else:
                                    print("Cannot use registered mask due to dimension mismatch, falling back to difference method")
                                    lesion_mask = None
                        else:
                            print(f"Registered lesion mask not found: {registered_lesion_path}")
                    else:
                        print(f"Could not determine lesion name from filename: {modified_adc_file}")
                else:
                    print(f"Patient directory not found in registered lesions: {patient_lesion_dir}")
            
            # If no registered mask was loaded, fall back to calculating from difference
            if lesion_mask is None:
                print("Using difference-based lesion mask")
                lesion_mask = get_lesion_mask(orig_adc_data, modified_adc_data)
            
            # Get lesion statistics from precomputed values
            # These are needed for proper Z-score calculation
            if lesion_name and lesion_name in lesion_stats:
                lesion_mean_norm = lesion_stats[lesion_name]['mean_norm']
                lesion_mean_stdev = lesion_stats[lesion_name]['mean_stdev']
                sigma_scaling = default_sigma_scaling  # Use default scaling (1.0)
                lesion_usage[lesion_name] += 1
                
                if processed_count < 5 or processed_count % 50 == 0:
                    print(f"Using atlas statistics for lesion {lesion_name}:")
                    print(f"  Mean norm: {lesion_mean_norm:.2f}")
                    print(f"  Mean stdev: {lesion_mean_stdev:.2f}")
            else:
                # If we don't have precomputed stats, use global estimate
                # This is less accurate but allows processing to continue
                print(f"No precomputed statistics for lesion {lesion_name}, using global estimates...")
                
                # Use healthy tissue around the lesion as reference
                # First create a dilated mask
                from scipy import ndimage
                dilated_mask = ndimage.binary_dilation(lesion_mask, iterations=2)
                healthy_mask = dilated_mask & ~lesion_mask
                
                # Calculate statistics from healthy tissue
                healthy_adc = orig_adc_data[healthy_mask]
                lesion_mean_norm = np.mean(healthy_adc)
                lesion_mean_stdev = np.std(healthy_adc)
                
                print(f"  Estimated mean from healthy tissue: {lesion_mean_norm:.2f}")
                print(f"  Estimated stdev from healthy tissue: {lesion_mean_stdev:.2f}")
            
            # Generate modified ZADC map
            if processed_count < 5 or processed_count % 50 == 0:
                print(f"Generating modified ZADC for {patient_id} with lesion {lesion_info}")
            
            modified_zadc_data, lesion_mask = generate_modified_zadc(
                orig_zadc_data, 
                orig_adc_data, 
                modified_adc_data,
                lesion_mean_norm,
                lesion_mean_stdev,
                sigma_scaling,
                lesion_mask
            )
            
            # Save the modified ZADC map
            save_mha_file(modified_zadc_data, orig_zadc_img, output_path)
            processed_count += 1
            
            # Create visualizations
            if processed_count <= 5 or visualize_all:
                # Find a slice with lesions
                if np.any(lesion_mask):
                    # Find the slice with the largest lesion area
                    lesion_areas = np.sum(np.sum(lesion_mask, axis=2), axis=1)
                    slice_idx = np.argmax(lesion_areas)
                    
                    # Create enhanced visualization
                    create_enhanced_visualization(
                        orig_zadc_data, 
                        modified_zadc_data,
                        orig_adc_data,
                        modified_adc_data,
                        lesion_mask,
                        slice_idx,
                        patient_id,
                        lesion_info,
                        output_dir
                    )
        except Exception as e:
            print(f"Error processing {modified_adc_file}: {e}")
            error_count += 1
            continue
    
    # Print summary
    print("\n--- Processing Summary ---")
    print(f"Total files: {len(modified_adc_files)}")
    print(f"Processed: {processed_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Errors: {error_count}")
    
    # Print lesion usage statistics
    print("\n--- Lesion Usage Statistics ---")
    for lesion_name, count in sorted(lesion_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"{lesion_name}: {count} files")
    print("-----------------------")


def main():
    parser = argparse.ArgumentParser(description='Generate modified ZADC maps from original ZADC, original ADC, and modified ADC maps')
    parser.add_argument('--orig_zadc_dir', type=str, required=True,
                        help='Directory containing original ZADC maps')
    parser.add_argument('--orig_adc_dir', type=str, required=True,
                        help='Directory containing original ADC maps')
    parser.add_argument('--modified_adc_dir', type=str, required=True,
                        help='Directory containing modified ADC maps with synthetic lesions')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the modified ZADC maps')
    parser.add_argument('--lesions_dir', type=str, default=None,
                        help='Directory containing standardized lesion files (required for precomputing statistics)')
    parser.add_argument('--registered_lesions_dir', type=str, default=None,
                        help='Directory containing registered lesion masks organized by patient')
    parser.add_argument('--norm_atlas', type=str, default=None,
                        help='Path to normative atlas (.nii file) (required for precomputing statistics)')
    parser.add_argument('--stdev_atlas', type=str, default=None,
                        help='Path to standard deviation atlas (.nii file) (required for precomputing statistics)')
    parser.add_argument('--sigma_scaling', type=float, default=1.0,
                        help='Scaling factor for standard deviation (default: 1.0)')
    parser.add_argument('--visualize_all', action='store_true',
                        help='Create visualizations for all processed files (default: only first 5)')
    
    args = parser.parse_args()
    
    # Check if we have all required parameters for precomputing statistics
    if args.lesions_dir and args.norm_atlas and args.stdev_atlas:
        # Precompute statistics for all lesions
        lesion_stats = precompute_lesion_statistics(
            args.lesions_dir,
            args.norm_atlas,
            args.stdev_atlas
        )
    else:
        print("Warning: Missing parameters for precomputing lesion statistics")
        print("Will use estimated statistics for each lesion")
        lesion_stats = {}
    
    process_dataset(
        args.orig_zadc_dir,
        args.orig_adc_dir,
        args.modified_adc_dir,
        args.output_dir,
        lesion_stats,
        args.registered_lesions_dir,
        args.sigma_scaling,
        args.visualize_all
    )


if __name__ == "__main__":
    main() 