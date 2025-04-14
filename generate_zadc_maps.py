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
            print(f"Warning: Lesion {lesion_name} dimensions {lesion_mask.shape} don't match atlas dimensions {norm_atlas_data.shape}, skipping")
            continue
        
        # Extract values from atlases
        norm_values = norm_atlas_data[lesion_mask]
        stdev_values = stdev_atlas_data[lesion_mask]
        
        # Calculate statistics
        mean_norm = np.mean(norm_values)
        mean_stdev = np.mean(stdev_values)
        
        # Calculate sigma scaling factor
        sigma_scaling = (mean_stdev / mean_norm) * 0.05
        
        # Store statistics
        lesion_stats[lesion_name] = {
            'mean_norm': mean_norm,
            'mean_stdev': mean_stdev,
            'sigma_scaling': sigma_scaling,
            'voxel_count': np.sum(lesion_mask)
        }
        
        print(f"Lesion {lesion_name}: Mean norm={mean_norm:.2f}, Mean stdev={mean_stdev:.2f}, "
              f"Sigma scaling={sigma_scaling:.4f}, Voxels={lesion_stats[lesion_name]['voxel_count']}")
    
    print(f"Processed {len(lesion_stats)} lesions with valid statistics")
    return lesion_stats


def generate_modified_zadc(zadc_orig_data, adc_orig_data, adc_modified_data, sigma_scaling=0.01):
    """
    Generate modified ZADC map based on original ZADC, original ADC, and modified ADC
    
    Args:
        zadc_orig_data: Original ZADC map
        adc_orig_data: Original ADC map
        adc_modified_data: Modified ADC map with synthetic lesions
        sigma_scaling: Scaling factor for normalization
    
    Returns:
        Modified ZADC map and lesion mask
    """
    # Create a mask of where changes were made (where ADC values differ)
    lesion_mask = get_lesion_mask(adc_orig_data, adc_modified_data)
    
    # Calculate the normalization factor sigma for the changed regions
    sigma = np.zeros_like(adc_orig_data, dtype=np.float32)
    
    # Set sigma based on global statistics, scaled by the factor
    adc_std = np.std(adc_orig_data[adc_orig_data > 0])
    sigma[lesion_mask] = adc_std * sigma_scaling
    
    # Calculate ADC difference
    adc_diff = adc_modified_data - adc_orig_data
    
    # Create modified ZADC by adding the normalized ADC difference to original ZADC
    zadc_modified_data = zadc_orig_data.copy()
    zadc_modified_data[lesion_mask] = zadc_orig_data[lesion_mask] + sigma[lesion_mask] * adc_diff[lesion_mask]
    
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
    adc_diff = adc_modified_data - adc_orig_data
    
    # Transpose data to get axial view
    # SimpleITK/nibabel data is in format [z, y, x] where:
    # - z is the number of slices (axial direction)
    # - y is the height of each slice
    # - x is the width of each slice
    # For axial view, we want to visualize the data as looking at the [y, x] plane
    
    # Create a more comprehensive figure with 3x2 subplots
    plt.figure(figsize=(18, 12))
    
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
    
    # ADC Difference
    plt.subplot(233)
    plt.imshow(adc_diff[slice_idx, :, :], cmap='hot', origin='lower')
    plt.title('ADC Difference')
    plt.colorbar()
    
    # Row 2: ZADC maps
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
    
    # ZADC Difference with lesion contour
    plt.subplot(236)
    plt.imshow(zadc_diff[slice_idx, :, :], cmap='hot', origin='lower')
    
    # Add lesion contour
    if np.any(lesion_mask[slice_idx, :, :]):
        from skimage import measure
        contours = measure.find_contours(lesion_mask[slice_idx, :, :].astype(float), 0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'g-', linewidth=2)
    
    plt.title('ZADC Difference with Lesion Contour')
    plt.colorbar()
    
    # Add some statistical information
    if np.any(lesion_mask):
        zadc_change_mean = np.mean(zadc_diff[lesion_mask])
        zadc_change_max = np.max(np.abs(zadc_diff[lesion_mask]))
        plt.figtext(0.5, 0.01, 
                   f"Lesion area: {np.sum(lesion_mask)} voxels\n"
                   f"Mean ZADC change: {zadc_change_mean:.4f}\n"
                   f"Max ZADC change magnitude: {zadc_change_max:.4f}",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Save visualization
    viz_path = os.path.join(viz_dir, f"{patient_id}-{lesion_info}-ZADC_detailed_viz.png")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150)
    plt.close()
    
    print(f"Enhanced visualization saved to {viz_path}")
    
    # Create a 3D visualization of changes across multiple slices
    # Find slices with lesion
    lesion_slice_indices = np.where(np.any(np.any(lesion_mask, axis=1), axis=1))[0]
    
    if len(lesion_slice_indices) > 0:
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
            
            # ADC difference
            plt.subplot(len(selected_slices), 3, 3*i + 2)
            plt.imshow(adc_diff[slice_i, :, :], cmap='hot', origin='lower')
            plt.title(f'ADC Diff (Axial Slice {slice_i})')
            plt.colorbar()
            
            # Lesion mask
            plt.subplot(len(selected_slices), 3, 3*i + 3)
            plt.imshow(adc_orig_data[slice_i, :, :], cmap='gray', origin='lower')
            plt.imshow(lesion_mask[slice_i, :, :], cmap='Reds', alpha=0.5, origin='lower')
            plt.title(f'Lesion Mask (Axial Slice {slice_i})')
        
        # Save multi-slice visualization
        multi_viz_path = os.path.join(viz_dir, f"{patient_id}-{lesion_info}-ZADC_multi_slice_viz.png")
        plt.tight_layout()
        plt.savefig(multi_viz_path, dpi=150)
        plt.close()
        
        print(f"Multi-slice visualization saved to {multi_viz_path}")


def process_dataset(orig_zadc_dir, orig_adc_dir, modified_adc_dir, output_dir, 
                  lesion_stats, default_sigma_scaling=0.01, visualize_all=False):
    """
    Process the dataset by generating modified ZADC maps
    
    Args:
        orig_zadc_dir: Directory containing original ZADC maps
        orig_adc_dir: Directory containing original ADC maps
        modified_adc_dir: Directory containing modified ADC maps with synthetic lesions
        output_dir: Directory to save the modified ZADC maps
        lesion_stats: Dictionary with precomputed lesion statistics
        default_sigma_scaling: Default scaling factor if lesion stats not found
        visualize_all: Whether to create visualizations for all processed files
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
            
            # Get sigma scaling factor for this lesion
            if lesion_name and lesion_name in lesion_stats:
                sigma_scaling = lesion_stats[lesion_name]['sigma_scaling']
                lesion_usage[lesion_name] += 1
                if processed_count < 5 or processed_count % 50 == 0:
                    print(f"Using precomputed sigma_scaling={sigma_scaling:.4f} for lesion {lesion_name}")
            else:
                sigma_scaling = default_sigma_scaling
                if processed_count < 5 or processed_count % 50 == 0:
                    print(f"Lesion {lesion_name} not found in precomputed stats, using default sigma_scaling={sigma_scaling}")
            
            # Generate modified ZADC map
            if processed_count < 5 or processed_count % 50 == 0:
                print(f"Generating modified ZADC for {patient_id} with lesion {lesion_info}")
            modified_zadc_data, lesion_mask = generate_modified_zadc(
                orig_zadc_data, 
                orig_adc_data, 
                modified_adc_data,
                sigma_scaling
            )
            
            # Save the modified ZADC map
            save_mha_file(modified_zadc_data, orig_zadc_img, output_path)
            processed_count += 1
            
            # Create visualizations
            if processed_count <= 5 or visualize_all:
                # Find a slice with changes
                diff = np.abs(orig_adc_data - modified_adc_data)
                if np.any(diff > 0):
                    # Find the slice with maximum difference (now searching in correct dimension for axial view)
                    slice_idx = np.argmax(np.sum(np.sum(diff, axis=1), axis=1))
                    
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
    parser.add_argument('--norm_atlas', type=str, default=None,
                        help='Path to normative atlas (.nii file) (required for precomputing statistics)')
    parser.add_argument('--stdev_atlas', type=str, default=None,
                        help='Path to standard deviation atlas (.nii file) (required for precomputing statistics)')
    parser.add_argument('--sigma_scaling', type=float, default=0.01,
                        help='Default scaling factor for normalization (default: 0.01, used if lesion stats not found)')
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
        print("Will use default sigma_scaling for all lesions")
        lesion_stats = {}
    
    process_dataset(
        args.orig_zadc_dir,
        args.orig_adc_dir,
        args.modified_adc_dir,
        args.output_dir,
        lesion_stats,
        args.sigma_scaling,
        args.visualize_all
    )


if __name__ == "__main__":
    main() 