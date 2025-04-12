import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import gaussian_filter, binary_dilation
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from diffusers.models import AutoencoderKL
import torch.serialization
from scipy import ndimage
import SimpleITK as sitk

# Try to load numpy types if needed for safe deserialization
try:
    from numpy.core.multiarray import _reconstruct
    from numpy import ndarray, dtype
    from monai.data.meta_tensor import MetaTensor
    
    # Add safe globals for deserialization
    torch.serialization.add_safe_globals([_reconstruct])
    torch.serialization.add_safe_globals([ndarray])
    torch.serialization.add_safe_globals([dtype])
    try:
        torch.serialization.add_safe_globals([MetaTensor])
    except:
        print("MetaTensor not added to safe globals")
except ImportError:
    print("Warning: Could not import some numpy types for serialization")

def load_mha_image(file_path):
    """
    Load a medical image using SimpleITK
    
    Args:
        file_path: Path to the image file (MHA, NIfTI) or directory (DICOM)
        
    Returns:
        Object mimicking nibabel's NiftiImage
    """
    try:
        # Check if it's a directory (potentially DICOM)
        if os.path.isdir(file_path):
            return convert_dicom_to_array(file_path)
        
        # Try loading with SimpleITK for other file types
        sitk_img = sitk.ReadImage(file_path)
        
        # Convert to numpy array
        data = sitk.GetArrayFromImage(sitk_img)
        
        # Create affine matrix from SimpleITK information
        direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
        spacing = np.array(sitk_img.GetSpacing())
        origin = np.array(sitk_img.GetOrigin())
        
        # Create affine matrix
        affine = np.eye(4)
        affine[:3, :3] = direction * np.expand_dims(spacing, 1)
        affine[:3, 3] = origin
        
        # SimpleITK uses different axis ordering (z,y,x), so we transpose the data
        # to match nibabel convention (x,y,z)
        # data = np.transpose(data, (2, 1, 0))
        
        # Create a simple wrapper class to mimic nibabel's NiftiImage
        class MHAImage:
            def __init__(self, data, affine, shape):
                self.data = data
                self.affine = affine
                self.shape = shape
                
            def get_fdata(self):
                return self.data
        
        return MHAImage(data, affine, data.shape)
        
    except Exception as e:
        print(f"Error loading file with SimpleITK: {e}")
        # Fall back to nibabel if SimpleITK fails
        try:
            return nib.load(file_path)
        except Exception as e2:
            raise Exception(f"Failed to load {file_path} with both SimpleITK and nibabel: {e2}")

class BrainLesionDataset(Dataset):
    def __init__(self, csv_file, patch_size=(64, 64, 64), transformation=None, max_patches_per_volume=10):
        """
        Dataset class for brain lesion data
        
        Args:
            csv_file (str): Path to the CSV file with the data paths
            patch_size (tuple): Size of the patches to extract
            transformation: Transformation to apply to the data
            max_patches_per_volume (int): Maximum number of patches to extract per volume
        """
        self.df = pd.read_csv(csv_file)
        self.patch_size = np.array(patch_size)
        self.transformation = transformation
        self.max_patches_per_volume = max_patches_per_volume
        
        self.patches = []
        
        print(f"Loading {len(self.df)} subjects for training...")
        
        # Extract patches for each subject
        for i, row in self.df.iterrows():
            try:
                print(f"Processing subject {i+1}/{len(self.df)}: {os.path.basename(row['pseudo_healthy'])}")
                
                # Load data
                pseudo_healthy_path = row['pseudo_healthy']
                adc_path = row['adc']
                lesion_path = row['lesion']
                
                # Determine if we're dealing with files or directories
                is_dicom_dataset = any([os.path.isdir(p) for p in [pseudo_healthy_path, adc_path, lesion_path]])
                
                if is_dicom_dataset:
                    print(f"  Detected DICOM directories for subject {i+1}")
                
                # Load the images
                try:
                    pseudo_healthy_img = load_mha_image(pseudo_healthy_path)
                    adc_img = load_mha_image(adc_path)
                    lesion_img = load_mha_image(lesion_path)
                except Exception as e:
                    print(f"  Error loading data for subject {i+1}: {e}")
                    continue
                
                # Get the data and affine matrix
                pseudo_healthy_data = pseudo_healthy_img.get_fdata()
                adc_data = adc_img.get_fdata()
                lesion_data = lesion_img.get_fdata()
                
                # Print shapes for debugging
                print(f"  Pseudo-healthy shape: {pseudo_healthy_data.shape}")
                print(f"  ADC shape: {adc_data.shape}")
                print(f"  Lesion shape: {lesion_data.shape}")
                
                # Ensure data is normalized
                if np.max(pseudo_healthy_data) > 1.0:
                    pseudo_healthy_data = pseudo_healthy_data / np.max(pseudo_healthy_data)
                
                if np.max(adc_data) > 1.0:
                    adc_data = adc_data / np.max(adc_data)
                
                # Ensure lesion is binary
                lesion_data = (lesion_data > 0.5).astype(np.float32)
                
                # Get the coordinates of the lesions
                lesion_coords = np.where(lesion_data > 0.5)
                
                if len(lesion_coords[0]) == 0:
                    print(f"  No lesions found for subject {i+1}")
                    continue
                
                # Sample coordinates from the lesion
                num_lesion_voxels = len(lesion_coords[0])
                indices = np.random.choice(num_lesion_voxels, min(num_lesion_voxels, self.max_patches_per_volume), replace=False)
                
                for idx in indices:
                    center = np.array([lesion_coords[0][idx], lesion_coords[1][idx], lesion_coords[2][idx]])
                    
                    # Define the bounding box
                    lower_bound = np.maximum(center - self.patch_size // 2, 0)
                    upper_bound = np.minimum(lower_bound + self.patch_size, pseudo_healthy_data.shape)
                    lower_bound = np.maximum(upper_bound - self.patch_size, 0)
                    
                    # Extract the patches
                    ph_patch = pseudo_healthy_data[lower_bound[0]:upper_bound[0], 
                                                 lower_bound[1]:upper_bound[1], 
                                                 lower_bound[2]:upper_bound[2]]
                    
                    adc_patch = adc_data[lower_bound[0]:upper_bound[0], 
                                       lower_bound[1]:upper_bound[1], 
                                       lower_bound[2]:upper_bound[2]]
                    
                    lesion_patch = lesion_data[lower_bound[0]:upper_bound[0], 
                                             lower_bound[1]:upper_bound[1], 
                                             lower_bound[2]:upper_bound[2]]
                    
                    # Check if patch size is correct
                    if ph_patch.shape != tuple(self.patch_size) or adc_patch.shape != tuple(self.patch_size) or lesion_patch.shape != tuple(self.patch_size):
                        print(f"  Skipping patch with invalid shape: {ph_patch.shape}, {adc_patch.shape}, {lesion_patch.shape}")
                        continue
                    
                    # Add the patches to the list
                    self.patches.append({
                        'pseudo_healthy': ph_patch,
                        'adc': adc_patch,
                        'lesion': lesion_patch,
                        'subject_id': i,
                        'center': center,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    })
                
                print(f"  Added {len(indices)} patches for subject {i+1}")
                
            except Exception as e:
                print(f"  Error processing subject {i+1}: {e}")
                traceback.print_exc()
                continue
        
        print(f"Created {len(self.patches)} patches from {len(self.df)} subjects")
        
        # Add at least one dummy patch if no patches were created to avoid DataLoader errors
        if len(self.patches) == 0:
            print("WARNING: No valid patches created. Adding dummy patch to avoid DataLoader errors.")
            dummy_shape = tuple(self.patch_size)
            self.patches.append({
                'pseudo_healthy': np.zeros(dummy_shape, dtype=np.float32),
                'adc': np.zeros(dummy_shape, dtype=np.float32),
                'lesion': np.zeros(dummy_shape, dtype=np.float32),
                'subject_id': 0,
                'center': np.zeros(3, dtype=np.int32),
                'lower_bound': np.zeros(3, dtype=np.int32),
                'upper_bound': np.array(dummy_shape, dtype=np.int32)
            })
            
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        
        # Convert to tensor
        ph_tensor = torch.from_numpy(patch['pseudo_healthy']).float().unsqueeze(0)
        adc_tensor = torch.from_numpy(patch['adc']).float().unsqueeze(0)
        lesion_tensor = torch.from_numpy(patch['lesion']).float().unsqueeze(0)
        
        if self.transformation:
            ph_tensor = self.transformation(ph_tensor)
            adc_tensor = self.transformation(adc_tensor)
            lesion_tensor = self.transformation(lesion_tensor)
        
        return {
            'pseudo_healthy': ph_tensor,
            'adc': adc_tensor,
            'lesion': lesion_tensor,
            'subject_id': patch['subject_id'],
            'center': patch['center'],
            'lower_bound': patch['lower_bound'],
            'upper_bound': patch['upper_bound']
        }

def weighted_lesion_loss(outputs, targets, masks, lesion_weight=10.0):
    """
    Weighted loss function that puts more emphasis on lesion areas.
    
    Args:
        outputs: Model outputs
        targets: Target values
        masks: Binary lesion masks
        lesion_weight: Weight applied to lesion areas (1.0 means no weighting)
        
    Returns:
        Total loss
    """
    # Basic L1 loss for the entire image
    base_loss = torch.abs(outputs - targets)
    
    # Create weight map: 1.0 for healthy areas, lesion_weight for lesion areas
    weight_map = 1.0 + (lesion_weight - 1.0) * masks
    
    # Apply weight map to loss
    weighted_loss = base_loss * weight_map
    
    # Return average
    return weighted_loss.mean()

# Function for smoothing output as post-processing
def apply_smoothing(outputs, masks, inputs, sigma=0.8, iterations=2):
    """
    Applies Gaussian smoothing only around lesion boundaries, not on the lesion itself.
    Preserves lesion structure and creates smooth transition at boundaries.
    
    Args:
        outputs: Tensor of model output in shape (B, 1, D, H, W)
        masks: Tensor of lesion masks in shape (B, 1, D, H, W)
        inputs: Tensor of inputs in shape (B, 1, D, H, W) - used for creating brain mask
        sigma: Gaussian filter dispersion parameter
        iterations: Number of dilation iterations for lesion neighborhood
    
    Returns:
        Tensor of modified output with same shape as input
    """
    smoothed_results = []
    
    # Convert to numpy for more efficient processing
    outputs_np = outputs.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    inputs_np = inputs.detach().cpu().numpy()
    
    for b in range(outputs.shape[0]):
        # Get data for current batch
        curr_output = outputs_np[b, 0]
        curr_mask = masks_np[b, 0]
        curr_input = inputs_np[b, 0]
        
        # Create binary brain mask - everywhere input has non-zero value
        brain_mask = (curr_input > 0.01).astype(np.float32)
        
        # Create binary lesion mask
        lesion_mask = (curr_mask > 0.1).astype(np.float32)
        
        # Create dilated mask for lesion surroundings
        dilated_mask = binary_dilation(lesion_mask, iterations=iterations).astype(np.float32)
        
        # Create mask only for lesion surroundings (dilated area minus original lesion)
        # This is key change - separate lesion surroundings from lesion itself
        transition_zone_mask = dilated_mask - lesion_mask
        
        # Ensure transition zone remains within brain mask
        transition_zone_mask = transition_zone_mask * brain_mask
        
        # Apply Gaussian smoothing to entire output
        smoothed = gaussian_filter(curr_output, sigma=sigma)
        
        # Ensure smoothed output remains within brain mask
        smoothed = smoothed * brain_mask
        
        # Combination:
        # 1. Original output in places outside transition zone (including lesions)
        # 2. Smoothed output in transition zone
        final_output = curr_output * (1 - transition_zone_mask) + smoothed * transition_zone_mask
        
        # Convert back to tensor
        smoothed_results.append(torch.from_numpy(final_output).to(outputs.device))
    
    # Compose final batch tensor
    result = torch.stack(smoothed_results).unsqueeze(1)
    
    return result

def calculate_lesion_metrics(outputs, targets, masks, inputs=None):
    """
    Calculates metrics focused specifically on lesion reconstruction quality.
    
    Args:
        outputs: Model outputs
        targets: Target ADC maps
        masks: Binary lesion masks
        inputs: Original pseudo-healthy image (optional)
        
    Returns:
        Dictionary with lesion metrics
    """
    # Convert to numpy for calculation
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy() > 0.5
    
    # If no lesions, return empty dictionary
    if np.sum(masks_np) == 0:
        if inputs is None:
            return {'lesion_mae': 0.0}
        else:
            return {'lesion_mae': 0.0, 'input_output_lesion_mae': 0.0}
    
    # MAE only for lesion areas (output vs. target)
    lesion_mae = np.sum(np.abs(outputs_np[masks_np] - targets_np[masks_np])) / np.sum(masks_np)
    
    # If we have original inputs, calculate MAE between input and output in lesion area
    if inputs is not None:
        inputs_np = inputs.detach().cpu().numpy()
        input_output_lesion_mae = np.sum(np.abs(outputs_np[masks_np] - inputs_np[masks_np])) / np.sum(masks_np)
        return {'lesion_mae': float(lesion_mae), 'input_output_lesion_mae': float(input_output_lesion_mae)}
    
    return {'lesion_mae': float(lesion_mae)}

class MRIAutoencoder(nn.Module):
    def __init__(self, model_path, device="cpu"):
        super(MRIAutoencoder, self).__init__()
        
        print(f"Loading MRI autoencoder from: {model_path}")
        try:
            # Load the MRI autoencoder model from Hugging Face
            self.vae = AutoencoderKL.from_pretrained(model_path)
            self.vae.to(device)
            print("MRI autoencoder model loaded successfully")
            
            # Freeze encoder parameters
            for name, param in self.vae.named_parameters():
                if "encoder" in name:
                    # Unfreeze only last layers of encoder
                    if any(pattern in name for pattern in ["encoder.down.2", "encoder.down.3", "encoder.mid"]):
                        param.requires_grad = True
                        print(f"Unfrozen layer: {name}")
                    else:
                        param.requires_grad = False
                        
            print("Encoder layers partially unfrozen - last layers are active for training")
            
        except Exception as e:
            print(f"Error loading MRI autoencoder model: {e}")
            import traceback
            traceback.print_exc()
            raise Exception("Failed to load the MRI autoencoder model")
    
    def forward(self, x, mask):
        # If we're in training mode, we need gradients from encoder
        if self.training:
            z_mu, z_sigma = self.vae.encode(x)
        else:
            with torch.no_grad():
                z_mu, z_sigma = self.vae.encode(x)
        
        # Decode from latent representation
        decoded = self.vae.decode(z_mu)
        
        # Ensure output has same shape as input
        if decoded.shape != x.shape:
            decoded = F.interpolate(decoded, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        # Use mask to blend original input (pseudo-healthy) with decoded output
        # Keep original where mask is 0, use decoded where mask is 1
        raw_output = x * (1 - mask) + decoded * mask
        
        return raw_output

def visualize_full_subject(dataset, model, subject_id, epoch, output_dir, device):
    """
    Creates and saves visualizations for entire brain volume of one subject.
    
    Args:
        dataset: Instance of BrainLesionDataset
        model: Trained MRIAutoencoder model
        subject_id: ID of subject for visualization
        epoch: Epoch number
        output_dir: Directory for saving visualizations
        device: Device for running model (CPU/GPU)
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"Visualizing entire subject {subject_id} for epoch {epoch}...")
    
    # Get original subject data
    subject_data = dataset.get_full_subject_data(subject_id)
    if subject_data is None:
        print(f"  WARNING: Data for subject {subject_id} not found!")
        return
    
    ph_data = subject_data['ph_data']
    adc_data = subject_data['adc_data']
    lesion_data = subject_data['lesion_data']
    orig_shape = subject_data['orig_shape']
    
    # Get all patches for this subject
    patch_locations = dataset.get_patch_locations(subject_id)
    
    # Prepare patches for reconstruction
    output_patches = []
    patch_centers = []
    
    # For each patch location
    for patch_info in patch_locations:
        center = patch_info['center']
        
        # Extract patch for model input
        ph_patch = dataset._extract_patch(ph_data, center)
        lesion_patch = dataset._extract_patch(lesion_data, center)
        
        # Convert to tensors
        ph_tensor = torch.from_numpy(ph_patch).float().unsqueeze(0).unsqueeze(0).to(device)
        lesion_tensor = torch.from_numpy(lesion_patch).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Model inference
        with torch.no_grad():
            output = model(ph_tensor, lesion_tensor)
            
            # Apply smoothing to improve transitions
            output = apply_smoothing(output, lesion_tensor, ph_tensor)
        
        # Add to list for reconstruction
        output_patches.append(output.cpu().numpy()[0, 0])
        patch_centers.append(center)
    
    # Reconstruct full volume from patches
    reconstructed_volume = dataset.reconstruct_full_volume(output_patches, patch_centers, orig_shape)
    
    # Create PDF with visualizations
    pdf_path = os.path.join(viz_dir, f'subject_{subject_id}_epoch_{epoch}.pdf')
    with PdfPages(pdf_path) as pdf:
        # Find slices with lesions to visualize
        lesion_slices = {}
        for dim in range(3):
            # Sum along dimension to find slices with lesions
            if dim == 0:
                dim_sum = np.sum(lesion_data, axis=(1, 2))
            elif dim == 1:
                dim_sum = np.sum(lesion_data, axis=(0, 2))
            else:
                dim_sum = np.sum(lesion_data, axis=(0, 1))
            
            # Find indices of slices with lesions
            indices = np.where(dim_sum > 0)[0]
            if len(indices) > 0:
                # Take up to 5 slices with lesions
                step = max(1, len(indices) // 5)
                lesion_slices[dim] = indices[::step][:5]
            else:
                # If no lesions, take middle slice
                lesion_slices[dim] = [orig_shape[dim] // 2]
        
        # For each dimension
        for dim in range(3):
            # For each selected slice
            for slice_idx in lesion_slices[dim]:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Extract slice from each volume
                if dim == 0:
                    ph_slice = ph_data[slice_idx, :, :]
                    adc_slice = adc_data[slice_idx, :, :]
                    lesion_slice = lesion_data[slice_idx, :, :]
                    output_slice = reconstructed_volume[slice_idx, :, :]
                elif dim == 1:
                    ph_slice = ph_data[:, slice_idx, :]
                    adc_slice = adc_data[:, slice_idx, :]
                    lesion_slice = lesion_data[:, slice_idx, :]
                    output_slice = reconstructed_volume[:, slice_idx, :]
                else:
                    ph_slice = ph_data[:, :, slice_idx]
                    adc_slice = adc_data[:, :, slice_idx]
                    lesion_slice = lesion_data[:, :, slice_idx]
                    output_slice = reconstructed_volume[:, :, slice_idx]
                
                # Display slices
                axes[0].imshow(ph_slice, cmap='gray')
                axes[0].set_title(f'Pseudo-Healthy (dim{dim}, slice {slice_idx})')
                
                axes[1].imshow(adc_slice, cmap='gray')
                axes[1].set_title('Target ADC')
                
                # Show lesion as overlay
                axes[2].imshow(ph_slice, cmap='gray')
                lesion_mask = np.ma.masked_where(lesion_slice < 0.5, lesion_slice)
                axes[2].imshow(lesion_mask, cmap='hot', alpha=0.7)
                axes[2].set_title('Lesion Mask')
                
                axes[3].imshow(output_slice, cmap='gray')
                axes[3].set_title('Model Output')
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    
    print(f"Visualization saved to: {pdf_path}")
    
    # Save reconstructed volume as NIfTI
    nii_path = os.path.join(output_dir, f'subject_{subject_id}_epoch_{epoch}.nii.gz')
    nii_img = nib.Nifti1Image(reconstructed_volume, subject_data['affine'])
    nib.save(nii_img, nii_path)
    print(f"Reconstructed volume saved to: {nii_path}")

def convert_dicom_to_array(dicom_directory):
    """
    Convert a directory of DICOM files to a 3D numpy array.
    
    Args:
        dicom_directory: Path to directory containing DICOM files for a single volume
        
    Returns:
        tuple: (data_array, affine_matrix)
    """
    try:
        # Read the DICOM directory
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory)
        if not dicom_names:
            raise ValueError(f"No DICOM series found in {dicom_directory}")
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Get the image data
        data = sitk.GetArrayFromImage(image)
        
        # Create affine matrix
        direction = np.array(image.GetDirection()).reshape(3, 3)
        spacing = np.array(image.GetSpacing())
        origin = np.array(image.GetOrigin())
        
        affine = np.eye(4)
        affine[:3, :3] = direction * np.expand_dims(spacing, 1)
        affine[:3, 3] = origin
        
        # Create a wrapper to mimic nibabel's NiftiImage
        class DicomImage:
            def __init__(self, data, affine, shape):
                self.data = data
                self.affine = affine
                self.shape = shape
            
            def get_fdata(self):
                return self.data
        
        return DicomImage(data, affine, data.shape)
        
    except Exception as e:
        raise Exception(f"Failed to convert DICOM directory {dicom_directory}: {e}")

def create_dataset_csv(pseudo_healthy_dir, adc_dir, label_dir, output_csv_path, file_type="mha"):
    """
    Creates a CSV file with paths to corresponding files for training.
    
    Args:
        pseudo_healthy_dir: Directory with pseudo-healthy ADC images (.mha or DICOM dirs)
        adc_dir: Directory with target ADC images (.mha or DICOM dirs)
        label_dir: Directory with lesion masks (.mha or DICOM dirs)
        output_csv_path: Path to output CSV file
        file_type: Type of files to look for ('mha', 'nii', 'nii.gz', or 'dicom')
    
    Returns:
        Path to created CSV file
    """
    import glob
    import re
    import os
    import pandas as pd
    
    print(f"Creating dataset CSV file...")
    print(f"Pseudo-healthy directory: {pseudo_healthy_dir}")
    print(f"ADC directory: {adc_dir}")
    print(f"Lesion mask directory: {label_dir}")
    print(f"File type: {file_type}")
    
    # Check if directories exist
    for directory in [pseudo_healthy_dir, adc_dir, label_dir]:
        if not os.path.exists(directory):
            raise ValueError(f"Directory not found: {directory}")
    
    # Get lists of files based on file type
    if file_type.lower() == 'dicom':
        # For DICOM, we look for directories that contain DICOM files
        ph_dirs = [d for d in os.listdir(pseudo_healthy_dir) 
                  if os.path.isdir(os.path.join(pseudo_healthy_dir, d))]
        ph_files = [os.path.join(pseudo_healthy_dir, d) for d in ph_dirs]
        
        adc_dirs = [d for d in os.listdir(adc_dir) 
                  if os.path.isdir(os.path.join(adc_dir, d))]
        adc_files = [os.path.join(adc_dir, d) for d in adc_dirs]
        
        lesion_dirs = [d for d in os.listdir(label_dir) 
                      if os.path.isdir(os.path.join(label_dir, d))]
        lesion_files = [os.path.join(label_dir, d) for d in lesion_dirs]
    else:
        # For other formats, look for files with the right extension
        ph_files = sorted(glob.glob(os.path.join(pseudo_healthy_dir, f"*.{file_type}")))
        adc_files = sorted(glob.glob(os.path.join(adc_dir, f"*.{file_type}")))
        lesion_files = sorted(glob.glob(os.path.join(label_dir, f"*.{file_type}")))
    
    print(f"Found files: Pseudo-healthy: {len(ph_files)}, ADC: {len(adc_files)}, Lesions: {len(lesion_files)}")
    
    # Verify that files can be opened
    valid_ph_files = []
    for file_path in ph_files:
        try:
            if file_type.lower() == 'dicom':
                # Check if the directory contains valid DICOM files
                dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path)
                if dicom_names:
                    valid_ph_files.append(file_path)
                else:
                    print(f"Warning: No DICOM series found in {file_path}")
            else:
                # Try to open with SimpleITK
                sitk.ReadImage(file_path)
                valid_ph_files.append(file_path)
        except Exception as e:
            print(f"Warning: Could not open {os.path.basename(file_path)}: {e}")
    
    valid_adc_files = []
    for file_path in adc_files:
        try:
            if file_type.lower() == 'dicom':
                dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path)
                if dicom_names:
                    valid_adc_files.append(file_path)
                else:
                    print(f"Warning: No DICOM series found in {file_path}")
            else:
                sitk.ReadImage(file_path)
                valid_adc_files.append(file_path)
        except Exception as e:
            print(f"Warning: Could not open {os.path.basename(file_path)}: {e}")
    
    valid_lesion_files = []
    for file_path in lesion_files:
        try:
            if file_type.lower() == 'dicom':
                dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path)
                if dicom_names:
                    valid_lesion_files.append(file_path)
                else:
                    print(f"Warning: No DICOM series found in {file_path}")
            else:
                sitk.ReadImage(file_path)
                valid_lesion_files.append(file_path)
        except Exception as e:
            print(f"Warning: Could not open {os.path.basename(file_path)}: {e}")
    
    print(f"Valid files: Pseudo-healthy: {len(valid_ph_files)}/{len(ph_files)}, ADC: {len(valid_adc_files)}/{len(adc_files)}, Lesions: {len(valid_lesion_files)}/{len(lesion_files)}")
    
    # Use valid files for matching
    ph_files = valid_ph_files
    adc_files = valid_adc_files
    lesion_files = valid_lesion_files
    
    # Try to match files based on common ID patterns
    matched_data = []
    
    # Try to match using common ID pattern in filenames
    for ph_file in ph_files:
        ph_basename = os.path.basename(ph_file)
        
        # Try to extract ID from filename using regex - adjust pattern as needed
        # This looks for patterns like "MGHNICU_123-VISIT_456" in filenames
        match = re.search(r'(MGHNICU_\d+-VISIT_\d+)', ph_basename)
        
        if match:
            pattern_id = match.group(1)
            # Look for corresponding ADC and lesion files
            adc_matches = [f for f in adc_files if pattern_id in os.path.basename(f)]
            lesion_matches = [f for f in lesion_files if pattern_id in os.path.basename(f)]
            
            if adc_matches and lesion_matches:
                matched_data.append({
                    'subject_id': pattern_id,
                    'pseudo_healthy': ph_file,
                    'adc': adc_matches[0],
                    'lesion': lesion_matches[0]
                })
                print(f"Found match for: {pattern_id}")
    
    # If no matches found with regex, try more flexible matching
    if not matched_data:
        print("No matches found with ID pattern. Trying more flexible matching...")
        
        # For each pseudo-healthy file
        for ph_file in ph_files:
            ph_basename = os.path.basename(ph_file)
            
            # Extract base name without extension and specific markers
            base_name = ph_basename.replace('-PSEUDO_HEALTHY.mha', '').replace('_PSEUDO_HEALTHY.mha', '')
            if file_type.lower() == 'dicom':
                # For DICOM dirs, the basename might be different
                base_name = ph_basename
            
            # Find corresponding files by base name
            for adc_file in adc_files:
                adc_basename = os.path.basename(adc_file)
                
                # Check if ADC file contains the base name or vice versa
                if base_name in adc_basename or adc_basename in base_name:
                    # Look for corresponding lesion file
                    for lesion_file in lesion_files:
                        lesion_basename = os.path.basename(lesion_file)
                        
                        # Check if lesion file contains the base name or ADC name
                        if (base_name in lesion_basename or adc_basename in lesion_basename or 
                            lesion_basename in base_name or lesion_basename in adc_basename):
                            matched_data.append({
                                'subject_id': base_name,
                                'pseudo_healthy': ph_file,
                                'adc': adc_file,
                                'lesion': lesion_file
                            })
                            break  # After finding lesion, move to next ADC
    
    # If still no matches, try simple index-based matching (last resort)
    if not matched_data:
        print("No matches found with flexible matching. Using simple index-based matching...")
        
        # Get the minimum number of files among the three directories
        min_files = min(len(ph_files), len(adc_files), len(lesion_files))
        
        for i in range(min_files):
            matched_data.append({
                'subject_id': f'subject_{i+1}',
                'pseudo_healthy': ph_files[i],
                'adc': adc_files[i],
                'lesion': lesion_files[i]
            })
    
    # Create dataframe and save to CSV
    df = pd.DataFrame(matched_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Created CSV file with {len(df)} records at: {output_csv_path}")
    
    if len(df) == 0:
        print("WARNING: No valid matches found! Please check your data files.")
    
    return output_csv_path

def train_model(args):
    """
    Main training function for MRI autoencoder model.
    
    Args:
        args: Command line arguments/configuration
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
    print(f"Using device: {device}")
    
    # Set up paths
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # Create dataset and dataloader
    dataset = BrainLesionDataset(args.csv_path, patch_size=tuple(args.patch_size))
    
    # Split dataset into train and validation sets
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if 'cuda' in args.device else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if 'cuda' in args.device else False
    )
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Create model
    model = MRIAutoencoder(args.model_path, device=device)
    model.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_lesion_metrics = {'lesion_mae': 0.0, 'input_output_lesion_mae': 0.0}
        train_samples = 0
        
        for batch in train_loader:
            inputs = batch['pseudo_healthy'].to(device)
            targets = batch['adc'].to(device)
            masks = batch['lesion'].to(device)
            
            # Forward pass
            outputs = model(inputs, masks)
            
            # Calculate loss
            loss = weighted_lesion_loss(outputs, targets, masks, lesion_weight=args.lesion_weight)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            
            # Calculate lesion-specific metrics
            metrics = calculate_lesion_metrics(outputs, targets, masks, inputs)
            for k, v in metrics.items():
                train_lesion_metrics[k] += v * batch_size
            
            train_samples += batch_size
        
        # Average losses and metrics
        train_loss /= train_samples
        for k in train_lesion_metrics:
            train_lesion_metrics[k] /= train_samples
        
        # Log training metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        for k, v in train_lesion_metrics.items():
            writer.add_scalar(f'Metrics/{k}/train', v, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_lesion_metrics = {'lesion_mae': 0.0, 'input_output_lesion_mae': 0.0}
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['pseudo_healthy'].to(device)
                targets = batch['adc'].to(device)
                masks = batch['lesion'].to(device)
                
                # Forward pass
                outputs = model(inputs, masks)
                
                # Apply smoothing
                smoothed_outputs = apply_smoothing(outputs, masks, inputs)
                
                # Calculate loss
                loss = weighted_lesion_loss(smoothed_outputs, targets, masks, lesion_weight=args.lesion_weight)
                
                # Track metrics
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                
                # Calculate lesion-specific metrics
                metrics = calculate_lesion_metrics(smoothed_outputs, targets, masks, inputs)
                for k, v in metrics.items():
                    val_lesion_metrics[k] += v * batch_size
                
                val_samples += batch_size
        
        # Average losses and metrics
        val_loss /= val_samples
        for k in val_lesion_metrics:
            val_lesion_metrics[k] /= val_samples
        
        # Log validation metrics
        writer.add_scalar('Loss/val', val_loss, epoch)
        for k, v in val_lesion_metrics.items():
            writer.add_scalar(f'Metrics/{k}/val', v, epoch)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Train Lesion MAE: {train_lesion_metrics['lesion_mae']:.4f}, Val Lesion MAE: {val_lesion_metrics['lesion_mae']:.4f}")
        
        # Save checkpoint if we have the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model checkpoint to: {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint to: {checkpoint_path}")
            
            # Visualize random validation subjects
            if args.visualize:
                unique_subjects = set()
                for batch in val_loader:
                    for subject_id in batch['subject_id']:
                        unique_subjects.add(subject_id)
                
                # Pick up to 3 random subjects to visualize
                subjects_to_viz = list(unique_subjects)[:3]
                for subject_id in subjects_to_viz:
                    visualize_full_subject(dataset, model, subject_id, epoch+1, args.output_dir, device)
    
    writer.close()
    print("Training completed!")

def main():
    """
    Main function to parse arguments and start training.
    """
    parser = argparse.ArgumentParser(description="Train MRI autoencoder for lesion synthesis")
    
    # Data directories and output paths
    parser.add_argument('--pseudo_healthy_dir', type=str, default=None,
                        help='Directory containing pseudo-healthy ADC images (.mha)')
    parser.add_argument('--adc_dir', type=str, default=None, 
                        help='Directory containing target ADC images (.mha)')
    parser.add_argument('--label_dir', type=str, default=None,
                        help='Directory containing lesion masks (.mha)')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to existing CSV file with dataset information. If not provided, will be created from directories.')
    parser.add_argument('--model_path', type=str, default="microsoft/mri-autoencoder-v0.1",
                        help='Path or HF model ID for the MRI autoencoder model')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save model checkpoints, visualizations and outputs')
    parser.add_argument('--file_type', type=str, default='mha',
                        help='Type of files to process: mha, nii, nii.gz, or dicom')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lesion_weight', type=float, default=10.0,
                        help='Weight for lesion areas in loss function')
    parser.add_argument('--patch_size', nargs=3, type=int, default=[80, 96, 80],
                        help='Patch size for 3D patches [depth, height, width]')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to train on (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                        help='Frequency of saving model checkpoints (in epochs)')
    parser.add_argument('--visualize', action='store_true',
                        help='Whether to generate visualizations during training')
    
    args = parser.parse_args()
    
    # Either use existing CSV or create a new one from directories
    if args.csv_path is None:
        # Check if directories are provided
        if args.pseudo_healthy_dir is None or args.adc_dir is None or args.label_dir is None:
            parser.error("Either provide --csv_path or all of --pseudo_healthy_dir, --adc_dir, and --label_dir")
        
        # Create CSV file
        csv_path = os.path.join(args.output_dir, 'dataset.csv')
        args.csv_path = create_dataset_csv(
            args.pseudo_healthy_dir,
            args.adc_dir,
            args.label_dir,
            csv_path,
            file_type=args.file_type
        )
    
    # Print configuration
    print("Configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Start training
    train_model(args)

if __name__ == "__main__":
    main()
