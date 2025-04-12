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

class BrainLesionDataset(Dataset):
    def __init__(self, csv_path, target_shape=(80, 96, 80), patch_overlap=0.5):
        """
        Dataset for training a patch-based model for brain lesions.
        
        Args:
            csv_path: Path to CSV file with data
            target_shape: Target patch shape (D, H, W)
            patch_overlap: Overlap between neighboring patches (0-1)
        """
        self.data = pd.read_csv(csv_path)
        self.target_shape = target_shape
        self.patch_overlap = patch_overlap
        
        # List of all patches for all subjects
        self.all_patches = []
        
        # Dictionary mapping subject_id -> original data and shapes
        self.subject_data = {}
        
        # Precompute all patches for all subjects
        print("Preparing patches for all subjects...")
        for idx, row in self.data.iterrows():
            subject_id = row['subject_id']
            print(f"Processing subject {subject_id} ({idx+1}/{len(self.data)})")
            
            # Load original data without modifications
            try:
                ph_img = nib.load(row['pseudo_healthy'])
                adc_img = nib.load(row['adc'])
                lesion_img = nib.load(row['lesion'])
                
                # Check if all images have the same affine matrix and dimensions
                if not np.allclose(ph_img.affine, adc_img.affine) or not np.allclose(ph_img.affine, lesion_img.affine):
                    print(f"  WARNING: Images for subject {subject_id} have different affine matrices! Performing resampling...")
                    
                    # Use pseudo_healthy affine as reference
                    ref_affine = ph_img.affine
                    ref_shape = ph_img.shape
                    
                    # Resampling adc and lesion images to reference image space
                    # For ADC use linear interpolation
                    adc_data = np.array(adc_img.get_fdata())
                    adc_data = np.clip(adc_data, 0, None)
                    if adc_data.max() > 0: 
                        adc_data = adc_data / adc_data.max()
                    
                    # For lesion mask use nearest-neighbor interpolation to keep it binary
                    lesion_data = np.array(lesion_img.get_fdata())
                    lesion_data = (lesion_data > 0.1).astype(np.float32)
                    
                    # Try to use nibabel resample_to_img function if available, otherwise use custom implementation
                    try:
                        from nibabel.processing import resample_to_img
                        
                        # Resampling adc image (linear interpolation)
                        new_adc_img = resample_to_img(adc_img, ph_img, interpolation='linear')
                        adc_data = np.array(new_adc_img.get_fdata())
                        
                        # Resampling lesion image (nearest neighbor to keep it binary)
                        new_lesion_img = resample_to_img(lesion_img, ph_img, interpolation='nearest')
                        lesion_data = (np.array(new_lesion_img.get_fdata()) > 0.1).astype(np.float32)
                        
                    except (ImportError, AttributeError):
                        print("  nibabel.processing.resample_to_img not available, using SciPy for resampling...")
                        
                        # Alternative resampling using scipy
                        from scipy.ndimage import map_coordinates
                        
                        # Create grid for transformation from source to target coordinates
                        ijk_dest = np.mgrid[0:ref_shape[0], 0:ref_shape[1], 0:ref_shape[2]]
                        ijk_dest = ijk_dest.reshape(3, -1)
                        
                        # Add homogeneous coordinate
                        ijk_dest_homog = np.vstack((ijk_dest, np.ones((1, ijk_dest.shape[1]))))
                        
                        # Transform from target voxel coordinates to world (mm)
                        xyz_dest = ref_affine @ ijk_dest_homog
                        
                        # Transform from world (mm) to source voxel coordinates
                        ijk_src_adc = np.linalg.inv(adc_img.affine) @ xyz_dest
                        ijk_src_lesion = np.linalg.inv(lesion_img.affine) @ xyz_dest
                        
                        # Resample data
                        adc_data_resampled = np.zeros(ref_shape)
                        lesion_data_resampled = np.zeros(ref_shape)
                        
                        # Trilinear interpolation for ADC data
                        map_coordinates(adc_data, ijk_src_adc[:3], output=adc_data_resampled.ravel(), order=1)
                        
                        # Nearest neighbor interpolation for lesion mask
                        map_coordinates(lesion_data, ijk_src_lesion[:3], output=lesion_data_resampled.ravel(), order=0)
                        
                        adc_data = adc_data_resampled
                        lesion_data = (lesion_data_resampled > 0.1).astype(np.float32)
                    
                    # Get data from pseudo_healthy
                    ph_data = np.array(ph_img.get_fdata())
                    if ph_data.max() > 0: 
                        ph_data = ph_data / ph_data.max()
                    
                else:
                    # If they have the same affine matrices, use data directly
                    ph_data = np.array(ph_img.get_fdata())
                    adc_data = np.array(adc_img.get_fdata())
                    lesion_data = np.array(lesion_img.get_fdata())
                    
                    # Normalization (only for image data)
                    if ph_data.max() > 0: ph_data = ph_data / ph_data.max()
                    if adc_data.max() > 0: adc_data = adc_data / adc_data.max()
                    
                    # Normalize masks - ensure they are binary (0 or 1)
                    lesion_data = (lesion_data > 0.1).astype(np.float32)
                
                if ph_img.shape != adc_img.shape or ph_img.shape != lesion_img.shape:
                    print(f"  WARNING: Images for subject {subject_id} have different dimensions! Using dimensions from pseudo_healthy.")
                
                # Save original data and its shape to dictionary
                self.subject_data[subject_id] = {
                    'ph_data': ph_data,
                    'adc_data': adc_data,
                    'lesion_data': lesion_data,
                    'orig_shape': ph_data.shape,
                    'affine': ph_img.affine
                }
                
                # Find all lesions and their centers
                lesion_centers = self._find_all_lesion_centers(lesion_data)
                
                if not lesion_centers:
                    print(f"  Subject {subject_id}: No lesions found, creating patch in volume center")
                    # If no lesions, create at least one patch in the center of volume
                    center = [d // 2 for d in lesion_data.shape]
                    patch_info = {
                        'subject_id': subject_id,
                        'center': center,
                        'has_lesion': False
                    }
                    self.all_patches.append(patch_info)
                else:
                    print(f"  Subject {subject_id}: Found {len(lesion_centers)} lesion centers")
                    
                    # Create patch for each lesion center
                    for i, center in enumerate(lesion_centers):
                        patch_info = {
                            'subject_id': subject_id,
                            'center': center,
                            'has_lesion': True
                        }
                        self.all_patches.append(patch_info)
                    
                    # Look for large lesions that might require multiple patches
                    labeled_lesions, num_lesions = ndimage.label(lesion_data)
                    
                    for i in range(1, num_lesions + 1):
                        lesion_mask = labeled_lesions == i
                        lesion_size = np.sum(lesion_mask)
                        
                        # If lesion is larger than half the patch size, it might require multiple patches
                        if lesion_size > np.prod(self.target_shape) * 0.5:
                            print(f"  Subject {subject_id}: Lesion {i} is large ({lesion_size} voxels), generating additional patches")
                            
                            # Find lesion boundaries
                            indices = np.where(lesion_mask)
                            min_bounds = [np.min(idx) for idx in indices]
                            max_bounds = [np.max(idx) for idx in indices]
                            
                            # Create grid of points to cover lesion
                            # Determine step between patches (considering overlap)
                            steps = [int(ts * (1 - self.patch_overlap)) for ts in self.target_shape]
                            
                            # Calculate starting positions (extended by half a patch on each side)
                            start_positions = []
                            for dim, (min_b, max_b, step, ts) in enumerate(zip(min_bounds, max_bounds, steps, self.target_shape)):
                                half_size = ts // 2
                                start = max(0, min_b - half_size)
                                end = min(lesion_data.shape[dim] - step, max_b + half_size)
                                positions = list(range(start, end, step))
                                if not positions or end > positions[-1]:
                                    positions.append(min(end, lesion_data.shape[dim] - ts))
                                start_positions.append(positions)
                            
                            # Create all combinations of starting positions for all dimensions
                            for start_z, start_y, start_x in product(start_positions[0], start_positions[1], start_positions[2]):
                                # Calculate patch center
                                center = [
                                    start_z + self.target_shape[0] // 2,
                                    start_y + self.target_shape[1] // 2,
                                    start_x + self.target_shape[2] // 2
                                ]
                                
                                # Check if patch contains at least part of the lesion
                                patch_mask = self._extract_patch(lesion_mask.astype(float), center)
                                if np.sum(patch_mask) > 0:
                                    patch_info = {
                                        'subject_id': subject_id,
                                        'center': center,
                                        'has_lesion': True,
                                        'patch_id': f"lesion{i}_z{start_z}_y{start_y}_x{start_x}"
                                    }
                                    # Add only if similar patch doesn't already exist
                                    if not any(self._is_similar_patch(p, patch_info) for p in self.all_patches):
                                        self.all_patches.append(patch_info)
            
            except Exception as e:
                print(f"  ERROR processing subject {subject_id}: {e}")
        
        print(f"Total patches created: {len(self.all_patches)} for {len(self.data)} subjects")
    
    def _is_similar_patch(self, patch1, patch2, distance_threshold=20):
        """Checks if two patches are similar (close centers)"""
        if patch1['subject_id'] != patch2['subject_id']:
            return False
            
        center1 = patch1['center']
        center2 = patch2['center']
        
        # Calculate Euclidean distance between centers
        distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(center1, center2)))
        return distance < distance_threshold
    
    def __len__(self):
        return len(self.all_patches)
    
    def __getitem__(self, idx):
        """Returns specific patch by its index"""
        patch_info = self.all_patches[idx]
        subject_id = patch_info['subject_id']
        center = patch_info['center']
        
        # Get subject data from dictionary
        subject_data = self.subject_data[subject_id]
        
        # Create patch centered on lesion
        ph_patch = self._extract_patch(subject_data['ph_data'], center)
        adc_patch = self._extract_patch(subject_data['adc_data'], center)
        lesion_patch = self._extract_patch(subject_data['lesion_data'], center)
        
        # Convert to tensors and add channel dimension
        ph_tensor = torch.from_numpy(ph_patch).float().unsqueeze(0)
        adc_tensor = torch.from_numpy(adc_patch).float().unsqueeze(0)
        lesion_tensor = torch.from_numpy(lesion_patch).float().unsqueeze(0)
        
        # Prepare patch metadata for reconstruction
        patch_meta = {
            'subject_id': subject_id,
            'center': center,
            'patch_id': patch_info.get('patch_id', f"patch_{idx}"),
            'has_lesion': patch_info['has_lesion'],
            'orig_shape': subject_data['orig_shape'],
            'patch_shape': self.target_shape
        }
        
        return {
            'input': ph_tensor,
            'target': adc_tensor,
            'mask': lesion_tensor,
            'subject_id': subject_id,
            'patch_meta': patch_meta
        }
    
    def _find_all_lesion_centers(self, mask):
        """Finds centers of all connected lesions in mask"""
        # If mask contains no lesions, return empty list
        if np.max(mask) == 0:
            return []
        
        # Label connected regions in binary mask
        binary_mask = mask > 0.1
        labeled_mask, num_features = ndimage.label(binary_mask)
        
        centers = []
        # For each connected region find center
        for i in range(1, num_features + 1):
            region_mask = labeled_mask == i
            if np.sum(region_mask) > 10:  # Ignore very small lesions (less than 10 voxels)
                center = ndimage.center_of_mass(region_mask)
                centers.append([int(c) for c in center])
        
        return centers
    
    def _extract_patch(self, data, center, padding_value=0):
        """Extracts fixed-size patch from data around specified center without interpolation"""
        # Convert center to integer indices
        center = [int(c) for c in center]
        
        # Create empty patch of target size
        patch = np.ones(self.target_shape, dtype=np.float32) * padding_value
        
        # For each dimension calculate boundaries of the crop
        data_starts = []  # Starting indices in original data
        data_ends = []    # Ending indices in original data
        patch_starts = [] # Starting indices in patch
        patch_ends = []   # Ending indices in patch
        
        for dim, (center_pos, target_size, data_size) in enumerate(zip(center, self.target_shape, data.shape)):
            # Half size of target shape for dimension
            half_size = target_size // 2
            
            # Calculate boundaries in original data
            data_start = max(0, center_pos - half_size)
            data_end = min(data_size, center_pos + half_size + (target_size % 2))
            
            # Calculate corresponding boundaries in patch
            patch_start = max(0, half_size - center_pos)
            patch_end = patch_start + (data_end - data_start)
            
            data_starts.append(data_start)
            data_ends.append(data_end)
            patch_starts.append(patch_start)
            patch_ends.append(patch_end)
        
        # Copy data from original image to patch WITHOUT INTERPOLATION
        patch[patch_starts[0]:patch_ends[0], 
              patch_starts[1]:patch_ends[1], 
              patch_starts[2]:patch_ends[2]] = data[data_starts[0]:data_ends[0], 
                                                   data_starts[1]:data_ends[1], 
                                                   data_starts[2]:data_ends[2]]
        
        return patch
    
    def get_full_subject_data(self, subject_id):
        """Returns original data for given subject"""
        return self.subject_data.get(subject_id, None)
    
    def get_patch_locations(self, subject_id):
        """Returns list of all patch locations for given subject"""
        return [p for p in self.all_patches if p['subject_id'] == subject_id]
    
    def reconstruct_full_volume(self, patches, patch_centers, orig_shape, blend_weights=True):
        """
        Reconstructs full volume from patches.
        
        Args:
            patches: List of patches (numpy arrays)
            patch_centers: List of patch centers in original volume
            orig_shape: Shape of original volume
            blend_weights: Whether to use weighted averaging for overlapping areas
            
        Returns:
            Reconstructed volume
        """
        # Initialize empty output volume and weight volume for weighted averaging
        output_volume = np.zeros(orig_shape, dtype=np.float32)
        weight_volume = np.zeros(orig_shape, dtype=np.float32)
        
        # For each patch
        for patch, center in zip(patches, patch_centers):
            # Create weight mask for this patch
            if blend_weights:
                # Create Gaussian weight, higher in patch center and lower at edges
                weight_mask = np.ones(self.target_shape, dtype=np.float32)
                
                # For each dimension create Gaussian profile
                for dim in range(3):
                    # Create coordinate grid for dimension
                    coords = np.linspace(-1, 1, self.target_shape[dim])
                    
                    # Gaussian profile: exp(-x^2/sigma^2)
                    sigma = 0.4  # Width of Gaussian curve
                    gauss_profile = np.exp(-(coords**2) / (2 * sigma**2))
                    
                    # Expand profile to right shape for multiplication
                    if dim == 0:
                        gauss_profile = gauss_profile.reshape(-1, 1, 1)
                    elif dim == 1:
                        gauss_profile = gauss_profile.reshape(1, -1, 1)
                    else:
                        gauss_profile = gauss_profile.reshape(1, 1, -1)
                    
                    # Multiply current weight mask by Gaussian profile
                    weight_mask = weight_mask * gauss_profile
                
                # Normalize weights to 0-1 range
                weight_mask = weight_mask / np.max(weight_mask)
            else:
                # Uniform weight for entire patch
                weight_mask = np.ones(self.target_shape, dtype=np.float32)
            
            # Calculate boundaries for inserting patch into original volume
            data_starts = []
            data_ends = []
            patch_starts = []
            patch_ends = []
            
            for dim, (center_pos, target_size) in enumerate(zip(center, self.target_shape)):
                half_size = target_size // 2
                
                # Calculate boundaries in original volume
                data_start = max(0, center_pos - half_size)
                data_end = min(orig_shape[dim], center_pos + half_size + (target_size % 2))
                
                # Calculate corresponding boundaries in patch
                patch_start = max(0, half_size - center_pos)
                patch_end = patch_start + (data_end - data_start)
                
                data_starts.append(data_start)
                data_ends.append(data_end)
                patch_starts.append(patch_start)
                patch_ends.append(patch_end)
            
            # Add weighted patch to output volume
            output_volume[data_starts[0]:data_ends[0], 
                          data_starts[1]:data_ends[1], 
                          data_starts[2]:data_ends[2]] += \
                patch[patch_starts[0]:patch_ends[0], 
                      patch_starts[1]:patch_ends[1], 
                      patch_starts[2]:patch_ends[2]] * \
                weight_mask[patch_starts[0]:patch_ends[0], 
                           patch_starts[1]:patch_ends[1], 
                           patch_starts[2]:patch_ends[2]]
            
            # Add weights to weight volume
            weight_volume[data_starts[0]:data_ends[0], 
                          data_starts[1]:data_ends[1], 
                          data_starts[2]:data_ends[2]] += \
                weight_mask[patch_starts[0]:patch_ends[0], 
                           patch_starts[1]:patch_ends[1], 
                           patch_starts[2]:patch_ends[2]]
        
        # Normalize output volume by weights (avoid division by zero)
        valid_mask = weight_volume > 0
        output_volume[valid_mask] = output_volume[valid_mask] / weight_volume[valid_mask]
        
        return output_volume

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

def create_dataset_csv(pseudo_healthy_dir, adc_dir, label_dir, output_csv_path):
    """
    Creates a CSV file with paths to corresponding files for training.
    
    Args:
        pseudo_healthy_dir: Directory with pseudo-healthy ADC images (.mha)
        adc_dir: Directory with target ADC images (.mha)
        label_dir: Directory with lesion masks (.mha)
        output_csv_path: Path to output CSV file
    
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
    
    # Check if directories exist
    for directory in [pseudo_healthy_dir, adc_dir, label_dir]:
        if not os.path.exists(directory):
            raise ValueError(f"Directory not found: {directory}")
    
    # Get lists of files
    ph_files = sorted(glob.glob(os.path.join(pseudo_healthy_dir, "*.mha")))
    adc_files = sorted(glob.glob(os.path.join(adc_dir, "*.mha")))
    lesion_files = sorted(glob.glob(os.path.join(label_dir, "*.mha")))
    
    print(f"Found files: Pseudo-healthy: {len(ph_files)}, ADC: {len(adc_files)}, Lesions: {len(lesion_files)}")
    
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
            adc_pattern = os.path.join(adc_dir, f"{pattern_id}*ADC_ss.mha")
            lesion_pattern = os.path.join(label_dir, f"{pattern_id}*lesion.mha")
            
            adc_matches = glob.glob(adc_pattern)
            lesion_matches = glob.glob(lesion_pattern)
            
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
            
            # Find corresponding files by base name
            for adc_file in adc_files:
                adc_basename = os.path.basename(adc_file)
                
                # Check if ADC file contains the base name
                if base_name in adc_basename:
                    # Look for corresponding lesion file
                    for lesion_file in lesion_files:
                        lesion_basename = os.path.basename(lesion_file)
                        
                        # Check if lesion file contains the base name
                        if base_name in lesion_basename:
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
    dataset = BrainLesionDataset(args.csv_path, target_shape=tuple(args.patch_size))
    
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
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            masks = batch['mask'].to(device)
            
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
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                masks = batch['mask'].to(device)
                
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
            csv_path
        )
    
    # Print configuration
    print("Configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Start training
    train_model(args)

if __name__ == "__main__":
    main()
