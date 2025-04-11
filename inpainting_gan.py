import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import monai
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    RandRotated,
    RandAffined,
    RandScaleIntensityd,
    EnsureTyped,
    EnsureChannelFirstd,
    RandFlipd
)
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import SimpleITK as sitk
import glob
import random
from tqdm import tqdm
import datetime

class LesionInpaintingDataset(Dataset):
    def __init__(self, pseudo_healthy_dir, adc_dir, label_dir, transform=None, train=True):
        self.transform = transform
        self.train = train
        
        # Get all pseudo-healthy files
        pseudo_healthy_files = sorted(glob.glob(os.path.join(pseudo_healthy_dir, "*PSEUDO_HEALTHY.mha")))
        
        # Create a list of data dictionaries
        self.data = []
        for ph_file in pseudo_healthy_files:
            # Extract the patient ID from the filename
            patient_id = os.path.basename(ph_file).split('-PSEUDO_HEALTHY.mha')[0]
            
            # Find corresponding ADC and label files
            adc_file = os.path.join(adc_dir, f"{patient_id}-ADC_ss.mha")
            label_file = os.path.join(label_dir, f"{patient_id}_lesion.mha")
            
            if os.path.exists(adc_file) and os.path.exists(label_file):
                self.data.append({
                    "pseudo_healthy_file": ph_file,
                    "adc_file": adc_file,
                    "label_file": label_file
                })
        
        # Optional: Cache data for faster loading
        self.use_cache = False
        self.data_cache = {}
        
        print(f"{'Training' if train else 'Validation'} dataset contains {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_dict = self.data[idx]
        
        # Check if data is already cached
        if self.use_cache and idx in self.data_cache:
            return self.data_cache[idx]
        
        # Load data using SimpleITK with explicit reader creation (faster for repeated loads)
        pseudo_healthy_img = sitk.ReadImage(file_dict["pseudo_healthy_file"])
        adc_img = sitk.ReadImage(file_dict["adc_file"])
        label_img = sitk.ReadImage(file_dict["label_file"])
        
        # Convert to numpy arrays
        pseudo_healthy = sitk.GetArrayFromImage(pseudo_healthy_img).astype(np.float32)
        adc = sitk.GetArrayFromImage(adc_img).astype(np.float32)
        label = sitk.GetArrayFromImage(label_img).astype(np.float32)
        
        # Capture original ranges BEFORE normalization
        ph_original_range = (float(np.min(pseudo_healthy)), float(np.max(pseudo_healthy)))
        adc_original_range = (float(np.min(adc)), float(np.max(adc)))
        
        # Add channel dimension
        pseudo_healthy = np.expand_dims(pseudo_healthy, axis=0)
        adc = np.expand_dims(adc, axis=0)
        label = np.expand_dims(label, axis=0)
        
        # Convert to PyTorch tensors
        pseudo_healthy = torch.from_numpy(pseudo_healthy)
        adc = torch.from_numpy(adc)
        label = torch.from_numpy(label)
        
        # Create data dictionary
        data_dict = {
            "pseudo_healthy": pseudo_healthy,
            "adc": adc,
            "label": label,
            "pseudo_healthy_meta": {
                "filename": file_dict["pseudo_healthy_file"],
                "origin": pseudo_healthy_img.GetOrigin(),
                "spacing": pseudo_healthy_img.GetSpacing(),
                "direction": pseudo_healthy_img.GetDirection()
            },
            "original_ranges": {
                "pseudo_healthy": ph_original_range,
                "adc": adc_original_range
            }
        }
        
        # Apply additional transforms if provided
        if self.transform:
            data_dict = self.transform(data_dict)
        
        # Cache the data if enabled
        if self.use_cache:
            self.data_cache[idx] = data_dict
            
        return data_dict
        
    def enable_cache(self, enabled=True):
        """Enable or disable data caching to avoid reloading from disk"""
        self.use_cache = enabled
        if not enabled:
            # Clear the cache if disabling
            self.data_cache = {}
            
    def clear_cache(self):
        """Clear the data cache"""
        self.data_cache = {}

class PatchExtractor:
    def __init__(self, patch_size=(64, 64, 32), overlap=0.5, augment_patches=True, num_augmented_patches=3):
        self.patch_size = patch_size
        self.overlap = overlap
        self.augment_patches = augment_patches
        self.num_augmented_patches = num_augmented_patches
        
        # Cache to avoid reconstructing multiple times
        self.patches_cache = {}
        # Cache for fully processed volumes
        self.reconstructed_cache = {}
    
    def extract_patches_with_lesions(self, pseudo_healthy, label, adc, max_attempts=3, current_attempt=0):
        """
        Extract patches that contain lesions
        
        Args:
            pseudo_healthy: tensor of shape [C, D, H, W]
            label: tensor of shape [C, D, H, W]
            adc: tensor of shape [C, D, H, W]
            max_attempts: maximum number of recursive attempts
            current_attempt: current recursive attempt count
            
        Returns:
            patches_ph: list of pseudo_healthy patches
            patches_label: list of label patches
            patches_adc: list of adc patches
            patch_coords: list of patch coordinates (for reconstruction)
        """
        # Create a cache key based on tensor content hash
        # This avoids recomputing patches for the same input data
        cache_key = self._generate_cache_key(pseudo_healthy, label, adc)
        
        # Check if we already have these patches cached
        if cache_key in self.patches_cache:
            # Return cached results
            return self.patches_cache[cache_key]
        
        # Get non-zero indices from the label
        non_zero_indices = torch.nonzero(label[0])
        
        if len(non_zero_indices) == 0 or current_attempt >= max_attempts:
            # If no lesion or max attempts reached, return a single random patch
            d, h, w = pseudo_healthy.shape[1:]
            z_start = random.randint(0, max(0, d - self.patch_size[0]))
            y_start = random.randint(0, max(0, h - self.patch_size[1]))
            x_start = random.randint(0, max(0, w - self.patch_size[2]))
            
            ph_patch = pseudo_healthy[:, 
                                     z_start:z_start+self.patch_size[0], 
                                     y_start:y_start+self.patch_size[1], 
                                     x_start:x_start+self.patch_size[2]]
            
            label_patch = label[:, 
                              z_start:z_start+self.patch_size[0], 
                              y_start:y_start+self.patch_size[1], 
                              x_start:x_start+self.patch_size[2]]
            
            adc_patch = adc[:, 
                          z_start:z_start+self.patch_size[0], 
                          y_start:y_start+self.patch_size[1], 
                          x_start:x_start+self.patch_size[2]]
            
            # Pad if necessary
            if ph_patch.shape[1:] != self.patch_size:
                ph_patch = self._pad_patch(ph_patch)
                label_patch = self._pad_patch(label_patch)
                adc_patch = self._pad_patch(adc_patch)
            
            result = [ph_patch], [label_patch], [adc_patch], [(z_start, y_start, x_start)]
            self.patches_cache[cache_key] = result
            return result
        
        # Get bounding box of the lesion to ensure we capture entire lesions
        z_indices = non_zero_indices[:, 0]
        y_indices = non_zero_indices[:, 1]
        x_indices = non_zero_indices[:, 2]
        
        z_min, z_max = z_indices.min().item(), z_indices.max().item()
        y_min, y_max = y_indices.min().item(), y_indices.max().item()
        x_min, x_max = x_indices.min().item(), x_indices.max().item()
        
        # Calculate lesion center
        z_center = (z_min + z_max) // 2
        y_center = (y_min + y_max) // 2
        x_center = (x_min + x_max) // 2
        
        # Cluster lesion voxels to find centers of multiple lesions
        lesion_centers = self._cluster_lesion_voxels(non_zero_indices)
        
        patches_ph, patches_label, patches_adc, patch_coords = [], [], [], []
        
        # Extract patches around each lesion center
        for center in lesion_centers:
            z, y, x = center
            
            # Calculate base patch coordinates centered on the lesion
            # Ensure that coordinates are positive and within volume boundaries
            z_start = max(0, z - self.patch_size[0] // 2)
            y_start = max(0, y - self.patch_size[1] // 2)
            x_start = max(0, x - self.patch_size[2] // 2)
            
            # Adjust if patch exceeds volume boundaries
            z_start = min(z_start, max(0, pseudo_healthy.shape[1] - self.patch_size[0]))
            y_start = min(y_start, max(0, pseudo_healthy.shape[2] - self.patch_size[1]))
            x_start = min(x_start, max(0, pseudo_healthy.shape[3] - self.patch_size[2]))
            
            # Extract base patch
            self._extract_and_add_patch(
                pseudo_healthy, label, adc, 
                z_start, y_start, x_start,
                patches_ph, patches_label, patches_adc, patch_coords
            )
            
            # Generate augmented patches with different offsets
            if self.augment_patches:
                self._generate_augmented_patches(
                    pseudo_healthy, label, adc,
                    z, y, x,  # Lesion center
                    patches_ph, patches_label, patches_adc, patch_coords
                )
        
        # If no patches with lesions were found, try again with a slight variation
        # but limit the number of recursive attempts to avoid stack overflow
        if len(patches_ph) == 0 and current_attempt < max_attempts:
            # Apply small random shift to non-zero indices to try to find lesions
            if len(non_zero_indices) > 0:
                for i in range(len(non_zero_indices)):
                    non_zero_indices[i, 0] += random.randint(-5, 5)
                    non_zero_indices[i, 1] += random.randint(-5, 5)
                    non_zero_indices[i, 2] += random.randint(-5, 5)
            
            return self.extract_patches_with_lesions(pseudo_healthy, label, adc, max_attempts, current_attempt + 1)
        elif len(patches_ph) == 0:
            # If still no patches found, return a single random patch
            d, h, w = pseudo_healthy.shape[1:]
            z_start = random.randint(0, max(0, d - self.patch_size[0]))
            y_start = random.randint(0, max(0, h - self.patch_size[1]))
            x_start = random.randint(0, max(0, w - self.patch_size[2]))
            
            ph_patch = pseudo_healthy[:, 
                                     z_start:z_start+self.patch_size[0], 
                                     y_start:y_start+self.patch_size[1], 
                                     x_start:x_start+self.patch_size[2]]
            
            label_patch = label[:, 
                              z_start:z_start+self.patch_size[0], 
                              y_start:y_start+self.patch_size[1], 
                              x_start:x_start+self.patch_size[2]]
            
            adc_patch = adc[:, 
                          z_start:z_start+self.patch_size[0], 
                          y_start:y_start+self.patch_size[1], 
                          x_start:x_start+self.patch_size[2]]
            
            # Pad if necessary
            if ph_patch.shape[1:] != self.patch_size:
                ph_patch = self._pad_patch(ph_patch)
                label_patch = self._pad_patch(label_patch)
                adc_patch = self._pad_patch(adc_patch)
            
            result = [ph_patch], [label_patch], [adc_patch], [(z_start, y_start, x_start)]
            self.patches_cache[cache_key] = result
            return result
        
        result = patches_ph, patches_label, patches_adc, patch_coords
        self.patches_cache[cache_key] = result
        return result
    
    def _extract_and_add_patch(self, pseudo_healthy, label, adc, z_start, y_start, x_start, 
                              patches_ph, patches_label, patches_adc, patch_coords):
        """Helper method to extract a patch and add it to collections if it contains lesions"""
        # Handle negative coordinates - clip to 0
        z_start = max(0, z_start)
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        
        # Handle out-of-bounds coordinates
        if (z_start >= pseudo_healthy.shape[1] or 
            y_start >= pseudo_healthy.shape[2] or 
            x_start >= pseudo_healthy.shape[3]):
            print(f"Warning: Skipping out-of-bounds patch coordinates: ({z_start}, {y_start}, {x_start})")
            return False
            
        # Calculate actual patch size to avoid out-of-bounds
        z_end = min(z_start + self.patch_size[0], pseudo_healthy.shape[1])
        y_end = min(y_start + self.patch_size[1], pseudo_healthy.shape[2])
        x_end = min(x_start + self.patch_size[2], pseudo_healthy.shape[3])
        
        # Extract patches
        ph_patch = pseudo_healthy[:, 
                                 z_start:z_end, 
                                 y_start:y_end, 
                                 x_start:x_end]
        
        label_patch = label[:, 
                          z_start:z_end, 
                          y_start:y_end, 
                          x_start:x_end]
        
        adc_patch = adc[:, 
                      z_start:z_end, 
                      y_start:y_end, 
                      x_start:x_end]
        
        # Check if patches have expected dimensions
        expected_dim = 4  # [C, Z, Y, X]
        if ph_patch.dim() != expected_dim:
            print(f"Warning: Patch has unexpected dimension {ph_patch.dim()}, expected {expected_dim}")
            return False
        
        # Pad if necessary to match required patch size
        if (ph_patch.shape[1] != self.patch_size[0] or 
            ph_patch.shape[2] != self.patch_size[1] or 
            ph_patch.shape[3] != self.patch_size[2]):
            ph_patch = self._pad_patch(ph_patch)
            label_patch = self._pad_patch(label_patch)
            adc_patch = self._pad_patch(adc_patch)
        
        # Validate patch dimensions after padding
        if (ph_patch.shape[1] != self.patch_size[0] or 
            ph_patch.shape[2] != self.patch_size[1] or 
            ph_patch.shape[3] != self.patch_size[2]):
            print(f"Warning: After padding, patch still has incorrect shape: {ph_patch.shape}, expected: [C, {self.patch_size[0]}, {self.patch_size[1]}, {self.patch_size[2]}]")
            return False
        
        # Only add patch if it contains lesion
        if label_patch.sum() > 0:
            patches_ph.append(ph_patch)
            patches_label.append(label_patch)
            patches_adc.append(adc_patch)
            patch_coords.append((z_start, y_start, x_start))
            return True
        return False
    
    def _generate_augmented_patches(self, pseudo_healthy, label, adc, center_z, center_y, center_x, 
                                   patches_ph, patches_label, patches_adc, patch_coords):
        """Generate multiple augmented patches around a lesion center with various offsets"""
        # Define various offsets to capture the lesion from different angles
        # These offsets are relative to the center and give different context around the lesion
        offsets = []
        
        # Add offsets with different shifts in each dimension
        for i in range(self.num_augmented_patches):
            # Random offset in each dimension, but not too large to lose the lesion
            z_offset = random.randint(-self.patch_size[0]//4, self.patch_size[0]//4)
            y_offset = random.randint(-self.patch_size[1]//4, self.patch_size[1]//4)
            x_offset = random.randint(-self.patch_size[2]//4, self.patch_size[2]//4)
            offsets.append((z_offset, y_offset, x_offset))
        
        # Add diagonal offsets to see lesion from corner views
        offsets.extend([
            (self.patch_size[0]//6, self.patch_size[1]//6, self.patch_size[2]//6),
            (-self.patch_size[0]//6, -self.patch_size[1]//6, -self.patch_size[2]//6),
            (self.patch_size[0]//6, -self.patch_size[1]//6, self.patch_size[2]//6),
            (-self.patch_size[0]//6, self.patch_size[1]//6, -self.patch_size[2]//6)
        ])
        
        # Generate augmented patches with each offset
        for z_offset, y_offset, x_offset in offsets:
            # Calculate new patch start coordinates with offset
            z_start = max(0, center_z - self.patch_size[0]//2 + z_offset)
            y_start = max(0, center_y - self.patch_size[1]//2 + y_offset)
            x_start = max(0, center_x - self.patch_size[2]//2 + x_offset)
            
            # Adjust if patch exceeds volume boundaries
            z_start = min(z_start, max(0, pseudo_healthy.shape[1] - self.patch_size[0]))
            y_start = min(y_start, max(0, pseudo_healthy.shape[2] - self.patch_size[1]))
            x_start = min(x_start, max(0, pseudo_healthy.shape[3] - self.patch_size[2]))
            
            # Extract and add the patch
            self._extract_and_add_patch(
                pseudo_healthy, label, adc, 
                z_start, y_start, x_start,
                patches_ph, patches_label, patches_adc, patch_coords
            )
    
    def _cluster_lesion_voxels(self, non_zero_indices, min_distance=16):
        """Simple clustering of lesion voxels to find centers"""
        if len(non_zero_indices) == 0:
            return []
        
        # Convert to list of lists to avoid recursion issues with torch tensors
        indices_list = [idx.tolist() for idx in non_zero_indices]
        centers = [indices_list[0]]
        
        for idx in indices_list[1:]:
            # Check if this voxel is far enough from existing centers
            is_new_center = True
            for center in centers:
                distance = ((idx[0] - center[0])**2 + 
                           (idx[1] - center[1])**2 + 
                           (idx[2] - center[2])**2)**0.5
                if distance < min_distance:
                    is_new_center = False
                    break
            
            if is_new_center:
                centers.append(idx)
        
        return centers
    
    def _pad_patch(self, patch):
        """
        Pad patch to match required patch size
        
        Args:
            patch: Tensor of shape [C, D, H, W] or similar
            
        Returns:
            Padded patch with shape [C, patch_size[0], patch_size[1], patch_size[2]]
        """
        # Ensure patch has 4 dimensions [C, D, H, W]
        if patch.dim() != 4:
            print(f"Warning: Patch has {patch.dim()} dimensions, expected 4. Shape: {patch.shape}")
            # Try to fix the dimensions
            if patch.dim() == 3:  # [D, H, W] - missing channel dimension
                patch = patch.unsqueeze(0)
            elif patch.dim() == 5:  # [B, C, D, H, W] - has batch dimension
                patch = patch.squeeze(0)
            else:
                # Can't easily fix, so return as is
                return patch
        
        # Get current dimensions
        c, d, h, w = patch.shape
        
        # Calculate padding needed for each dimension
        pad_d = max(0, self.patch_size[0] - d)
        pad_h = max(0, self.patch_size[1] - h)
        pad_w = max(0, self.patch_size[2] - w)
        
        # Apply padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # Calculate padding for each side (PyTorch padding is applied from the last dimension backward)
            # Format: (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
            paddings = (0, pad_w, 0, pad_h, 0, pad_d)
            
            try:
                padded_patch = torch.nn.functional.pad(
                    patch, 
                    paddings, 
                    mode='constant', 
                    value=0
                )
                
                # Verify the dimensions after padding
                if padded_patch.shape[1:] != tuple(self.patch_size):
                    print(f"Warning: After padding, patch has shape {padded_patch.shape}, expected [C, {self.patch_size[0]}, {self.patch_size[1]}, {self.patch_size[2]}]")
                    # If dimensions still mismatch, try a different approach
                    # Create new tensor with the exact target size
                    target_patch = torch.zeros((c, self.patch_size[0], self.patch_size[1], self.patch_size[2]), 
                                              device=patch.device, dtype=patch.dtype)
                    # Copy the original data to the new tensor
                    target_d = min(d, self.patch_size[0])
                    target_h = min(h, self.patch_size[1])
                    target_w = min(w, self.patch_size[2])
                    target_patch[:, :target_d, :target_h, :target_w] = patch[:, :target_d, :target_h, :target_w]
                    return target_patch
                
                return padded_patch
            except Exception as e:
                print(f"Error during padding: {e}")
                # Create a new tensor with the correct size as fallback
                return torch.zeros((c, self.patch_size[0], self.patch_size[1], self.patch_size[2]), 
                                  device=patch.device, dtype=patch.dtype)
        
        # If patch size already matches or is larger, return as is
        return patch
    
    def _generate_cache_key(self, pseudo_healthy, label, adc):
        """Generate a cache key based on tensor shapes and contents hash"""
        # Use shape and sum as a simple hash
        ph_shape = pseudo_healthy.shape
        label_shape = label.shape
        adc_shape = adc.shape
        
        # Including data statistics helps differentiate volumes
        # Without computing expensive full hashes
        ph_sum = pseudo_healthy.sum().item()
        label_sum = label.sum().item()
        adc_sum = adc.sum().item()
        
        # Combine into a unique key
        return f"{ph_shape}_{label_shape}_{adc_shape}_{ph_sum:.2f}_{label_sum:.2f}_{adc_sum:.2f}"

    def reconstruct_from_patches(self, patches, patch_coords, output_shape):
        """
        Reconstruct full volume from patches.
        
        Parameters:
        -----------
        patches : list of torch.Tensor
            List of patches, each with shape [C, Z, Y, X]
        patch_coords : list of tuple
            List of coordinates (z, y, x) indicating the starting position of each patch
        output_shape : tuple
            Shape of the output volume [B, C, Z, Y, X] or [C, Z, Y, X]
            
        Returns:
        --------
        torch.Tensor
            Reconstructed volume
        """
        # Create a cache key for the reconstruction
        # Add hash of the first few patch contents to distinguish different reconstructions
        if len(patches) > 0:
            first_patch_sum = patches[0].sum().item()
            cache_key = f"{str(output_shape)}_{len(patches)}_{first_patch_sum:.2f}"
        else:
            cache_key = f"{str(output_shape)}_{len(patches)}"
        
        # Check if we have this reconstruction cached
        if cache_key in self.reconstructed_cache:
            return self.reconstructed_cache[cache_key]
        
        # Validate inputs
        if len(patches) == 0:
            raise ValueError("No patches provided for reconstruction")
        
        if len(patches) != len(patch_coords):
            raise ValueError(f"Number of patches ({len(patches)}) does not match number of coordinates ({len(patch_coords)})")
        
        # Force CUDA device - use the exact string "cuda:0" which is more reliable
        cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Forcing device {cuda_device} for reconstruction")
        
        # Validate that all patch coordinates are non-negative
        for i, (z, y, x) in enumerate(patch_coords):
            if z < 0 or y < 0 or x < 0:
                raise ValueError(f"Patch {i} has negative coordinates: ({z}, {y}, {x})")
        
        # Ensure output_shape has 4 dimensions [C, Z, Y, X] or 5 dimensions [B, C, Z, Y, X]
        if len(output_shape) == 5:  # [B, C, Z, Y, X]
            # Remove batch dimension for processing
            working_shape = output_shape[1:]
            has_batch_dim = True
        else:  # [C, Z, Y, X]
            working_shape = output_shape
            has_batch_dim = False
            
        # Log information for debugging
        print(f"Reconstructing from {len(patches)} patches to output shape {output_shape}")
        print(f"First patch shape: {patches[0].shape}, device: {patches[0].device}")
        
        # Force move all patches to the CUDA device
        cuda_patches = []
        for i, patch in enumerate(patches):
            # Only show message if actually moving
            if patch.device != cuda_device:
                print(f"Moving patch {i} from {patch.device} to {cuda_device}")
            cuda_patches.append(patch.to(cuda_device))
        
        # Create tensors for reconstruction directly on the CUDA device
        reconstructed = torch.zeros(working_shape, device=cuda_device)
        count = torch.zeros(working_shape, device=cuda_device)
        
        try:
            for i, (patch, (z, y, x)) in enumerate(zip(cuda_patches, patch_coords)):
                # Double-check the device
                if patch.device != cuda_device:
                    print(f"Warning: Patch {i} still on {patch.device} after moving! Fixing...")
                    patch = patch.to(cuda_device)
                
                # Get actual patch dimensions
                patch_shape = patch.shape
                
                # Check for dimension mismatch - patch should be [C, Z, Y, X]
                if len(patch_shape) != 4:
                    print(f"Warning: Patch {i} has unexpected dimensions: {patch_shape}, expected 4D tensor [C, Z, Y, X]")
                    continue
                
                # Calculate the end coordinates based on patch dimensions
                # and ensure they don't exceed output boundaries
                z_end = min(z + patch_shape[1], working_shape[1])
                y_end = min(y + patch_shape[2], working_shape[2])
                x_end = min(x + patch_shape[3], working_shape[3])
                
                # Calculate the actual patch size to copy (may be smaller than the patch itself)
                patch_z = min(patch_shape[1], z_end - z)
                patch_y = min(patch_shape[2], y_end - y)
                patch_x = min(patch_shape[3], x_end - x)
                
                # Skip patches that don't fit the volume dimensions
                if z >= working_shape[1] or y >= working_shape[2] or x >= working_shape[3]:
                    print(f"Warning: Patch {i} coordinates outside volume: ({z}, {y}, {x}), volume shape: {working_shape}")
                    continue
                
                # Only copy the valid portion of the patch
                if patch_z > 0 and patch_y > 0 and patch_x > 0:
                    try:
                        # Extract the slices for both output and patch
                        output_slice = reconstructed[:, z:z_end, y:y_end, x:x_end]
                        patch_slice = patch[:, :patch_z, :patch_y, :patch_x]
                        
                        # Verify the tensor is on the correct device before operation
                        if output_slice.device != cuda_device:
                            print(f"ERROR: output_slice on wrong device: {output_slice.device}")
                            output_slice = output_slice.to(cuda_device)
                            
                        if patch_slice.device != cuda_device:
                            print(f"ERROR: patch_slice on wrong device: {patch_slice.device}")
                            patch_slice = patch_slice.to(cuda_device)
                        
                        # Ensure dimensions match
                        if output_slice.shape != patch_slice.shape:
                            print(f"Patch {i} dimension mismatch: output_slice {output_slice.shape}, patch_slice {patch_slice.shape}")
                            
                            # Try to match dimensions by padding or cropping patch
                            if output_slice.dim() > patch_slice.dim():
                                # Output slice has more dimensions than patch slice
                                print(f"  -> Padding patch to match output dimensions")
                                # This is a rare case, we can't easily handle it without knowing which dimension is missing
                                continue
                            elif patch_slice.dim() > output_slice.dim():
                                # Patch slice has more dimensions than output slice
                                print(f"  -> Reducing patch dimensions to match output")
                                while patch_slice.dim() > output_slice.dim():
                                    patch_slice = patch_slice.squeeze(0)
                            
                            # Check if dimensions now match after adjustment
                            if output_slice.shape != patch_slice.shape:
                                # If still not matching, try to pad or crop specific dimensions
                                new_patch_slice = torch.zeros_like(output_slice)
                                # Verify new tensor is on the correct device
                                if new_patch_slice.device != cuda_device:
                                    print(f"ERROR: new_patch_slice on wrong device: {new_patch_slice.device}")
                                    new_patch_slice = new_patch_slice.to(cuda_device)
                                    
                                # Copy the overlapping part
                                min_c = min(output_slice.shape[0], patch_slice.shape[0])
                                min_z = min(output_slice.shape[1], patch_slice.shape[1]) if output_slice.dim() > 1 and patch_slice.dim() > 1 else 0
                                min_y = min(output_slice.shape[2], patch_slice.shape[2]) if output_slice.dim() > 2 and patch_slice.dim() > 2 else 0
                                min_x = min(output_slice.shape[3], patch_slice.shape[3]) if output_slice.dim() > 3 and patch_slice.dim() > 3 else 0
                                
                                # Create slice objects based on dimensions
                                if output_slice.dim() == 4 and patch_slice.dim() == 4:
                                    new_patch_slice[:min_c, :min_z, :min_y, :min_x] = patch_slice[:min_c, :min_z, :min_y, :min_x]
                                else:
                                    print(f"  -> Cannot reconcile dimensions, skipping patch")
                                    continue
                                
                                patch_slice = new_patch_slice
                            
                        # Add patch to reconstructed volume and update count
                        reconstructed[:, z:z_end, y:y_end, x:x_end] += patch_slice
                        count[:, z:z_end, y:y_end, x:x_end] += 1
                    except Exception as e:
                        print(f"Error adding patch {i} at coords ({z}, {y}, {x}): {e}")
                        print(f"Patch shape: {patch.shape}, device: {patch.device}")
                        print(f"Output slice shape: {output_slice.shape if 'output_slice' in locals() else 'N/A'}, "
                              f"device: {output_slice.device if 'output_slice' in locals() else 'N/A'}")
                        continue
                else:
                    print(f"Warning: Invalid patch dimensions for coords ({z}, {y}, {x}): shape={patch_shape}, output_shape={working_shape}")
            
            # Average overlapping regions
            count[count == 0] = 1  # Avoid division by zero
            reconstructed = reconstructed / count
            
            # Check if reconstruction was successful
            if torch.isnan(reconstructed).any():
                print("Warning: NaN values detected in reconstruction")
                reconstructed = torch.nan_to_num(reconstructed, 0.0)
            
            # Add batch dimension back if original shape had it
            if has_batch_dim:
                reconstructed = reconstructed.unsqueeze(0)
            
            # Final device verification
            print(f"Reconstruction complete. Output tensor shape: {reconstructed.shape}, device: {reconstructed.device}")
            
            # Cache the result
            self.reconstructed_cache[cache_key] = reconstructed
            
            return reconstructed
            
        except Exception as e:
            print(f"Error in reconstruct_from_patches: {e}")
            # Print detailed device information for debugging
            for i, patch in enumerate(cuda_patches):
                print(f"Patch {i}: shape={patch.shape}, coord={patch_coords[i]}, device={patch.device}")
            
            # Check if reconstructed and count are initialized and print their devices
            if 'reconstructed' in locals():
                print(f"reconstructed device: {reconstructed.device}")
            if 'count' in locals():
                print(f"count device: {count.device}")
                
            raise

class LesionInpaintingModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(LesionInpaintingModel, self).__init__()
        
        # SwinUNETR as the generator
        self.generator = SwinUNETR(
            img_size=(64, 64, 32),
            in_channels=in_channels,  # Pseudo-healthy and lesion mask
            out_channels=out_channels,  # Inpainted ADC
            feature_size=48,
            use_checkpoint=True
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pseudo_healthy, label):
        # Concatenate inputs along channel dimension
        x = torch.cat([pseudo_healthy, label], dim=1)
        
        # Generate inpainted output
        output = self.generator(x)
        
        return output

class MaskedMAELoss(nn.Module):
    def __init__(self, weight=1.0):
        super(MaskedMAELoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred, target, mask, orig_range=None):
        """
        Compute MAE loss strictly focused on lesion regions
        
        Args:
            pred: Predicted tensor
            target: Target tensor  
            mask: Lesion mask tensor
            orig_range: Original intensity range as tuple (min, max) before normalization
            
        Returns:
            Weighted MAE loss
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Apply mask to both prediction and target
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Calculate absolute difference
        abs_diff = torch.abs(masked_pred - masked_target)
        
        # Calculate mean only over masked regions
        mask_sum = binary_mask.sum()
        if mask_sum > 0:
            mae = abs_diff.sum() / mask_sum
            
            # Apply denormalization if original range is provided
            if orig_range is not None:
                orig_min, orig_max = orig_range
                
                # Check if orig_range is a tensor and move to device if needed
                device = pred.device
                if isinstance(orig_min, torch.Tensor):
                    if orig_min.device != device:
                        orig_min = orig_min.to(device)
                else:
                    orig_min = torch.tensor(orig_min, device=device)
                    
                if isinstance(orig_max, torch.Tensor):
                    if orig_max.device != device:
                        orig_max = orig_max.to(device)
                else:
                    orig_max = torch.tensor(orig_max, device=device)
                
                # Scale the loss by the original range
                range_factor = torch.abs(orig_max - orig_min)
                mae = mae * range_factor
            
            # Apply weight
            return self.weight * mae
        else:
            return torch.tensor(0.0, device=pred.device)

class GradientSmoothingLoss(nn.Module):
    def __init__(self, weight=0.01):
        super(GradientSmoothingLoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred, mask):
        """
        Compute gradient smoothing loss for better visual quality
        The weight is kept very low to not interfere with primary losses
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Calculate gradients in 3 dimensions
        grad_z = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        grad_y = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        grad_x = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        # Apply mask to gradients (using proper dimensions)
        mask_z = binary_mask[:, :, 1:, :, :]
        mask_y = binary_mask[:, :, :, 1:, :]
        mask_x = binary_mask[:, :, :, :, 1:]
        
        # Calculate gradient smoothing loss
        loss_z = (grad_z**2 * mask_z).sum() / (mask_z.sum() + 1e-8)
        loss_y = (grad_y**2 * mask_y).sum() / (mask_y.sum() + 1e-8)
        loss_x = (grad_x**2 * mask_x).sum() / (mask_x.sum() + 1e-8)
        
        # Apply very low weight to avoid dominating the primary losses
        return self.weight * (loss_z + loss_y + loss_x)

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms - all SimpleITK loading is handled in the dataset
    train_transforms = Compose([
        # Normalization - keep this as it's not an augmentation but required preprocessing
        ScaleIntensityd(keys=["pseudo_healthy", "adc"], minv=0.0, maxv=1.0),
        # Spatial augmentations only
        RandAffined(
            keys=["pseudo_healthy", "adc", "label"],
            prob=0.15,
            rotate_range=(0.05, 0.05, 0.05),
            scale_range=(0.05, 0.05, 0.05),
            mode=("bilinear", "bilinear", "nearest"),
            padding_mode="zeros"
        ),
        # Add random flips
        RandFlipd(
            keys=["pseudo_healthy", "adc", "label"],
            spatial_axis=[0, 1, 2],
            prob=0.15
        ),
        # Remove intensity augmentation: RandScaleIntensityd
        EnsureTyped(keys=["pseudo_healthy", "adc", "label"]),
    ])
    
    val_transforms = Compose([
        ScaleIntensityd(keys=["pseudo_healthy", "adc"], minv=0.0, maxv=1.0),
        EnsureTyped(keys=["pseudo_healthy", "adc", "label"]),
    ])
    
    # Create datasets
    full_dataset = LesionInpaintingDataset(
        pseudo_healthy_dir=args.pseudo_healthy_dir,
        adc_dir=args.adc_dir,
        label_dir=args.label_dir,
        transform=train_transforms
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    val_size = max(1, int(dataset_size * 0.2))
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Enable caching for datasets to avoid repeated file I/O
    # The underlying dataset is the same for both splits
    full_dataset.enable_cache(True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # Process one volume at a time
        shuffle=True, 
        num_workers=4,
        collate_fn=list_data_collate
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=4,
        collate_fn=list_data_collate
    )
    
    # Initialize model
    model = LesionInpaintingModel(in_channels=2, out_channels=1).to(device)
    
    # Define loss functions - focus primarily on MAE
    masked_mae_loss = MaskedMAELoss(weight=1.0).to(device)  
    gradient_loss = GradientSmoothingLoss(weight=0.01).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize patch extractor with caching
    patch_extractor = PatchExtractor(patch_size=(64, 64, 32))
    
    # Training loop
    best_val_lesion_loss = float('inf')
    best_val_lesion_dice = 0.0
    best_val_lesion_mae = float('inf')
    best_val_lesion_ssim = 0.0
    
    # Track previously processed volumes to avoid duplicate work
    processed_volumes = {}
    
    try:
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            epoch_mae_loss = 0
            epoch_gradient_loss = 0
            train_samples = 0
            
            # Clear patch extractor cache at the start of each epoch to avoid memory issues
            # Dataset cache is kept across epochs for faster loading
            patch_extractor.clear_cache()
            
            for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
                pseudo_healthy = batch_data["pseudo_healthy"].to(device)
                adc = batch_data["adc"].to(device)
                label = batch_data["label"].to(device)
                
                # Get the original ranges from the dataset for MAE loss calculation
                adc_orig_range = None
                if "original_ranges" in batch_data and "adc" in batch_data["original_ranges"]:
                    adc_orig_range = batch_data["original_ranges"]["adc"]
                    # Move ranges to the appropriate device
                    if isinstance(adc_orig_range[0], torch.Tensor) and adc_orig_range[0].device != device:
                        adc_orig_range = (adc_orig_range[0].to(device), adc_orig_range[1].to(device))
                
                # Extract patches containing lesions
                patches_ph, patches_label, patches_adc, patch_coords = patch_extractor.extract_patches_with_lesions(
                    pseudo_healthy[0], label[0], adc[0]
                )
                
                # Filter out patches with invalid coordinates (negative values)
                valid_patches = []
                for i, (patch, coords) in enumerate(zip(patches_ph, patch_coords)):
                    if all(c >= 0 for c in coords):
                        valid_patches.append(i)
                
                # Skip if no valid patches
                if not valid_patches:
                    print(f"Warning: No valid patches found for training sample {batch_idx}")
                    continue
                    
                # Use only valid patches
                filtered_patches_ph = [patches_ph[i] for i in valid_patches]
                filtered_patches_label = [patches_label[i] for i in valid_patches]
                filtered_patches_adc = [patches_adc[i] for i in valid_patches]
                
                batch_loss = 0
                for ph_patch, label_patch, adc_patch in zip(filtered_patches_ph, filtered_patches_label, filtered_patches_adc):
                    # Add batch dimension
                    ph_patch = ph_patch.unsqueeze(0)
                    label_patch = label_patch.unsqueeze(0)
                    adc_patch = adc_patch.unsqueeze(0)
                    
                    # Generate inpainted output
                    output = model(ph_patch, label_patch)
                    
                    # Calculate losses
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    gradient_value = gradient_loss(output, label_patch)
                    
                    # Calculate total loss - focus entirely on MAE with minimal smoothing
                    loss = mae_value + gradient_value
                    
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss += loss.item()
                    train_samples += 1
                    
                    # Track individual loss components
                    epoch_mae_loss += mae_value.item()
                    epoch_gradient_loss += gradient_value.item()
                
                # Average loss over patches
                if len(filtered_patches_ph) > 0:
                    avg_batch_loss = batch_loss / len(filtered_patches_ph)
                    epoch_loss += avg_batch_loss
                    
            avg_epoch_loss = epoch_loss / max(1, len(train_loader))
            avg_epoch_mae_loss = epoch_mae_loss / max(1, train_samples)
            avg_epoch_gradient_loss = epoch_gradient_loss / max(1, train_samples)
            
            print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")
            print(f"  - MAE Loss: {avg_epoch_mae_loss:.4f}")
            print(f"  - Gradient Loss: {avg_epoch_gradient_loss:.4f}")
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_lesion_mae = 0.0
            val_lesion_ssim = 0.0
            val_lesion_dice = 0.0
            val_examples = 0
            
            # Clear patch extractor cache again before validation
            patch_extractor.clear_cache()
            
            with torch.no_grad():
                for val_batch_idx, val_batch_data in enumerate(val_loader):
                    pseudo_healthy = val_batch_data["pseudo_healthy"].to(device)
                    adc = val_batch_data["adc"].to(device)
                    label = val_batch_data["label"].to(device)
                    
                    # Get the original ranges from the dataset
                    if "original_ranges" in val_batch_data:
                        ph_orig_range = val_batch_data["original_ranges"]["pseudo_healthy"]
                        adc_orig_range = val_batch_data["original_ranges"]["adc"]
                        
                        # Move ranges to the appropriate device
                        cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        if isinstance(adc_orig_range[0], torch.Tensor) and adc_orig_range[0].device != cuda_device:
                            adc_orig_range = (adc_orig_range[0].to(cuda_device), adc_orig_range[1].to(cuda_device))
                        if isinstance(ph_orig_range[0], torch.Tensor) and ph_orig_range[0].device != cuda_device:  
                            ph_orig_range = (ph_orig_range[0].to(cuda_device), ph_orig_range[1].to(cuda_device))
                    else:
                        # Fallback if original ranges are not available
                        print("Warning: Original ranges not found in data. Using normalized range.")
                        cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        ph_orig_range = (torch.tensor(0.0, device=cuda_device), torch.tensor(1.0, device=cuda_device))
                        adc_orig_range = (torch.tensor(0.0, device=cuda_device), torch.tensor(1.0, device=cuda_device))
                    
                    try:
                        # Extract patches containing lesions, ensuring valid coordinates
                        patches_ph, patches_label, patches_adc, patch_coords = patch_extractor.extract_patches_with_lesions(
                            pseudo_healthy[0], label[0], adc[0]
                        )
                        
                        # ... [rest of validation code continues] ...
            
                    except Exception as e:
                        print(f"Error processing validation sample {val_batch_idx}: {e}")
                        continue
            
            # Save checkpoint with all metrics
            if (epoch + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'mae_loss': avg_epoch_mae_loss,
                    'gradient_loss': avg_epoch_gradient_loss,
                    'val_loss': avg_val_loss,
                    'val_lesion_mae': avg_val_lesion_mae,
                    'val_lesion_ssim': avg_val_lesion_ssim,
                    'val_lesion_dice': avg_val_lesion_dice
                }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    
    finally:
        # Clean up cache when done to free memory
        print("Cleaning up caches...")
        full_dataset.clear_cache()
        patch_extractor.clear_cache()
        print("Cache cleanup complete.")

def visualize_results(pseudo_healthy, label, output, target, output_path, metrics=None):
    """
    Simplified visualization function that generates a PDF with the key images
    
    Args:
        pseudo_healthy: Tensor of pseudo-healthy input
        label: Tensor of lesion labels
        output: Tensor of model output
        target: Tensor of target data
        output_path: Path to save the PDF
        metrics: Dictionary of metrics to include in the PDF
    """
    try:
        # Save a simplified PDF with just the essential visualizations
        with PdfPages(output_path) as pdf:
            # Create a metrics summary page if metrics are provided
            if metrics is not None and isinstance(metrics, dict):
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.95, "Lesion Metrics Summary", ha='center', fontsize=16, weight='bold')
                
                # Draw values in a table-like format
                y_pos = 0.85
                for metric_name, metric_value in metrics.items():
                    formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else str(metric_value)
                    plt.text(0.1, y_pos, f"{metric_name}:", fontsize=12)
                    plt.text(0.7, y_pos, formatted_value, fontsize=12)
                    y_pos -= 0.05
                
                plt.axis('off')
                pdf.savefig()
                plt.close()
            
            # Convert data to numpy with minimal processing
            try:
                ph_np = pseudo_healthy.detach().cpu().numpy() if isinstance(pseudo_healthy, torch.Tensor) else pseudo_healthy
                lbl_np = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label
                out_np = output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else output
                tgt_np = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else target
                
                # Get original range if available for MAE calculation
                orig_range = None
                if metrics is not None and "Original ADC Range" in metrics:
                    range_text = metrics["Original ADC Range"]
                    if isinstance(range_text, str) and "to" in range_text:
                        try:
                            parts = range_text.split("to")
                            min_val = float(parts[0].strip())
                            max_val = float(parts[1].strip())
                            orig_range = (min_val, max_val)
                        except:
                            pass
                
                # Simplify dimensions - just keep reducing until we get to 3D
                while len(ph_np.shape) > 3:
                    ph_np = ph_np[0]
                while len(lbl_np.shape) > 3:
                    lbl_np = lbl_np[0]
                while len(out_np.shape) > 3:
                    out_np = out_np[0]
                while len(tgt_np.shape) > 3:
                    tgt_np = tgt_np[0]
                
                # Get minimum depth to avoid index errors
                depths = [ph_np.shape[0], lbl_np.shape[0], out_np.shape[0], tgt_np.shape[0]]
                depth = min(depths)
                
                # Find slices with lesions for visualization
                lesion_slices = []
                for z in range(depth):
                    if np.any(lbl_np[z] > 0):
                        lesion_slices.append(z)
                
                # Prioritize lesion slices
                if lesion_slices:
                    # Just use lesion slices (max 10)
                    slice_indices = lesion_slices[:min(10, len(lesion_slices))]
                else:
                    # If no lesion slices found, use evenly spaced slices
                    num_slices = min(10, depth)
                    slice_indices = list(range(0, depth, max(1, depth // num_slices)))[:num_slices]
                
                # Add pages showing slices for each volume
                for z_idx, z in enumerate(slice_indices):
                    try:
                        # Create a 2x2 grid for the 4 key images
                        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                        axes = axes.flatten()
                        
                        # Get 2D slices
                        ph_slice = ph_np[z]
                        lbl_slice = lbl_np[z]
                        out_slice = out_np[z]
                        tgt_slice = tgt_np[z]
                        
                        # Ensure slices are 2D
                        if len(ph_slice.shape) > 2:
                            ph_slice = ph_slice[0] if ph_slice.shape[0] == 1 else np.mean(ph_slice, axis=0)
                        if len(lbl_slice.shape) > 2:
                            lbl_slice = lbl_slice[0] if lbl_slice.shape[0] == 1 else np.mean(lbl_slice, axis=0)
                        if len(out_slice.shape) > 2:
                            out_slice = out_slice[0] if out_slice.shape[0] == 1 else np.mean(out_slice, axis=0)
                        if len(tgt_slice.shape) > 2:
                            tgt_slice = tgt_slice[0] if tgt_slice.shape[0] == 1 else np.mean(tgt_slice, axis=0)
                        
                        # Check if this is a lesion slice
                        has_lesion = np.any(lbl_slice > 0)
                        
                        # Calculate slice-specific MAE for lesion areas
                        slice_mae = None
                        if has_lesion:
                            # Create a clean copy of the pseudo-healthy slice
                            inpainted_slice = np.copy(ph_slice)
                            
                            # Only replace voxels in the lesion area with the output from generator
                            lesion_mask = lbl_slice > 0
                            inpainted_slice[lesion_mask] = out_slice[lesion_mask]
                            
                            # Calculate MAE only for the lesion area
                            abs_diff = np.abs(inpainted_slice[lesion_mask] - tgt_slice[lesion_mask])
                            slice_mae = np.mean(abs_diff)
                            
                            # Denormalize if original range is available
                            if orig_range is not None:
                                orig_min, orig_max = orig_range
                                slice_mae_orig = slice_mae * (orig_max - orig_min)
                                slice_mae_display = f"Slice MAE: {slice_mae:.4f} (norm) / {slice_mae_orig:.4f} (orig)"
                            else:
                                slice_mae_display = f"Slice MAE: {slice_mae:.4f}"
                        
                        slice_title = f"Slice {z}" + (" (contains lesion)" if has_lesion else "")
                        if has_lesion and slice_mae is not None:
                            slice_title += f" - {slice_mae_display}"
                        
                        # 1. Pseudo-healthy
                        axes[0].imshow(ph_slice, cmap='gray')
                        axes[0].set_title('Pseudo-healthy')
                        axes[0].axis('off')
                        
                        # 2. Label Overlay
                        axes[1].imshow(ph_slice, cmap='gray')  # Background
                        if has_lesion:  # Only add overlay if there's a lesion
                            mask = lbl_slice > 0
                            # Create red mask
                            red_mask = np.zeros((*lbl_slice.shape, 4))  # RGBA
                            red_mask[mask, 0] = 1.0  # Red channel
                            red_mask[mask, 3] = 0.5  # Alpha for mask
                            axes[1].imshow(red_mask)
                        axes[1].set_title('Label Overlay')
                        axes[1].axis('off')
                        
                        # 3. Inpainted Brain (output only replaces lesion areas)
                        # First, create a clean copy of the pseudo-healthy slice
                        inpainted_slice = np.copy(ph_slice)
                        
                        # Only replace voxels in the lesion area with the output from generator
                        if has_lesion:
                            lesion_mask = lbl_slice > 0
                            inpainted_slice[lesion_mask] = out_slice[lesion_mask]
                        
                        # Show inpainted result
                        axes[2].imshow(inpainted_slice, cmap='gray')
                        title = 'Output (Inpainted Brain)'
                        if has_lesion and slice_mae is not None:
                            # Add MAE directly in the plot title
                            if orig_range is not None:
                                title += f"\nMAE: {slice_mae:.4f} (norm) / {slice_mae * (orig_max - orig_min):.4f} (orig)"
                            else:
                                title += f"\nMAE: {slice_mae:.4f}"
                        axes[2].set_title(title)
                        axes[2].axis('off')
                        
                        # 4. Target
                        axes[3].imshow(tgt_slice, cmap='gray')
                        axes[3].set_title('Target')
                        axes[3].axis('off')
                        
                        # Set overall title
                        plt.suptitle(slice_title, fontsize=16)
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                    except Exception as slice_error:
                        print(f"Error visualizing slice {z}: {slice_error}")
                        continue
            
            except Exception as data_error:
                print(f"Error processing data: {data_error}")
                # Add error page
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f"Error processing data:\n{str(data_error)}", 
                         ha='center', va='center', wrap=True)
                plt.axis('off')
                pdf.savefig()
                plt.close()
        
        # Verify the PDF is valid
        import os
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"PDF visualization created successfully: {output_path} ({os.path.getsize(output_path)} bytes)")
        else:
            print(f"Warning: PDF file may not have been created properly")
            raise RuntimeError("PDF file creation failed")
            
    except Exception as e:
        print(f"Failed to create PDF visualization: {e}")
        
        # Last resort - create a text file instead
        try:
            with open(output_path.replace('.pdf', '.txt'), 'w') as f:
                f.write(f"Visualization Error: {str(e)}\n")
                f.write(f"Pseudo-healthy shape: {pseudo_healthy.shape}\n")
                f.write(f"Label shape: {label.shape}\n")
                f.write(f"Output shape: {output.shape}\n")
                f.write(f"Target shape: {target.shape}\n")
                if metrics:
                    f.write("\nMetrics:\n")
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")
            print(f"Created text file with information instead")
        except Exception as txt_error:
            print(f"Failed to create even a text file: {txt_error}")

def inference(args):
    """
    Run inference on new data
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pseudo-healthy and lesion files directly with SimpleITK
    pseudo_healthy_img = sitk.ReadImage(args.input_pseudo_healthy)
    label_img = sitk.ReadImage(args.input_label)
    
    # Convert to numpy arrays
    pseudo_healthy_np = sitk.GetArrayFromImage(pseudo_healthy_img).astype(np.float32)
    label_np = sitk.GetArrayFromImage(label_img).astype(np.float32)
    
    # Add channel dimension and convert to PyTorch tensors
    pseudo_healthy = torch.from_numpy(np.expand_dims(pseudo_healthy_np, axis=0))
    label = torch.from_numpy(np.expand_dims(label_np, axis=0))
    
    # Normalize intensity
    pseudo_healthy = (pseudo_healthy - pseudo_healthy.min()) / (pseudo_healthy.max() - pseudo_healthy.min())
    
    # Move to device
    pseudo_healthy = pseudo_healthy.to(device)
    label = label.to(device)
    
    # Load model
    model = LesionInpaintingModel(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Initialize patch extractor
    patch_extractor = PatchExtractor(patch_size=(64, 64, 32))
    
    # Perform inference
    with torch.no_grad():
        # Use explicit cuda:0 device
        cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Inference using device: {cuda_device}")
        
        # Ensure all inputs are on CUDA
        pseudo_healthy = pseudo_healthy.to(cuda_device)
        label = label.to(cuda_device)
        
        # Extract patches with lesions
        patches_ph, patches_label, _, patch_coords = patch_extractor.extract_patches_with_lesions(
            pseudo_healthy.unsqueeze(0)[0], label.unsqueeze(0)[0], pseudo_healthy.unsqueeze(0)[0]  # Use pseudo_healthy as placeholder for adc
        )
        
        output_patches = []
        for ph_patch, label_patch in zip(patches_ph, patches_label):
            # Move patches to CUDA
            ph_patch = ph_patch.to(cuda_device).unsqueeze(0)
            label_patch = label_patch.to(cuda_device).unsqueeze(0)
            
            # Generate inpainted output
            output_patch = model(ph_patch, label_patch)
            output_patches.append(output_patch[0])
        
        # Reconstruct full volume
        if output_patches:
            # Ensure all patches are on the same device (GPU)
            for i in range(len(output_patches)):
                if output_patches[i].device != cuda_device:
                    output_patches[i] = output_patches[i].to(cuda_device)
            
            # Print device information
            print(f"Inference: First patch device: {output_patches[0].device}, target device: {cuda_device}")
                    
            reconstructed = patch_extractor.reconstruct_from_patches(
                output_patches, patch_coords, pseudo_healthy.shape
            )
            
            # Apply the lesion mask to only modify lesion regions
            dilated_mask = torch.nn.functional.max_pool3d(
                label.unsqueeze(0), kernel_size=3, stride=1, padding=1
            )
            
            # Ensure the reconstructed tensor is on the same device as pseudo_healthy
            if reconstructed.device != pseudo_healthy.device:
                reconstructed = reconstructed.to(pseudo_healthy.device)
            
            # Ensure dilated_mask is on the same device as pseudo_healthy
            if dilated_mask.device != pseudo_healthy.device:
                dilated_mask = dilated_mask.to(pseudo_healthy.device)
                
            # Create final inpainted image - copy pseudo-healthy and replace only in lesion regions
            final_output = pseudo_healthy.unsqueeze(0).clone()
            final_output = final_output * (1 - dilated_mask) + reconstructed * dilated_mask
            
            # Convert to numpy and rescale if needed
            output_np = final_output[0, 0].cpu().numpy()
            
            # Create SimpleITK image with same properties as input
            output_image = sitk.GetImageFromArray(output_np)
            output_image.CopyInformation(pseudo_healthy_img)
            
            # Save output
            sitk.WriteImage(output_image, args.output_file)
            print(f"Saved inpainted result to {args.output_file}")
            
            # Visualize result if requested
            if args.visualize:
                # For inference, we don't have metrics, so just create a simple info dict
                inference_info = {
                    "Model": os.path.basename(args.model_path),
                    "Mode": "Inference",
                    "Time": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                }
                
                visualize_results(
                    pseudo_healthy.unsqueeze(0), 
                    label.unsqueeze(0), 
                    final_output, 
                    pseudo_healthy.unsqueeze(0),  # Use pseudo-healthy as placeholder for target
                    args.output_file.replace(".mha", ".pdf"),
                    metrics=inference_info
                )

def main():
    parser = argparse.ArgumentParser(description="3D Lesion Inpainting Model")
    
    # Common arguments
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    # Mode selection
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train mode")
    train_parser.add_argument("--pseudo_healthy_dir", type=str, required=True, help="Directory with pseudo-healthy ADC maps")
    train_parser.add_argument("--adc_dir", type=str, required=True, help="Directory with target ADC maps")
    train_parser.add_argument("--label_dir", type=str, required=True, help="Directory with lesion labels")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    train_parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint frequency")
    train_parser.add_argument("--vis_freq", type=int, default=1, help="Visualization frequency")
    
    # Inference arguments
    infer_parser = subparsers.add_parser("infer", help="Inference mode")
    infer_parser.add_argument("--input_pseudo_healthy", type=str, required=True, help="Input pseudo-healthy ADC file")
    infer_parser.add_argument("--input_label", type=str, required=True, help="Input lesion label file")
    infer_parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    infer_parser.add_argument("--output_file", type=str, required=True, help="Output inpainted file path")
    infer_parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run in selected mode
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        inference(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
