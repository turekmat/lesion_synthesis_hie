import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import monai
from monai.networks.nets import SwinUNETR, AttentionUnet
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
        
        print(f"{'Training' if train else 'Validation'} dataset contains {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_dict = self.data[idx]
        
        # Load data using SimpleITK
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
        
        return data_dict

class PatchExtractor:
    def __init__(self, patch_size=(16, 16, 16), overlap=0.5, augment_patches=True, num_augmented_patches=3):
        self.patch_size = patch_size
        self.overlap = overlap
        self.augment_patches = augment_patches
        self.num_augmented_patches = num_augmented_patches
    
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
            
            return [ph_patch], [label_patch], [adc_patch], [(z_start, y_start, x_start)]
        
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
            # If still no patches with lesions after max attempts, return random patch
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
            
            return [ph_patch], [label_patch], [adc_patch], [(z_start, y_start, x_start)]
        
        return patches_ph, patches_label, patches_adc, patch_coords
    
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
        
        # For smaller patches, we need to be more careful with offsets to not lose the lesion
        # Smaller random offsets for 16x16x16 patches to avoid losing the lesion
        for i in range(self.num_augmented_patches + 2):  # Add more augmented patches for small lesions
            # Random offset in each dimension, but smaller to ensure lesion is captured
            z_offset = random.randint(-self.patch_size[0]//6, self.patch_size[0]//6)
            y_offset = random.randint(-self.patch_size[1]//6, self.patch_size[1]//6)
            x_offset = random.randint(-self.patch_size[2]//6, self.patch_size[2]//6)
            offsets.append((z_offset, y_offset, x_offset))
        
        # Add very small offsets to see lesion from multiple angles
        offsets.extend([
            (self.patch_size[0]//10, self.patch_size[1]//10, self.patch_size[2]//10),
            (-self.patch_size[0]//10, -self.patch_size[1]//10, -self.patch_size[2]//10),
            (self.patch_size[0]//10, -self.patch_size[1]//10, self.patch_size[2]//10),
            (-self.patch_size[0]//10, self.patch_size[1]//10, -self.patch_size[2]//10),
            (0, self.patch_size[1]//10, 0),  # Subtle y-axis shifts
            (0, -self.patch_size[1]//10, 0),
            (self.patch_size[0]//10, 0, 0),  # Subtle z-axis shifts
            (-self.patch_size[0]//10, 0, 0),
            (0, 0, self.patch_size[2]//10),  # Subtle x-axis shifts
            (0, 0, -self.patch_size[2]//10)
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
    
    def _cluster_lesion_voxels(self, non_zero_indices, min_distance=8):
        """Simple clustering of lesion voxels to find centers"""
        if len(non_zero_indices) == 0:
            return []
        
        # Use smaller min_distance for smaller patches to identify more clusters
        # 8 is half of our patch dimension (16x16x16)
        
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
        
        # For very small lesions with few centers, add the arithmetic mean of all voxels
        # as an additional center to ensure coverage
        if len(centers) <= 2 and len(indices_list) > 3:
            z_sum, y_sum, x_sum = 0, 0, 0
            for idx in indices_list:
                z_sum += idx[0]
                y_sum += idx[1]
                x_sum += idx[2]
            mean_center = [
                z_sum // len(indices_list),
                y_sum // len(indices_list),
                x_sum // len(indices_list)
            ]
            
            # Only add if not too close to existing centers
            is_unique = True
            for center in centers:
                distance = ((mean_center[0] - center[0])**2 + 
                           (mean_center[1] - center[1])**2 + 
                           (mean_center[2] - center[2])**2)**0.5
                if distance < min_distance // 2:  # Use smaller threshold
                    is_unique = False
                    break
            
            if is_unique:
                centers.append(mean_center)
                
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
        # Validate inputs
        if len(patches) == 0:
            raise ValueError("No patches provided for reconstruction")
        
        if len(patches) != len(patch_coords):
            raise ValueError(f"Number of patches ({len(patches)}) does not match number of coordinates ({len(patch_coords)})")
        
        # Force CUDA device - use the exact string "cuda:0" which is more reliable
        cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
        
        # AttentionUnet jako generátor, optimalizovaný pro malé patche 16x16x16
        # Upraveno pro předejití problému s batch normalizací při malých prostorových rozměrech
        self.generator = AttentionUnet(
            spatial_dims=3,  # 3D model
            in_channels=in_channels,  # Pseudo-healthy and lesion mask
            out_channels=out_channels,  # Inpainted ADC
            channels=(16, 32, 48, 64),  # Méně kanálů a méně vrstev
            strides=(2, 2, 2),  # Jen tři úrovně downsamplingu místo čtyř
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.2  # Small dropout to prevent overfitting
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
        raw_output = self.generator(x)
        
        # DŮLEŽITÉ: Použijeme masku k vytvoření finálního výstupu
        # Pouze oblast léze je nahrazena generátorem, zbytek zůstává stejný
        final_output = pseudo_healthy * (1.0 - label) + raw_output * label
        
        return final_output

class LesionDiscriminator(nn.Module):
    """
    Diskriminátor pro CGAN, který se zaměřuje pouze na oblasti lézí.
    Hodnotí pouze reálnost vygenerovaných lézí, ignoruje zbytek mozku.
    """
    def __init__(self, in_channels=3):
        super(LesionDiscriminator, self).__init__()
        
        # Vstup: [pseudo_healthy, label, adc], kde adc může být reálné nebo generované
        
        # Konvoluční bloky optimalizované pro malé patche 16x16x16
        self.conv_blocks = nn.Sequential(
            # První blok - zmenšit velikost
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),  # 16x16x16 -> 8x8x8
            
            # Druhý blok
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),  # 8x8x8 -> 4x4x4
            
            # Třetí blok
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Finální predikce - full patch output (4x4x4 -> 4x4x4)
            nn.Conv3d(128, 1, kernel_size=3, stride=1, padding=1)
        )
        
        # Inicializace vah pro stabilnější trénink
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm3d):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pseudo_healthy, label, adc):
        """
        Forward pass diskriminátoru.
        
        Args:
            pseudo_healthy: vstupní pseudo-zdravý obraz
            label: maska léze
            adc: skutečná nebo generovaná ADC mapa
        
        Returns:
            Diskriminační skóre zaměřené jen na oblast léze
        """
        # Spojení vstupů podél kanálové dimenze
        x = torch.cat([pseudo_healthy, label, adc], dim=1)
        
        # Průchod konvolučními bloky
        patch_predictions = self.conv_blocks(x)
        
        # Uložíme si velikost původní masky
        original_shape = label.shape
        
        # Interpolace predikcí zpět na původní velikost masky
        if patch_predictions.shape[2:] != label.shape[2:]:
            upsampled_predictions = torch.nn.functional.interpolate(
                patch_predictions,
                size=label.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        else:
            upsampled_predictions = patch_predictions
        
        # Aplikovat masku na predikce - hodnotíme pouze oblasti léze
        masked_predictions = upsampled_predictions * label
        
        # Průměrování pouze přes maskované oblasti
        mask_sum = torch.sum(label)
        if mask_sum > 0:
            masked_score = torch.sum(masked_predictions) / mask_sum
        else:
            masked_score = torch.tensor(0.0, device=x.device)
        
        return masked_score

class MaskedMAELoss(nn.Module):
    def __init__(self, weight=25.0):
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
        
        # Zkontrolujeme, zda maska obsahuje nějaké léze
        mask_sum = binary_mask.sum()
        if mask_sum <= 0:
            # Pokud nejsou léze, vrátíme nulovou ztrátu
            return torch.tensor(0.0, device=pred.device)
        
        # Apply mask to both prediction and target
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Calculate absolute difference
        abs_diff = torch.abs(masked_pred - masked_target)
        
        # Calculate mean only over masked regions
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
            
            # Scale the loss by the original range for more intuitivní výsledky
            range_factor = torch.abs(orig_max - orig_min)
            mae = mae * range_factor
            
            # Pro lepší debugging
            if torch.isnan(mae).any() or torch.isinf(mae).any():
                print(f"Warning: NaN/Inf in MAE loss! Range: {orig_min.item()} to {orig_max.item()}")
                return torch.tensor(1.0, device=pred.device) * self.weight
        
        # Apply weight - tato váha nyní bude výrazně vyšší (25.0 místo 1.0)
        return self.weight * mae

class SSIMLoss(nn.Module):
    def __init__(self, weight=5.0):
        super(SSIMLoss, self).__init__()
        self.weight = weight
    
    def forward(self, pred, target, mask):
        """
        Computes 1-SSIM as a loss (since higher SSIM is better)
        Strictly focuses on lesion regions
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            mask: Lesion mask tensor
            
        Returns:
            SSIM loss (1-SSIM) weighted
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Apply mask
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Only compute if mask has non-zero elements
        mask_sum = binary_mask.sum()
        if mask_sum <= 0:
            return torch.tensor(0.0, device=pred.device)
            
        # Compute means
        mu_pred = masked_pred.sum() / mask_sum
        mu_target = masked_target.sum() / mask_sum
        
        # Compute variances
        var_pred = ((masked_pred - mu_pred * binary_mask) ** 2).sum() / mask_sum
        var_target = ((masked_target - mu_target * binary_mask) ** 2).sum() / mask_sum
        
        # Compute covariance
        cov = ((masked_pred - mu_pred * binary_mask) * (masked_target - mu_target * binary_mask)).sum() / mask_sum
        
        # Constants for stability
        c1 = (0.01 * 1) ** 2
        c2 = (0.03 * 1) ** 2
        
        # Compute SSIM
        ssim = ((2 * mu_pred * mu_target + c1) * (2 * cov + c2)) / \
               ((mu_pred**2 + mu_target**2 + c1) * (var_pred + var_target + c2))
               
        # Ensure SSIM is in [0,1] range (clip if numerical issues)
        ssim = torch.clamp(ssim, 0.0, 1.0)
        
        # Return 1-SSIM as the loss (since higher SSIM is better)
        # Apply higher weight to make this term significant
        return self.weight * (1.0 - ssim)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=25.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, pred, target, mask, orig_range=None):
        """
        Compute Focal Loss for regression, focusing more on difficult examples (large errors)
        
        Args:
            pred: Predicted tensor
            target: Target tensor  
            mask: Lesion mask tensor (to focus only on lesions)
            orig_range: Original intensity range as tuple (min, max) before normalization
            
        Returns:
            Weighted Focal Loss
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Check if mask contains any lesions
        mask_sum = binary_mask.sum()
        if mask_sum <= 0:
            # If no lesions, return zero loss
            return torch.tensor(0.0, device=pred.device)
        
        # Apply mask to both prediction and target
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Calculate absolute difference
        abs_diff = torch.abs(masked_pred - masked_target)
        
        # Normalize the diff to [0, 1] for the focal weight calculation
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
            
            range_factor = torch.abs(orig_max - orig_min)
            norm_diff = abs_diff / range_factor
        else:
            # Assume data is already normalized
            norm_diff = abs_diff
        
        # Apply focal weight: (1 - e^(-norm_diff))^gamma
        # This gives higher weight to larger errors
        focal_weight = (1 - torch.exp(-norm_diff * 5.0)).pow(self.gamma)  # Multiply by 5 to make the curve steeper
        
        # Apply alpha weighting for positive examples
        weighted_loss = self.alpha * focal_weight * abs_diff
        
        # Take mean over masked region
        focal_loss = weighted_loss.sum() / mask_sum
        
        # Apply denormalization if original range is provided
        if orig_range is not None:
            focal_loss = focal_loss * range_factor
            
            # Handle NaN/Inf for debugging
            if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
                print(f"Warning: NaN/Inf in Focal loss! Range: {orig_min.item()} to {orig_max.item()}")
                return torch.tensor(1.0, device=pred.device) * self.weight
        
        # Apply weight
        return self.weight * focal_loss

class PerceptualLoss(nn.Module):
    def __init__(self, weight=10.0):
        super(PerceptualLoss, self).__init__()
        self.weight = weight
        
        # Využijeme konvoluční vrstvy jako extraktory příznaků
        # Pro 3D data s malými patch rozměry (16x16x16) použijeme 3D konvoluce s menšími filtry
        self.feature_extractor = nn.Sequential(
            # První vrstva - zachycení základních textur (16x16x16 -> 14x14x14)
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Druhá vrstva - zachycení středních detailů (14x14x14 -> 12x12x12)
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Třetí vrstva - zachycení vyšších detailů (12x12x12 -> 10x10x10)
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Zmrazíme parametry feature extraktoru - nebudeme je aktualizovat během tréninku
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target, mask, orig_range=None):
        """
        Výpočet perceptuálního lossu - porovnává příznaky extrahované z predikce a cíle
        Zaměřujeme se POUZE na oblast léze (použití masky)
        
        Args:
            pred: Predikovaný 3D obraz
            target: Cílový 3D obraz
            mask: Maska léze
            orig_range: Původní rozsah intenzit (pro škálování)
            
        Returns:
            Vážená perceptuální ztráta
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Check if mask contains any lesions
        mask_sum = binary_mask.sum()
        if mask_sum <= 0:
            # If no lesions, return zero loss
            return torch.tensor(0.0, device=pred.device)
        
        # Aplikujeme masku - počítáme loss pouze v oblasti léze
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Extrahujeme příznaky pouze z oblasti léze
        try:
            # Extrakce příznaků
            pred_features = self.feature_extractor(masked_pred)
            target_features = self.feature_extractor(masked_target)
            
            # Vytvoříme upravenou masku pro feature mapy
            # Původní maska je 16x16x16, feature mapy jsou 10x10x10
            # Proto musíme přizpůsobit masku
            if binary_mask.shape[-1] == 16:  # Pokud je maska 16x16x16
                # Feature mapa je 10x10x10 po třech konvolucích 3x3
                feature_mask = torch.nn.functional.interpolate(
                    binary_mask, 
                    size=(10, 10, 10), 
                    mode='trilinear', 
                    align_corners=False
                )
                feature_mask = (feature_mask > 0.5).float()
            else:
                # Pro jiné velikosti masek - potřebujeme výpočet nové velikosti
                # Po třech vrstvách s kernelem 3x3 bez paddingu se každá dimenze zmenší o 6
                d, h, w = binary_mask.shape[2:]
                new_d, new_h, new_w = d-6, h-6, w-6
                
                # Interpolace na novou velikost
                if new_d > 0 and new_h > 0 and new_w > 0:
                    feature_mask = torch.nn.functional.interpolate(
                        binary_mask, 
                        size=(new_d, new_h, new_w), 
                        mode='trilinear', 
                        align_corners=False
                    )
                    feature_mask = (feature_mask > 0.5).float()
                else:
                    # Pokud jsou rozměry příliš malé, tuto loss nemůžeme použít
                    return torch.tensor(0.0, device=pred.device)
            
            # L1 loss na příznacích vážené maskou
            feature_diff = torch.abs(pred_features - target_features)
            masked_feature_diff = feature_diff * feature_mask
            
            # Průměrujeme pouze přes platné voxely léze v feature mapách
            feature_mask_sum = feature_mask.sum()
            if feature_mask_sum > 0:
                perceptual_loss = masked_feature_diff.sum() / feature_mask_sum
            else:
                perceptual_loss = torch.tensor(0.0, device=pred.device)
            
            # Škálování podle původního rozsahu
            if orig_range is not None:
                orig_min, orig_max = orig_range
                if isinstance(orig_min, torch.Tensor):
                    if orig_min.device != pred.device:
                        orig_min = orig_min.to(pred.device)
                else:
                    orig_min = torch.tensor(orig_min, device=pred.device)
                    
                if isinstance(orig_max, torch.Tensor):
                    if orig_max.device != pred.device:
                        orig_max = orig_max.to(pred.device)
                else:
                    orig_max = torch.tensor(orig_max, device=pred.device)
                
                range_factor = torch.abs(orig_max - orig_min)
                perceptual_loss = perceptual_loss * range_factor
                
                # Zpracování NaN/Inf pro debugging
                if torch.isnan(perceptual_loss).any() or torch.isinf(perceptual_loss).any():
                    print(f"Warning: NaN/Inf in Perceptual loss! Range: {orig_min.item()} to {orig_max.item()}")
                    return torch.tensor(1.0, device=pred.device) * self.weight
            
            # Vracíme váženou ztrátu
            return self.weight * perceptual_loss
            
        except Exception as e:
            print(f"Error in perceptual loss calculation: {e}")
            # V případě chyby vrátíme nulovou ztrátu
            return torch.tensor(0.0, device=pred.device)

class GradientSmoothingLoss(nn.Module):
    def __init__(self, weight=0.02):
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

class DynamicWeightedMAELoss(nn.Module):
    def __init__(self, base_weight=30.0, scaling_factor=8.0):
        super(DynamicWeightedMAELoss, self).__init__()
        self.base_weight = base_weight
        self.scaling_factor = scaling_factor
        
    def forward(self, pred, target, mask, orig_range=None):
        """
        Compute MAE loss STRICTLY in lesion areas only with dynamic weighting based on lesion size.
        Smaller lesions receive higher weights to ensure they're properly inpainted.
        
        Args:
            pred: Predicted tensor
            target: Target tensor  
            mask: Lesion mask tensor
            orig_range: Original intensity range as tuple (min, max) before normalization
            
        Returns:
            Dynamically weighted MAE loss (calculated ONLY in lesion voxels)
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Zjistit relativní velikost léze v patchi
        lesion_voxels = binary_mask.sum()
        total_voxels = torch.numel(binary_mask)
        lesion_ratio = lesion_voxels / total_voxels
        
        # Zkontrolujeme, zda maska obsahuje nějaké léze
        if lesion_voxels <= 0:
            # Pokud nejsou léze, vrátíme nulovou ztrátu
            return torch.tensor(0.0, device=pred.device)
            
        # Dynamicky zvýšit váhu při malém podílu léze (inverzní vztah)
        # Čím menší léze, tím větší váha (ale zachovává minimální váhu base_weight)
        dynamic_weight = self.base_weight * (1.0 + self.scaling_factor * (1.0 - lesion_ratio))
        
        # DŮLEŽITÉ: Aplikujeme masku, aby se počítala ztráta POUZE v oblasti léze
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Calculate absolute difference POUZE v oblasti léze
        abs_diff = torch.abs(masked_pred - masked_target)
        
        # Calculate mean only over masked regions - POUZE v oblasti léze
        mae = abs_diff.sum() / lesion_voxels
        
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
            
            # Scale the loss by the original range for more intuitive results
            range_factor = torch.abs(orig_max - orig_min)
            mae = mae * range_factor
            
            # Pro lepší debugging
            if torch.isnan(mae).any() or torch.isinf(mae).any():
                print(f"Warning: NaN/Inf in Dynamic MAE loss! Range: {orig_min.item()} to {orig_max.item()}")
                return torch.tensor(1.0, device=pred.device) * self.base_weight
        
        # Pro debugování - výpis aktuální váhy
        # print(f"Lesion ratio: {lesion_ratio.item():.4f}, Dynamic weight: {dynamic_weight.item():.2f}")
        
        # Apply dynamic weight
        return dynamic_weight * mae

class WGANLoss:
    """
    Implementace WGAN-GP loss pro stabilní trénink GAN.
    Používá Wasserstein vzdálenost a gradient penalty.
    """
    def __init__(self, lambda_gp=10.0):
        """
        Inicializace WGAN-GP loss.
        
        Args:
            lambda_gp: Váha gradient penalty
        """
        self.lambda_gp = lambda_gp
    
    def discriminator_loss(self, real_validity, fake_validity):
        """
        Wasserstein loss pro diskriminátor.
        
        Args:
            real_validity: Skóre diskriminátoru pro reálné vzorky
            fake_validity: Skóre diskriminátoru pro vygenerované vzorky
            
        Returns:
            Loss pro diskriminátor
        """
        # Kritérium Wasserstein distance: max E[D(real)] - E[D(fake)]
        # Implementováno jako minimalizace: E[D(fake)] - E[D(real)]
        return torch.mean(fake_validity) - torch.mean(real_validity)
    
    def generator_loss(self, fake_validity):
        """
        Wasserstein loss pro generátor.
        
        Args:
            fake_validity: Skóre diskriminátoru pro vygenerované vzorky
            
        Returns:
            Loss pro generátor
        """
        # Kritérium Wasserstein distance: min -E[D(fake)]
        # Implementováno jako minimalizace: -E[D(fake)]
        return -torch.mean(fake_validity)
    
    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples, pseudo_healthy, label):
        """
        Výpočet gradient penalty pro WGAN-GP.
        
        Args:
            discriminator: Diskriminátor model
            real_samples: Reálné ADC mapy
            fake_samples: Vygenerované ADC mapy
            pseudo_healthy: Pseudo-zdravé vstupy
            label: Masky lézí
            
        Returns:
            Gradient penalty term
        """
        # Náhodná váha pro interpolaci mezi reálnými a generovanými vzorky
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=real_samples.device)
        
        # Získáme náhodné body na přímce mezi reálnými a generovanými vzorky
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Vyhodnotíme diskriminátor v interpolovaných bodech
        d_interpolates = discriminator(pseudo_healthy, label, interpolates)
        
        # Automatický výpočet gradientů
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Spočítáme normu gradientu pro každý batch
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # Gradient penalty: (||∇D(x)||_2 - 1)^2
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty * self.lambda_gp

class BiasedMAELoss(nn.Module):
    """
    MAE loss s preferencí nižších hodnot v lézích, založená na medicínské znalosti.
    Penalizuje více případy, kdy předpověď má vyšší hodnoty než okolí (což je proti očekávání).
    """
    def __init__(self, weight=5.0, high_penalty_factor=2.0, neighbor_radius=2):
        super(BiasedMAELoss, self).__init__()
        self.weight = weight
        self.high_penalty_factor = high_penalty_factor  # Faktor, o kolik víc penalizujeme vyšší hodnoty
        self.neighbor_radius = neighbor_radius  # Radius pro výpočet průměrných hodnot okolí
        
    def forward(self, pred, target, mask, orig_range=None):
        """
        Args:
            pred: Predikované hodnoty (B, 1, D, H, W)
            target: Skutečné hodnoty (B, 1, D, H, W)
            mask: Maska léze (B, 1, D, H, W)
            orig_range: Původní rozsah hodnot pro škálování
            
        Returns:
            Weighted biased MAE loss
        """
        # Vytvoření binární masky
        binary_mask = (mask > 0.5).float()
        
        # Ověření, že maska obsahuje nenulové hodnoty
        mask_sum = binary_mask.sum()
        if mask_sum < 1:
            return torch.tensor(0.0, device=pred.device)
        
        # Absolutní rozdíl
        abs_diff = torch.abs(pred - target) * binary_mask
        
        # Výpočet průměrných hodnot okolí pro každý pixel léze
        # Nejprve rozšíříme masku pro zahrnutí okolí
        kernel_size = 2 * self.neighbor_radius + 1
        padding = self.neighbor_radius
        
        # Použijeme průměrový pooling pro výpočet okolních hodnot
        neighbor_values = F.avg_pool3d(
            target * (1 - binary_mask),  # Pouze hodnoty mimo lézi
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # Identifikace míst, kde predikce je vyšší než okolí
        # Což je proti medicínskému očekávání (léze mají nižší ADC)
        higher_than_neighbors = (pred > neighbor_values) * binary_mask
        
        # Aplikace vyšší penalizace na tyto případy
        biased_diff = abs_diff * (1 + (self.high_penalty_factor - 1) * higher_than_neighbors)
        
        # Výpočet celkové loss
        loss = biased_diff.sum() / mask_sum
        
        # Aplikace původního rozsahu hodnot, pokud je k dispozici
        if orig_range is not None:
            orig_min, orig_max = orig_range
            if isinstance(orig_min, torch.Tensor) and isinstance(orig_max, torch.Tensor):
                try:
                    range_width = orig_max.item() - orig_min.item()
                    loss = loss * range_width
                except (RuntimeError, ValueError) as e:
                    if torch.isnan(orig_min).any() or torch.isnan(orig_max).any() or \
                       torch.isinf(orig_min).any() or torch.isinf(orig_max).any():
                        print(f"Warning: NaN/Inf in BiasedMAE loss! Range: {orig_min.item()} to {orig_max.item()}")
                    else:
                        print(f"Error in BiasedMAE loss calculation: {e}")
                    loss = loss  # Use unnormalized loss as fallback
            else:
                range_width = orig_max - orig_min
                loss = loss * range_width
                
        return self.weight * loss

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
    
    # Inicializace diskriminátoru pro CGAN
    discriminator = LesionDiscriminator(in_channels=3).to(device)
    
    # Define loss functions based on the selected loss type
    loss_type = args.loss_type if hasattr(args, 'loss_type') else 'mae'
    print(f"Using loss type: {loss_type}")
    
    # Initialize loss functions - všechny rekonstrukční ztráty mají nižší váhy pro dominanci adversariální složky
    # All reconstruction losses have reduced weights to prioritize the adversarial component
    masked_mae_loss = MaskedMAELoss(weight=5.0).to(device)
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0, weight=5.0).to(device)
    dynamic_mae_loss = DynamicWeightedMAELoss(base_weight=7.5, scaling_factor=8.0).to(device)
    biased_mae_loss = BiasedMAELoss(weight=7.5, high_penalty_factor=3.0, neighbor_radius=2).to(device)
    gradient_loss = GradientSmoothingLoss(weight=0.01).to(device)
    
    # Inicializace perceptuálního lossu - drasticky snížená váha, protože měla příliš vysoké hodnoty
    perceptual_loss = PerceptualLoss(weight=0.05).to(device)
    
    # Inicializace WGAN lossu pro adversariální trénink
    wgan_loss = WGANLoss(lambda_gp=10.0)
    
    # Váha adversariální loss - hlavní komponent GAN architektury
    # Higher weight emphasizes the adversarial aspect of the model
    # Váha 20.0 dává adversariální složce dominantní roli oproti rekonstrukčním ztrátám
    adv_weight = args.adv_weight if hasattr(args, 'adv_weight') else 20.0
    print(f"Adversarial weight: {adv_weight}")
    
    # Initialize optimizers
    optimizer_G = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Initialize patch extractor
    patch_extractor = PatchExtractor(patch_size=(16, 16, 16))
    
    # Define max patches per volume for training - this prevents excessive patch extraction
    max_patches_per_volume = args.max_patches if hasattr(args, 'max_patches') else 32
    max_val_patches_per_volume = 16  # Can use fewer patches for validation
    
    # Training loop
    best_val_lesion_loss = float('inf')
    best_val_lesion_dice = 0.0
    best_val_lesion_mae = float('inf')
    best_val_lesion_ssim = 0.0
    
    # Inicializace start_epoch pro případy, kdy není použit --resume
    args.start_epoch = 0
    
    # Načíst checkpoint, pokud je k dispozici
    if hasattr(args, 'resume') and args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        discriminator.load_state_dict(checkpoint.get('discriminator_state_dict', {}))  # Bude fungovat i pro starší checkpointy
        optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'discriminator_optimizer_state_dict' in checkpoint:
            optimizer_D.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        
        args.start_epoch = checkpoint.get('epoch', 0)
        best_val_lesion_loss = checkpoint.get('best_val_lesion_loss', float('inf'))
        best_val_lesion_mae = checkpoint.get('best_val_lesion_mae', float('inf'))
        best_val_lesion_dice = checkpoint.get('best_val_lesion_dice', 0.0)
        best_val_lesion_ssim = checkpoint.get('best_val_lesion_ssim', 0.0)
        print(f"Resuming from epoch {args.start_epoch}")
    
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        discriminator.train()
        epoch_loss = 0
        epoch_mae_loss = 0
        epoch_focal_loss = 0
        epoch_dynamic_mae_loss = 0
        epoch_biased_mae_loss = 0
        epoch_gradient_loss = 0
        epoch_perceptual_loss = 0
        epoch_adv_loss = 0
        epoch_d_loss = 0
        epoch_g_raw_loss = 0  # Čistá generátor loss bez váhy
        epoch_d_raw_loss = 0  # Diskriminátor loss bez GP
        train_samples = 0
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            pseudo_healthy = batch_data["pseudo_healthy"].to(device)
            adc = batch_data["adc"].to(device)
            label = batch_data["label"].to(device)
            
            # Get the original ranges from the dataset for loss calculation
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
                    # Also ensure the patch actually contains lesions
                    if patches_label[i].sum() > 0:
                        valid_patches.append(i)
            
            # Skip if no valid patches with lesions
            if not valid_patches:
                print(f"Warning: No valid patches with lesions found for training sample {batch_idx}")
                continue
                
            # Limit the number of patches to prevent excessive memory usage and imbalanced training
            if len(valid_patches) > args.max_patches:
                # Use random sampling
                random.seed(epoch * 1000 + batch_idx)  # For reproducibility but different each epoch
                valid_patches = random.sample(valid_patches, args.max_patches)
            
            # Filter patches based on valid_patches
            filtered_patches_ph = [patches_ph[i] for i in valid_patches]
            filtered_patches_label = [patches_label[i] for i in valid_patches]
            filtered_patches_adc = [patches_adc[i] for i in valid_patches]
            filtered_patches_coords = [patch_coords[i] for i in valid_patches]
            
            # Debug info about lesion content
            if len(filtered_patches_label) > 0:
                avg_lesion_voxels = sum(patch.sum().item() for patch in filtered_patches_label) / len(filtered_patches_label)
            
            batch_loss = 0
            for ph_patch, label_patch, adc_patch in zip(filtered_patches_ph, filtered_patches_label, filtered_patches_adc):
                # Add batch dimension
                ph_patch = ph_patch.unsqueeze(0).to(device)
                label_patch = label_patch.unsqueeze(0).to(device)
                adc_patch = adc_patch.unsqueeze(0).to(device)
                
                # ------
                # KROK 1: Trénink diskriminátoru (WGAN-GP)
                # ------
                for _ in range(1):  # Train discriminator once per generator update
                    optimizer_D.zero_grad()
                    
                    # Generovat fake výstup
                    with torch.no_grad():
                        fake_adc = model(ph_patch, label_patch)
                    
                    # Spočítat validitu pro real a fake
                    real_validity = discriminator(ph_patch, label_patch, adc_patch)
                    fake_validity = discriminator(ph_patch, label_patch, fake_adc.detach())
                    
                    # Wasserstein loss pro diskriminátor
                    d_loss = wgan_loss.discriminator_loss(real_validity, fake_validity)
                    
                    # Gradient penalty
                    gp = wgan_loss.compute_gradient_penalty(
                        discriminator, adc_patch, fake_adc.detach(), ph_patch, label_patch
                    )
                    
                    # Celková loss pro diskriminátor
                    d_total_loss = d_loss + gp
                    
                    # Backpropagation pro diskriminátor
                    d_total_loss.backward()
                    optimizer_D.step()
                
                # ------
                # KROK 2: Trénink generátoru
                # ------
                optimizer_G.zero_grad()
                
                # Generovat fake výstup
                output = model(ph_patch, label_patch)
                
                # Adversariální loss - WGAN
                fake_validity = discriminator(ph_patch, label_patch, output)
                g_adv_loss = wgan_loss.generator_loss(fake_validity)
                
                # Calculate losses based on selected loss type
                if loss_type == 'mae':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    focal_value = torch.tensor(0.0, device=device)  # Not used
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = mae_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                    
                elif loss_type == 'focal':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)  # For tracking
                    focal_value = focal_loss(output, adc_patch, label_patch, adc_orig_range)
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = focal_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                    
                elif loss_type == 'dynamic_mae':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)  # For tracking
                    focal_value = torch.tensor(0.0, device=device)  # Not used
                    dynamic_mae_value = dynamic_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = dynamic_mae_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                    
                elif loss_type == 'combined':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    focal_value = focal_loss(output, adc_patch, label_patch, adc_orig_range)
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = mae_value + focal_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                    
                elif loss_type == 'perceptual':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)  # For tracking
                    focal_value = torch.tensor(0.0, device=device)  # Not used
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = perceptual_value + gradient_value
                    loss = recon_loss + adv_weight * g_adv_loss
                    
                elif loss_type == 'biased_mae':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)  # For tracking
                    focal_value = torch.tensor(0.0, device=device)  # Not used
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    biased_mae_value = biased_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = biased_mae_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                
                else:  # Default to MAE
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    focal_value = torch.tensor(0.0, device=device)  # Not used
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = mae_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                
                # Backpropagation pro generátor
                loss.backward()
                optimizer_G.step()
                
                # Přidání hodnot lossu pro monitoring
                batch_loss += loss.item()
                epoch_loss += loss.item()
                epoch_mae_loss += mae_value.item() if loss_type in ['mae', 'combined'] else 0
                epoch_focal_loss += focal_value.item() if loss_type in ['focal', 'combined'] else 0
                epoch_dynamic_mae_loss += dynamic_mae_value.item() if loss_type == 'dynamic_mae' else 0
                epoch_biased_mae_loss += biased_mae_value.item() if loss_type == 'biased_mae' else 0
                epoch_gradient_loss += gradient_value.item()
                epoch_perceptual_loss += perceptual_value.item()
                epoch_adv_loss += g_adv_loss.item()  # Přidání adversariální ztráty pro monitoring
                epoch_d_loss += d_total_loss.item()  # Přidání diskriminátor ztráty pro monitoring
                epoch_g_raw_loss += g_adv_loss.item()  # Čistá WGAN generátor loss
                epoch_d_raw_loss += d_loss.item()  # Čistá WGAN diskriminátor loss
                train_samples += 1
            
            # Average loss over patches
            if len(filtered_patches_ph) > 0:
                avg_batch_loss = batch_loss / len(filtered_patches_ph)
                epoch_loss += avg_batch_loss
                
        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        avg_epoch_mae_loss = epoch_mae_loss / max(1, train_samples) if loss_type not in ['focal', 'dynamic_mae'] else 0
        avg_epoch_focal_loss = epoch_focal_loss / max(1, train_samples) if loss_type not in ['mae', 'dynamic_mae'] else 0
        avg_epoch_dynamic_mae_loss = epoch_dynamic_mae_loss / max(1, train_samples) if loss_type == 'dynamic_mae' else 0
        avg_epoch_biased_mae_loss = epoch_biased_mae_loss / max(1, train_samples) if loss_type == 'biased_mae' else 0
        avg_epoch_gradient_loss = epoch_gradient_loss / max(1, train_samples)
        avg_epoch_perceptual_loss = epoch_perceptual_loss / max(1, train_samples)
        avg_epoch_adv_loss = epoch_adv_loss / max(1, train_samples)  # Průměrná adversariální ztráta
        avg_epoch_d_loss = epoch_d_loss / max(1, train_samples)  # Průměrná diskriminátor ztráta
        avg_epoch_g_raw_loss = epoch_g_raw_loss / max(1, train_samples)  # Průměrná čistá generátor loss
        avg_epoch_d_raw_loss = epoch_d_raw_loss / max(1, train_samples)  # Průměrná čistá diskriminátor loss
        
        # Get actual values by dividing by the weight for display purposes only
        actual_mae = avg_epoch_mae_loss / 5.0 if loss_type not in ['focal', 'dynamic_mae'] else 0
        actual_focal = avg_epoch_focal_loss / 5.0 if loss_type not in ['mae', 'dynamic_mae'] else 0
        actual_dynamic_mae = avg_epoch_dynamic_mae_loss / 7.5 if loss_type == 'dynamic_mae' else 0
        actual_biased_mae = avg_epoch_biased_mae_loss / 7.5 if loss_type == 'biased_mae' else 0
        
        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")
        if loss_type not in ['focal', 'dynamic_mae', 'biased_mae']:
            print(f"  - MAE: {actual_mae:.4f} (unweighted)")
        if loss_type not in ['mae', 'dynamic_mae', 'biased_mae']:
            print(f"  - Focal: {actual_focal:.4f} (unweighted)")
        if loss_type == 'dynamic_mae':
            print(f"  - Dynamic MAE: {actual_dynamic_mae:.4f} (unweighted)")
        if loss_type == 'biased_mae':
            print(f"  - Biased MAE: {actual_biased_mae:.4f} (unweighted)")
        print(f"  - Gradient Loss: {avg_epoch_gradient_loss:.4f}")
        print(f"  - Perceptual Loss: {avg_epoch_perceptual_loss:.4f}")
        
        # Výpis GAN komponent
        print(f"\n  ===== GAN PERFORMANCE =====")
        print(f"  - Generator Loss (weighted): {adv_weight * avg_epoch_g_raw_loss:.4f}")
        print(f"  - Discriminator Loss (total): {avg_epoch_d_loss:.4f}")
        print(f"  -----------------------")
        print(f"  - Generator WGAN Raw:   {avg_epoch_g_raw_loss:.4f}")
        print(f"  - Discriminator WGAN Raw: {avg_epoch_d_raw_loss:.4f}")
        print(f"  ===========================\n")
        
        # Validation
        model.eval()
        discriminator.eval()
        val_loss = 0.0
        val_lesion_mae = 0.0
        val_lesion_focal = 0.0
        val_lesion_dynamic_mae = 0.0
        val_lesion_biased_mae = 0.0
        val_lesion_ssim = 0.0
        val_lesion_dice = 0.0
        val_lesion_perceptual = 0.0
        val_examples = 0
        
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
                    
                    # Filter out patches with invalid coordinates (negative values)
                    valid_patches = []
                    for i, (patch, coords) in enumerate(zip(patches_ph, patch_coords)):
                        if all(c >= 0 for c in coords):
                            # Only include patches that contain lesions
                            if patches_label[i].sum() > 0:
                                valid_patches.append(i)
                    
                    if not valid_patches:
                        print(f"Warning: No valid patches with lesions found for validation sample {val_batch_idx}")
                        continue
                        
                    # Limit the number of patches to prevent excessive validation on large lesions
                    total_valid_patches = len(valid_patches)
                    max_val_patches = 16  # Lower limit for validation to speed it up
                    if total_valid_patches > max_val_patches:
                        # Use deterministic sampling for validation (fixed seed for reproducibility)
                        seed = 42 + val_batch_idx
                        random.seed(seed)
                        valid_patches = random.sample(valid_patches, max_val_patches)
                    
                    # Filter patches based on valid_patches list
                    filtered_patches_ph = [patches_ph[i] for i in valid_patches]
                    filtered_patches_label = [patches_label[i] for i in valid_patches]
                    filtered_patches_adc = [patches_adc[i] for i in valid_patches]
                    filtered_patch_coords = [patch_coords[i] for i in valid_patches]
                    
                    # Debug info about lesion content
                    if len(filtered_patches_label) > 0:
                        avg_lesion_voxels = sum(patch.sum().item() for patch in filtered_patches_label) / len(filtered_patches_label)
                    
                    # Convert patches to tensors and move to device
                    filtered_patches_ph = [p.to(device) for p in filtered_patches_ph]
                    filtered_patches_label = [p.to(device) for p in filtered_patches_label]
                    filtered_patches_adc = [p.to(device) for p in filtered_patches_adc]
                    # DO NOT move patch coordinates to device - they are tuples not tensors
                    # filtered_patch_coords are left as tuples of integers
                    
                    # Process patches one by one
                    output_patches = []
                    valid_patch_coords = []
                    
                    # Collect metrics for this batch
                    batch_mae = 0.0
                    batch_ssim = 0.0
                    batch_dice = 0.0
                    batch_focal = 0.0
                    batch_dynamic_mae = 0.0
                    batch_biased_mae = 0.0
                    batch_perceptual = 0.0
                    num_valid_patches = 0
                    
                    for patch_idx, (ph_patch, label_patch, adc_patch, coords) in enumerate(zip(
                            filtered_patches_ph, filtered_patches_label, 
                            filtered_patches_adc, filtered_patch_coords)):
                        try:
                            # Add batch dimension
                            ph_patch = ph_patch.unsqueeze(0)  
                            label_patch = label_patch.unsqueeze(0)
                            adc_patch = adc_patch.unsqueeze(0)
                            
                            # Generate inpainted output
                            output_patch = model(ph_patch, label_patch)
                            
                            # Check output shape 
                            expected_shape = ph_patch.shape[2:]
                            output_shape = output_patch.shape[2:]
                            
                            if expected_shape != output_shape:
                                print(f"Warning: Output patch dimensions {output_shape} don't match expected dimensions {expected_shape}")
                                # Try to fix dimensions using interpolation
                                if len(output_shape) == len(expected_shape):
                                    try:
                                        # Use interpolate to resize to expected dimensions
                                        output_patch = torch.nn.functional.interpolate(
                                            output_patch,
                                            size=expected_shape,
                                            mode='trilinear',
                                            align_corners=False
                                        )
                                    except Exception as resize_err:
                                        print(f"Error resizing patch: {resize_err}")
                                        continue
                                else:
                                    # Skip this patch if dimensions don't match and can't be fixed
                                    continue
                            
                            # Calculate metrics for this patch
                            binary_mask = (label_patch > 0.5).float()
                            
                            # Always calculate MAE for validation regardless of loss type
                            lesion_voxels = binary_mask.sum()
                            if lesion_voxels > 0:
                                # Calculate MAE only in lesion areas (masked)
                                masked_pred = output_patch * binary_mask
                                masked_target = adc_patch * binary_mask
                                abs_diff = torch.abs(masked_pred - masked_target)
                                mae_patch = abs_diff.sum() / lesion_voxels  # Average only over lesion voxels
                                
                                # Apply scaling if original range is provided
                                if adc_orig_range is not None:
                                    mae_patch = mae_patch * (adc_orig_range[1] - adc_orig_range[0])
                            else:
                                mae_patch = torch.tensor(0.0, device=device)
                            
                            # Pro SSIM metriku - pouze pro zpětnou kompatibilitu
                            if binary_mask.sum() > 0:
                                try:
                                    from pytorch_msssim import ssim
                                    # Upravit dimenze pro SSIM, který očekává 4D nebo 5D tenzor
                                    if output_patch.dim() == 5:  # [B,C,D,H,W]
                                        # Pro 3D data vezmeme průměr přes D dimenzi
                                        ssim_val = 0
                                        depth = output_patch.shape[2]
                                        for d in range(depth):
                                            # Použijeme ssim pro každý 2D řez
                                            slice_ssim = ssim(
                                                output_patch[:,:,d],
                                                adc_patch[:,:,d],
                                                data_range=1.0,
                                                size_average=True
                                            )
                                            ssim_val += slice_ssim
                                        ssim_val /= depth  # Průměr přes všechny řezy
                                    else:
                                        # Pokud není 5D, nastavíme defaultní hodnotu
                                        ssim_val = 0.5
                                        
                                    ssim_patch_val = ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val
                                except:
                                    # Pokud pytorch_msssim není k dispozici, použijeme placeholder
                                    ssim_patch_val = 0.5
                            else:
                                ssim_patch_val = 0.0
                            
                            # Calculate Dice coefficient 
                            if binary_mask.sum() > 0:
                                masked_pred = output_patch * binary_mask
                                masked_target = adc_patch * binary_mask
                                
                                # Normalize values for thresholding
                                norm_pred = (masked_pred - masked_pred.min()) / (masked_pred.max() - masked_pred.min() + 1e-8)
                                norm_target = (masked_target - masked_target.min()) / (masked_target.max() - masked_target.min() + 1e-8)
                                
                                # Apply threshold (0.5) to get binary masks
                                pred_binary = (norm_pred > 0.5).float()
                                target_binary = (norm_target > 0.5).float()
                                
                                # Calculate Dice
                                intersection = (pred_binary * target_binary * binary_mask).sum()
                                dice_patch = (2. * intersection) / (
                                    (pred_binary * binary_mask).sum() + (target_binary * binary_mask).sum() + 1e-8
                                )
                            else:
                                dice_patch = torch.tensor(0.0, device=device)
                            
                            # Calculate Focal Loss for validation if using focal loss
                            if loss_type in ['focal', 'combined']:
                                try:
                                    focal_patch = focal_loss(output_patch, adc_patch, label_patch, adc_orig_range).item() / 25.0  # Divide by weight for raw value
                                except Exception as e:
                                    print(f"Error calculating focal loss: {e}")
                                    focal_patch = 0.0
                            else:
                                focal_patch = 0.0
                            
                            # Calculate Dynamic MAE for validation if using dynamic_mae loss
                            if loss_type in ['dynamic_mae']:
                                try:
                                    dynamic_mae_patch = dynamic_mae_loss(output_patch, adc_patch, label_patch, adc_orig_range).item() / 7.5  # Divide by base_weight for raw value
                                except Exception as e:
                                    print(f"Error calculating dynamic mae loss: {e}")
                                    dynamic_mae_patch = 0.0
                            else:
                                dynamic_mae_patch = 0.0
                                
                            # Calculate Biased MAE for validation if using biased_mae loss
                            if loss_type in ['biased_mae']:
                                try:
                                    biased_mae_patch = biased_mae_loss(output_patch, adc_patch, label_patch, adc_orig_range).item() / 7.5  # Divide by weight for raw value
                                except Exception as e:
                                    print(f"Error calculating biased mae loss: {e}")
                                    biased_mae_patch = 0.0
                            else:
                                biased_mae_patch = 0.0
                            
                            # Calculate Perceptual Loss for validation if using perceptual loss
                            if loss_type in ['combined', 'perceptual']:
                                try:
                                    perceptual_patch = perceptual_loss(output_patch, adc_patch, label_patch, adc_orig_range).item() / 0.05  # Divide by weight for raw value
                                except Exception as e:
                                    print(f"Error calculating perceptual loss: {e}")
                                    perceptual_patch = 0.0
                            else:
                                perceptual_patch = 0.0
                            
                            # Add metrics to batch totals
                            # Always add MAE regardless of loss type for validation (for comparison)
                            batch_mae += mae_patch.item()
                            batch_ssim += ssim_patch_val
                            batch_dice += dice_patch.item()
                            if loss_type in ['focal', 'combined']:
                                batch_focal += focal_patch
                            if loss_type in ['dynamic_mae']:
                                batch_dynamic_mae += dynamic_mae_patch
                            if loss_type in ['biased_mae']:
                                batch_biased_mae += biased_mae_patch
                            if loss_type in ['combined', 'perceptual']:
                                batch_perceptual += perceptual_patch
                            num_valid_patches += 1
                            
                            # Store patch for reconstruction
                            output_patches.append(output_patch[0])
                            valid_patch_coords.append(coords)
                            
                        except Exception as patch_error:
                            print(f"Error processing patch {patch_idx}: {patch_error}")
                            continue
                    
                    # Average metrics for this batch
                    if num_valid_patches > 0:
                        # Always calculate average MAE for validation
                        batch_mae /= num_valid_patches
                        batch_ssim /= num_valid_patches
                        batch_dice /= num_valid_patches
                        if loss_type in ['focal', 'combined']:
                            batch_focal /= num_valid_patches
                        if loss_type in ['dynamic_mae']:
                            batch_dynamic_mae /= num_valid_patches
                        if loss_type in ['biased_mae']:
                            batch_biased_mae /= num_valid_patches
                        if loss_type in ['combined', 'perceptual']:
                            batch_perceptual /= num_valid_patches
                        
                        # Add to validation totals
                        # Always add MAE to validation totals for comparison
                        val_lesion_mae += batch_mae 
                        val_lesion_ssim += batch_ssim
                        val_lesion_dice += batch_dice
                        if loss_type in ['focal', 'combined']:
                            val_lesion_focal += batch_focal
                        if loss_type in ['dynamic_mae']:
                            val_lesion_dynamic_mae += batch_dynamic_mae
                        if loss_type in ['biased_mae']:
                            val_lesion_biased_mae += batch_biased_mae
                        if loss_type in ['combined', 'perceptual']:
                            val_lesion_perceptual += batch_perceptual
                        val_examples += 1
                        
                        # Calculate total validation loss based on loss_type
                        if loss_type == 'mae':
                            batch_val_loss = batch_mae
                        elif loss_type == 'focal':
                            batch_val_loss = batch_focal
                        elif loss_type == 'dynamic_mae':
                            batch_val_loss = batch_dynamic_mae
                        elif loss_type == 'combined':
                            batch_val_loss = batch_mae * 0.5 + batch_focal * 0.5 + batch_perceptual * 0.5
                        elif loss_type == 'perceptual':
                            batch_val_loss = batch_perceptual
                        else:  # Default to MAE
                            batch_val_loss = batch_mae
                            
                        val_loss += batch_val_loss
                    
                    # Reconstruct full volume
                    if output_patches:
                        try:
                            # Ensure all patches have the same number of dimensions
                            patch_dims = [p.dim() for p in output_patches]
                            if len(set(patch_dims)) > 1:
                                print(f"Warning: Inconsistent patch dimensions: {patch_dims}")
                                # Try to normalize dimensions
                                for i in range(len(output_patches)):
                                    while output_patches[i].dim() < 4:  # Ensure 4D [C, Z, Y, X]
                                        output_patches[i] = output_patches[i].unsqueeze(0)
                                    while output_patches[i].dim() > 4:
                                        output_patches[i] = output_patches[i].squeeze(0)
                            
                            # Reconstruct validation volume
                            reconstructed = patch_extractor.reconstruct_from_patches(
                                output_patches, filtered_patch_coords, pseudo_healthy.shape
                            )
                            
                            # Apply mask to show inpainted regions
                            # Rest of reconstruction code...
                            
                        except Exception as e:
                            print(f"Error during validation reconstruction: {e}")
                except Exception as batch_error:
                    print(f"Error processing validation batch {val_batch_idx}: {batch_error}")
                    continue
        
        if val_examples > 0:
            # Calculate average metrics
            avg_val_loss = val_loss / val_examples
            
            # Always calculate MAE for validation regardless of loss type
            avg_val_lesion_mae = val_lesion_mae / val_examples
            # Divide by the weight factor (25.0) to match training MAE display
            avg_val_lesion_mae = avg_val_lesion_mae
                
            if loss_type in ['focal', 'combined']:
                avg_val_lesion_focal = val_lesion_focal / val_examples
            else:
                # Set to zero if not using this metric
                avg_val_lesion_focal = 0.0
                
            if loss_type in ['dynamic_mae']:
                avg_val_lesion_dynamic_mae = val_lesion_dynamic_mae / val_examples
            else:
                # Set to zero if not using this metric
                avg_val_lesion_dynamic_mae = 0.0
                
            avg_val_lesion_ssim = val_lesion_ssim / val_examples
            avg_val_lesion_dice = val_lesion_dice / val_examples
            avg_val_lesion_perceptual = val_lesion_perceptual / val_examples
            
            # Divide overall loss by the weight factor (25.0) to match training display
            avg_val_loss = avg_val_loss / 25.0
            
            # Print validation metrics summary
            print("\nValidation Results:")
            
            # Make overall loss label more specific based on loss type but always show MAE
            if loss_type == 'mae':
                print(f"Overall MAE Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f}")
            elif loss_type == 'focal':
                print(f"Overall Focal Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f} (for comparison)")
                print(f"Lesion Focal: {avg_val_lesion_focal:.4f}")
            elif loss_type == 'dynamic_mae':
                print(f"Overall Dynamic MAE Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f} (for comparison)")
                print(f"Lesion Dynamic MAE: {avg_val_lesion_dynamic_mae:.4f}")
            elif loss_type == 'biased_mae':
                print(f"Overall Biased MAE Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f} (for comparison)")
                print(f"Lesion Biased MAE: {avg_val_lesion_biased_mae:.4f}")
            elif loss_type == 'combined':
                print(f"Overall Combined Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f}")
                print(f"Lesion Focal: {avg_val_lesion_focal:.4f}")
            elif loss_type == 'perceptual':
                print(f"Overall Perceptual Loss: {avg_val_loss:.4f}")
                print(f"Lesion Perceptual: {avg_val_lesion_perceptual:.4f}")
            else:
                print(f"Overall Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f}")
                
            print(f"Lesion Dice: {avg_val_lesion_dice:.4f}")
            print(f"Lesion SSIM: {avg_val_lesion_ssim:.4f} (informační metrika, neoptimalizováno)")
            
            # Generate and save validation visualizations based on vis_freq
            if (epoch + 1) % args.vis_freq == 0:
                # Sample a few validation cases for visualization
                vis_indices = random.sample(range(len(val_loader)), min(3, len(val_loader)))
                
                for vis_idx in vis_indices:
                    try:
                        # Get a sample from validation set
                        vis_data = list(val_loader)[vis_idx]
                        
                        vis_ph = vis_data["pseudo_healthy"].to(device)
                        vis_adc = vis_data["adc"].to(device)
                        vis_label = vis_data["label"].to(device)
                        
                        # Generate reconstruction for visualization
                        with torch.no_grad():
                            # Extract patches
                            vis_patches_ph, vis_patches_label, vis_patches_adc, vis_patch_coords = patch_extractor.extract_patches_with_lesions(
                                vis_ph[0], vis_label[0], vis_adc[0]
                            )
                            
                            # Filter and limit patches for visualization
                            valid_vis_patches = []
                            for i, (patch, coords) in enumerate(zip(vis_patches_ph, vis_patch_coords)):
                                if all(c >= 0 for c in coords) and vis_patches_label[i].sum() > 0:
                                    valid_vis_patches.append(i)
                            
                            # Limit number of visualization patches - use even fewer to keep visualization manageable
                            max_vis_patches = 8
                            if len(valid_vis_patches) > max_vis_patches:
                                # For visualization, use a deterministic sample
                                random.seed(vis_idx * 10000)  # Fixed seed for reproducibility
                                valid_vis_patches = random.sample(valid_vis_patches, max_vis_patches)
                            
                            # Only use patches that contain lesions
                            vis_out_patches = []
                            filtered_vis_coords = []
                            
                            for p_idx in valid_vis_patches:
                                # Add batch dimension
                                p_ph = vis_patches_ph[p_idx].unsqueeze(0).to(device)
                                p_label = vis_patches_label[p_idx].unsqueeze(0).to(device)
                                
                                # Generate output
                                p_out = model(p_ph, p_label)
                                vis_out_patches.append(p_out[0])
                                filtered_vis_coords.append(vis_patch_coords[p_idx])
                        
                            # Skip if no patches with lesions
                            if len(vis_out_patches) == 0:
                                continue
                                
                            # Reconstruct full volume
                            reconstructed = patch_extractor.reconstruct_from_patches(
                                vis_out_patches, filtered_vis_coords, vis_ph.shape
                            )
                            
                            # Apply the lesion mask to only modify lesion regions
                            # Oprava: Zajistíme, že tenzor má správné rozměry pro max_pool3d (5D tenzor [B,C,D,H,W])
                            # Zkontrolujeme rozměry
                            if vis_label.dim() == 4:  # [C,D,H,W]
                                vis_label_5d = vis_label.unsqueeze(0)  # Přidáme batch dimenzi -> [1,C,D,H,W]
                            elif vis_label.dim() == 5:  # [B,C,D,H,W]
                                vis_label_5d = vis_label
                            else:
                                raise ValueError(f"Neočekávaný rozměr vis_label: {vis_label.shape}")
                                
                            dilated_mask = torch.nn.functional.max_pool3d(
                                vis_label_5d, kernel_size=3, stride=1, padding=1
                            )
                            
                            # Upravíme také tvar rekonstruovaného výstupu a pseudozdravého vstupu
                            if reconstructed.dim() < 5:
                                reconstructed = reconstructed.unsqueeze(0)
                            
                            if vis_ph.dim() == 4:  # [C,D,H,W]
                                vis_ph_5d = vis_ph.unsqueeze(0)  # [1,C,D,H,W]
                            elif vis_ph.dim() == 5:  # [B,C,D,H,W]
                                vis_ph_5d = vis_ph
                            else:
                                raise ValueError(f"Neočekávaný rozměr vis_ph: {vis_ph.shape}")
                            
                            # Create final output - copy pseudo-healthy and replace only in lesion regions
                            final_output = vis_ph_5d.clone()
                            final_output = final_output * (1 - dilated_mask) + reconstructed * dilated_mask
                            
                            # Get the original ranges for metrics
                            adc_orig_range_str = None
                            if "original_ranges" in vis_data and "adc" in vis_data["original_ranges"]:
                                orig_range = vis_data["original_ranges"]["adc"]
                                adc_orig_range_str = f"{float(orig_range[0]):.4f} to {float(orig_range[1]):.4f}"
                            
                            # Create metrics dictionary for visualization
                            vis_metrics = {
                                "Epoch": epoch + 1,
                                "Dataset": "Validation",
                                "Loss Type": loss_type,
                                "Sample": f"{vis_idx+1}/{len(val_loader)}"
                            }
                            
                            # Always add MAE to metrics regardless of loss type for comparison
                            vis_metrics["Lesion MAE"] = avg_val_lesion_mae
                                
                            if loss_type in ['focal', 'combined']:
                                vis_metrics["Lesion Focal"] = avg_val_lesion_focal
                                
                            # Add perceptual loss to visualization metrics
                            if loss_type in ['perceptual', 'combined']:
                                vis_metrics["Lesion Perceptual"] = avg_val_lesion_perceptual
                                
                            # Add remaining metrics
                            vis_metrics["Lesion Dice"] = avg_val_lesion_dice
                            vis_metrics["Lesion SSIM (info)"] = avg_val_lesion_ssim
                            vis_metrics["Original ADC Range"] = adc_orig_range_str if adc_orig_range_str else "Not available"
                            
                            # Save visualization
                            vis_output_file = os.path.join(
                                args.output_dir, 
                                f"val_visualization_epoch{epoch+1}_sample{vis_idx+1}.pdf"
                            )
                            
                            visualize_results(
                                vis_ph, 
                                vis_label, 
                                final_output.squeeze(0) if final_output.dim() > 4 else final_output, 
                                vis_adc, 
                                vis_output_file,
                                metrics=vis_metrics
                            )
                            
                            print(f"Saved validation visualization to {vis_output_file}")
                    except Exception as vis_error:
                        print(f"Error creating visualization for validation sample {vis_idx}: {vis_error}")
            
            # Save best models based on metrics
            # Track the best value for each metric type
            best_val_lesion_focal = getattr(train, 'best_val_lesion_focal', float('inf')) if loss_type in ['focal', 'combined'] else float('inf')
            
            # Prioritize metric based on loss type
            if loss_type == 'mae':
                # Prioritize MAE (lower is better)
                if avg_val_lesion_mae < best_val_lesion_mae:
                    best_val_lesion_mae = avg_val_lesion_mae
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lesion_mae_model.pth"))
                    print(f"Saved new best model with lesion MAE: {best_val_lesion_mae:.4f}")
                    
            elif loss_type == 'focal':
                # Prioritize Focal Loss (lower is better)
                if avg_val_lesion_focal < best_val_lesion_focal:
                    best_val_lesion_focal = avg_val_lesion_focal
                    train.best_val_lesion_focal = best_val_lesion_focal  # Store for future comparisons
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lesion_focal_model.pth"))
                    print(f"Saved new best model with lesion Focal: {best_val_lesion_focal:.4f}")
                    
            elif loss_type == 'combined':
                # Save both metrics
                if avg_val_lesion_mae < best_val_lesion_mae:
                    best_val_lesion_mae = avg_val_lesion_mae
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lesion_mae_model.pth"))
                    print(f"Saved new best model with lesion MAE: {best_val_lesion_mae:.4f}")
                    
                if avg_val_lesion_focal < best_val_lesion_focal:
                    best_val_lesion_focal = avg_val_lesion_focal
                    train.best_val_lesion_focal = best_val_lesion_focal  # Store for future comparisons
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lesion_focal_model.pth"))
                    print(f"Saved new best model with lesion Focal: {best_val_lesion_focal:.4f}")
            
            # Always save Dice as a secondary metric (higher is better)
            if avg_val_lesion_dice > best_val_lesion_dice:
                best_val_lesion_dice = avg_val_lesion_dice
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lesion_dice_model.pth"))
                print(f"Saved new best model with lesion Dice: {best_val_lesion_dice:.4f}")
            
            # SSIM is just informational now
            if avg_val_lesion_ssim > best_val_lesion_ssim:
                best_val_lesion_ssim = avg_val_lesion_ssim
                # Save with experimental suffix
                torch.save(model.state_dict(), os.path.join(args.output_dir, "experimental_lesion_ssim_model.pth"))
                print(f"Saved experimental model with lesion SSIM: {best_val_lesion_ssim:.4f} (neoptimalizovaná metrika)")
            
            # Save checkpoint with all metrics
            if (epoch + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'loss': avg_epoch_loss,
                    'mae_loss': avg_epoch_mae_loss,
                    'focal_loss': avg_epoch_focal_loss,
                    'gradient_loss': avg_epoch_gradient_loss,
                    'perceptual_loss': avg_epoch_perceptual_loss,
                    'val_loss': avg_val_loss,
                    'val_lesion_mae': avg_val_lesion_mae,
                    'val_lesion_focal': avg_val_lesion_focal,
                    'val_lesion_ssim': avg_val_lesion_ssim,  # Still save for tracking
                    'val_lesion_dice': avg_val_lesion_dice
                }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        else:
            print("Warning: No valid validation examples found")

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
    patch_extractor = PatchExtractor(patch_size=(16, 16, 16))
    
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
        
        # Filter out patches with invalid coordinates (negative values)
        valid_patches = []
        for i, (patch, coords) in enumerate(zip(patches_ph, patch_coords)):
            if all(c >= 0 for c in coords):
                # Also ensure the patch actually contains lesions
                if patches_label[i].sum() > 0:
                    valid_patches.append(i)
        
        # Skip if no valid patches with lesions
        if not valid_patches:
            print(f"Warning: No valid patches with lesions found in the input image")
            return
            
        # Limit the number of patches to prevent excessive memory usage
        max_inference_patches = 32  # You can adjust this or make it a parameter
        if len(valid_patches) > max_inference_patches:
            # Use deterministic sampling for inference
            random.seed(42)  # Fixed seed for reproducibility
            valid_patches = random.sample(valid_patches, max_inference_patches)
            print(f"Limiting from {len(patches_ph)} to {max_inference_patches} patches for inference")
        
        # Filter patches based on valid_patches list
        filtered_patches_ph = [patches_ph[i] for i in valid_patches]
        filtered_patches_label = [patches_label[i] for i in valid_patches]
        filtered_patch_coords = [patch_coords[i] for i in valid_patches]
        
        # Debug info about lesion content
        if len(filtered_patches_label) > 0:
            avg_lesion_voxels = sum(patch.sum().item() for patch in filtered_patches_label) / len(filtered_patches_label)
            print(f"Inference: {len(filtered_patches_label)} patches, avg lesion voxels per patch: {avg_lesion_voxels:.1f}")
        
        output_patches = []
        for ph_patch, label_patch in zip(filtered_patches_ph, filtered_patches_label):
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
                output_patches, filtered_patch_coords, pseudo_healthy.shape
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
    # Create argument parser
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
    train_parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch number")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    train_parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint frequency")
    train_parser.add_argument("--vis_freq", type=int, default=1, help="Visualization frequency")
    train_parser.add_argument("--loss_type", type=str, default="mae", 
                        choices=["mae", "focal", "combined", "dynamic_mae", "perceptual", "biased_mae"], 
                        help="Loss type (mae, focal, combined, dynamic_mae, perceptual, biased_mae)")
    train_parser.add_argument("--max_patches", type=int, default=32, help="Maximum number of patches per volume")
    train_parser.add_argument("--adv_weight", type=float, default=20.0, help="Adversarial weight for WGAN-GP")
    train_parser.add_argument("--resume", type=str, help="Path to checkpoint for resuming training")
    
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

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import monai
from monai.networks.nets import SwinUNETR, AttentionUnet
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
        
        print(f"{'Training' if train else 'Validation'} dataset contains {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_dict = self.data[idx]
        
        # Load data using SimpleITK
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
        
        return data_dict

class PatchExtractor:
    def __init__(self, patch_size=(16, 16, 16), overlap=0.5, augment_patches=True, num_augmented_patches=3):
        self.patch_size = patch_size
        self.overlap = overlap
        self.augment_patches = augment_patches
        self.num_augmented_patches = num_augmented_patches
    
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
            
            return [ph_patch], [label_patch], [adc_patch], [(z_start, y_start, x_start)]
        
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
            # If still no patches with lesions after max attempts, return random patch
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
            
            return [ph_patch], [label_patch], [adc_patch], [(z_start, y_start, x_start)]
        
        return patches_ph, patches_label, patches_adc, patch_coords
    
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
        
        # For smaller patches, we need to be more careful with offsets to not lose the lesion
        # Smaller random offsets for 16x16x16 patches to avoid losing the lesion
        for i in range(self.num_augmented_patches + 2):  # Add more augmented patches for small lesions
            # Random offset in each dimension, but smaller to ensure lesion is captured
            z_offset = random.randint(-self.patch_size[0]//6, self.patch_size[0]//6)
            y_offset = random.randint(-self.patch_size[1]//6, self.patch_size[1]//6)
            x_offset = random.randint(-self.patch_size[2]//6, self.patch_size[2]//6)
            offsets.append((z_offset, y_offset, x_offset))
        
        # Add very small offsets to see lesion from multiple angles
        offsets.extend([
            (self.patch_size[0]//10, self.patch_size[1]//10, self.patch_size[2]//10),
            (-self.patch_size[0]//10, -self.patch_size[1]//10, -self.patch_size[2]//10),
            (self.patch_size[0]//10, -self.patch_size[1]//10, self.patch_size[2]//10),
            (-self.patch_size[0]//10, self.patch_size[1]//10, -self.patch_size[2]//10),
            (0, self.patch_size[1]//10, 0),  # Subtle y-axis shifts
            (0, -self.patch_size[1]//10, 0),
            (self.patch_size[0]//10, 0, 0),  # Subtle z-axis shifts
            (-self.patch_size[0]//10, 0, 0),
            (0, 0, self.patch_size[2]//10),  # Subtle x-axis shifts
            (0, 0, -self.patch_size[2]//10)
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
    
    def _cluster_lesion_voxels(self, non_zero_indices, min_distance=8):
        """Simple clustering of lesion voxels to find centers"""
        if len(non_zero_indices) == 0:
            return []
        
        # Use smaller min_distance for smaller patches to identify more clusters
        # 8 is half of our patch dimension (16x16x16)
        
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
        
        # For very small lesions with few centers, add the arithmetic mean of all voxels
        # as an additional center to ensure coverage
        if len(centers) <= 2 and len(indices_list) > 3:
            z_sum, y_sum, x_sum = 0, 0, 0
            for idx in indices_list:
                z_sum += idx[0]
                y_sum += idx[1]
                x_sum += idx[2]
            mean_center = [
                z_sum // len(indices_list),
                y_sum // len(indices_list),
                x_sum // len(indices_list)
            ]
            
            # Only add if not too close to existing centers
            is_unique = True
            for center in centers:
                distance = ((mean_center[0] - center[0])**2 + 
                           (mean_center[1] - center[1])**2 + 
                           (mean_center[2] - center[2])**2)**0.5
                if distance < min_distance // 2:  # Use smaller threshold
                    is_unique = False
                    break
            
            if is_unique:
                centers.append(mean_center)
                
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
        # Validate inputs
        if len(patches) == 0:
            raise ValueError("No patches provided for reconstruction")
        
        if len(patches) != len(patch_coords):
            raise ValueError(f"Number of patches ({len(patches)}) does not match number of coordinates ({len(patch_coords)})")
        
        # Force CUDA device - use the exact string "cuda:0" which is more reliable
        cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
        
        # AttentionUnet jako generátor, optimalizovaný pro malé patche 16x16x16
        # Upraveno pro předejití problému s batch normalizací při malých prostorových rozměrech
        self.generator = AttentionUnet(
            spatial_dims=3,  # 3D model
            in_channels=in_channels,  # Pseudo-healthy and lesion mask
            out_channels=out_channels,  # Inpainted ADC
            channels=(16, 32, 48, 64),  # Méně kanálů a méně vrstev
            strides=(2, 2, 2),  # Jen tři úrovně downsamplingu místo čtyř
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.2  # Small dropout to prevent overfitting
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
        raw_output = self.generator(x)
        
        # DŮLEŽITÉ: Použijeme masku k vytvoření finálního výstupu
        # Pouze oblast léze je nahrazena generátorem, zbytek zůstává stejný
        final_output = pseudo_healthy * (1.0 - label) + raw_output * label
        
        return final_output

class LesionDiscriminator(nn.Module):
    """
    Diskriminátor pro CGAN, který se zaměřuje pouze na oblasti lézí.
    Hodnotí pouze reálnost vygenerovaných lézí, ignoruje zbytek mozku.
    """
    def __init__(self, in_channels=3):
        super(LesionDiscriminator, self).__init__()
        
        # Vstup: [pseudo_healthy, label, adc], kde adc může být reálné nebo generované
        
        # Konvoluční bloky optimalizované pro malé patche 16x16x16
        self.conv_blocks = nn.Sequential(
            # První blok - zmenšit velikost
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),  # 16x16x16 -> 8x8x8
            
            # Druhý blok
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),  # 8x8x8 -> 4x4x4
            
            # Třetí blok
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Finální predikce - full patch output (4x4x4 -> 4x4x4)
            nn.Conv3d(128, 1, kernel_size=3, stride=1, padding=1)
        )
        
        # Inicializace vah pro stabilnější trénink
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm3d):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pseudo_healthy, label, adc):
        """
        Forward pass diskriminátoru.
        
        Args:
            pseudo_healthy: vstupní pseudo-zdravý obraz
            label: maska léze
            adc: skutečná nebo generovaná ADC mapa
        
        Returns:
            Diskriminační skóre zaměřené jen na oblast léze
        """
        # Spojení vstupů podél kanálové dimenze
        x = torch.cat([pseudo_healthy, label, adc], dim=1)
        
        # Průchod konvolučními bloky
        patch_predictions = self.conv_blocks(x)
        
        # Uložíme si velikost původní masky
        original_shape = label.shape
        
        # Interpolace predikcí zpět na původní velikost masky
        if patch_predictions.shape[2:] != label.shape[2:]:
            upsampled_predictions = torch.nn.functional.interpolate(
                patch_predictions,
                size=label.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        else:
            upsampled_predictions = patch_predictions
        
        # Aplikovat masku na predikce - hodnotíme pouze oblasti léze
        masked_predictions = upsampled_predictions * label
        
        # Průměrování pouze přes maskované oblasti
        mask_sum = torch.sum(label)
        if mask_sum > 0:
            masked_score = torch.sum(masked_predictions) / mask_sum
        else:
            masked_score = torch.tensor(0.0, device=x.device)
        
        return masked_score

class MaskedMAELoss(nn.Module):
    def __init__(self, weight=25.0):
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
        
        # Zkontrolujeme, zda maska obsahuje nějaké léze
        mask_sum = binary_mask.sum()
        if mask_sum <= 0:
            # Pokud nejsou léze, vrátíme nulovou ztrátu
            return torch.tensor(0.0, device=pred.device)
        
        # Apply mask to both prediction and target
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Calculate absolute difference
        abs_diff = torch.abs(masked_pred - masked_target)
        
        # Calculate mean only over masked regions
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
            
            # Scale the loss by the original range for more intuitivní výsledky
            range_factor = torch.abs(orig_max - orig_min)
            mae = mae * range_factor
            
            # Pro lepší debugging
            if torch.isnan(mae).any() or torch.isinf(mae).any():
                print(f"Warning: NaN/Inf in MAE loss! Range: {orig_min.item()} to {orig_max.item()}")
                return torch.tensor(1.0, device=pred.device) * self.weight
        
        # Apply weight - tato váha nyní bude výrazně vyšší (25.0 místo 1.0)
        return self.weight * mae

class SSIMLoss(nn.Module):
    def __init__(self, weight=5.0):
        super(SSIMLoss, self).__init__()
        self.weight = weight
    
    def forward(self, pred, target, mask):
        """
        Computes 1-SSIM as a loss (since higher SSIM is better)
        Strictly focuses on lesion regions
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            mask: Lesion mask tensor
            
        Returns:
            SSIM loss (1-SSIM) weighted
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Apply mask
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Only compute if mask has non-zero elements
        mask_sum = binary_mask.sum()
        if mask_sum <= 0:
            return torch.tensor(0.0, device=pred.device)
            
        # Compute means
        mu_pred = masked_pred.sum() / mask_sum
        mu_target = masked_target.sum() / mask_sum
        
        # Compute variances
        var_pred = ((masked_pred - mu_pred * binary_mask) ** 2).sum() / mask_sum
        var_target = ((masked_target - mu_target * binary_mask) ** 2).sum() / mask_sum
        
        # Compute covariance
        cov = ((masked_pred - mu_pred * binary_mask) * (masked_target - mu_target * binary_mask)).sum() / mask_sum
        
        # Constants for stability
        c1 = (0.01 * 1) ** 2
        c2 = (0.03 * 1) ** 2
        
        # Compute SSIM
        ssim = ((2 * mu_pred * mu_target + c1) * (2 * cov + c2)) / \
               ((mu_pred**2 + mu_target**2 + c1) * (var_pred + var_target + c2))
               
        # Ensure SSIM is in [0,1] range (clip if numerical issues)
        ssim = torch.clamp(ssim, 0.0, 1.0)
        
        # Return 1-SSIM as the loss (since higher SSIM is better)
        # Apply higher weight to make this term significant
        return self.weight * (1.0 - ssim)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=25.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, pred, target, mask, orig_range=None):
        """
        Compute Focal Loss for regression, focusing more on difficult examples (large errors)
        
        Args:
            pred: Predicted tensor
            target: Target tensor  
            mask: Lesion mask tensor (to focus only on lesions)
            orig_range: Original intensity range as tuple (min, max) before normalization
            
        Returns:
            Weighted Focal Loss
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Check if mask contains any lesions
        mask_sum = binary_mask.sum()
        if mask_sum <= 0:
            # If no lesions, return zero loss
            return torch.tensor(0.0, device=pred.device)
        
        # Apply mask to both prediction and target
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Calculate absolute difference
        abs_diff = torch.abs(masked_pred - masked_target)
        
        # Normalize the diff to [0, 1] for the focal weight calculation
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
            
            range_factor = torch.abs(orig_max - orig_min)
            norm_diff = abs_diff / range_factor
        else:
            # Assume data is already normalized
            norm_diff = abs_diff
        
        # Apply focal weight: (1 - e^(-norm_diff))^gamma
        # This gives higher weight to larger errors
        focal_weight = (1 - torch.exp(-norm_diff * 5.0)).pow(self.gamma)  # Multiply by 5 to make the curve steeper
        
        # Apply alpha weighting for positive examples
        weighted_loss = self.alpha * focal_weight * abs_diff
        
        # Take mean over masked region
        focal_loss = weighted_loss.sum() / mask_sum
        
        # Apply denormalization if original range is provided
        if orig_range is not None:
            focal_loss = focal_loss * range_factor
            
            # Handle NaN/Inf for debugging
            if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
                print(f"Warning: NaN/Inf in Focal loss! Range: {orig_min.item()} to {orig_max.item()}")
                return torch.tensor(1.0, device=pred.device) * self.weight
        
        # Apply weight
        return self.weight * focal_loss

class PerceptualLoss(nn.Module):
    def __init__(self, weight=10.0):
        super(PerceptualLoss, self).__init__()
        self.weight = weight
        
        # Využijeme konvoluční vrstvy jako extraktory příznaků
        # Pro 3D data s malými patch rozměry (16x16x16) použijeme 3D konvoluce s menšími filtry
        self.feature_extractor = nn.Sequential(
            # První vrstva - zachycení základních textur (16x16x16 -> 14x14x14)
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Druhá vrstva - zachycení středních detailů (14x14x14 -> 12x12x12)
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Třetí vrstva - zachycení vyšších detailů (12x12x12 -> 10x10x10)
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Zmrazíme parametry feature extraktoru - nebudeme je aktualizovat během tréninku
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target, mask, orig_range=None):
        """
        Výpočet perceptuálního lossu - porovnává příznaky extrahované z predikce a cíle
        Zaměřujeme se POUZE na oblast léze (použití masky)
        
        Args:
            pred: Predikovaný 3D obraz
            target: Cílový 3D obraz
            mask: Maska léze
            orig_range: Původní rozsah intenzit (pro škálování)
            
        Returns:
            Vážená perceptuální ztráta
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Check if mask contains any lesions
        mask_sum = binary_mask.sum()
        if mask_sum <= 0:
            # If no lesions, return zero loss
            return torch.tensor(0.0, device=pred.device)
        
        # Aplikujeme masku - počítáme loss pouze v oblasti léze
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Extrahujeme příznaky pouze z oblasti léze
        try:
            # Extrakce příznaků
            pred_features = self.feature_extractor(masked_pred)
            target_features = self.feature_extractor(masked_target)
            
            # Vytvoříme upravenou masku pro feature mapy
            # Původní maska je 16x16x16, feature mapy jsou 10x10x10
            # Proto musíme přizpůsobit masku
            if binary_mask.shape[-1] == 16:  # Pokud je maska 16x16x16
                # Feature mapa je 10x10x10 po třech konvolucích 3x3
                feature_mask = torch.nn.functional.interpolate(
                    binary_mask, 
                    size=(10, 10, 10), 
                    mode='trilinear', 
                    align_corners=False
                )
                feature_mask = (feature_mask > 0.5).float()
            else:
                # Pro jiné velikosti masek - potřebujeme výpočet nové velikosti
                # Po třech vrstvách s kernelem 3x3 bez paddingu se každá dimenze zmenší o 6
                d, h, w = binary_mask.shape[2:]
                new_d, new_h, new_w = d-6, h-6, w-6
                
                # Interpolace na novou velikost
                if new_d > 0 and new_h > 0 and new_w > 0:
                    feature_mask = torch.nn.functional.interpolate(
                        binary_mask, 
                        size=(new_d, new_h, new_w), 
                        mode='trilinear', 
                        align_corners=False
                    )
                    feature_mask = (feature_mask > 0.5).float()
                else:
                    # Pokud jsou rozměry příliš malé, tuto loss nemůžeme použít
                    return torch.tensor(0.0, device=pred.device)
            
            # L1 loss na příznacích vážené maskou
            feature_diff = torch.abs(pred_features - target_features)
            masked_feature_diff = feature_diff * feature_mask
            
            # Průměrujeme pouze přes platné voxely léze v feature mapách
            feature_mask_sum = feature_mask.sum()
            if feature_mask_sum > 0:
                perceptual_loss = masked_feature_diff.sum() / feature_mask_sum
            else:
                perceptual_loss = torch.tensor(0.0, device=pred.device)
            
            # Škálování podle původního rozsahu
            if orig_range is not None:
                orig_min, orig_max = orig_range
                if isinstance(orig_min, torch.Tensor):
                    if orig_min.device != pred.device:
                        orig_min = orig_min.to(pred.device)
                else:
                    orig_min = torch.tensor(orig_min, device=pred.device)
                    
                if isinstance(orig_max, torch.Tensor):
                    if orig_max.device != pred.device:
                        orig_max = orig_max.to(pred.device)
                else:
                    orig_max = torch.tensor(orig_max, device=pred.device)
                
                range_factor = torch.abs(orig_max - orig_min)
                perceptual_loss = perceptual_loss * range_factor
                
                # Zpracování NaN/Inf pro debugging
                if torch.isnan(perceptual_loss).any() or torch.isinf(perceptual_loss).any():
                    print(f"Warning: NaN/Inf in Perceptual loss! Range: {orig_min.item()} to {orig_max.item()}")
                    return torch.tensor(1.0, device=pred.device) * self.weight
            
            # Vracíme váženou ztrátu
            return self.weight * perceptual_loss
            
        except Exception as e:
            print(f"Error in perceptual loss calculation: {e}")
            # V případě chyby vrátíme nulovou ztrátu
            return torch.tensor(0.0, device=pred.device)

class GradientSmoothingLoss(nn.Module):
    def __init__(self, weight=0.02):
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

class DynamicWeightedMAELoss(nn.Module):
    def __init__(self, base_weight=30.0, scaling_factor=8.0):
        super(DynamicWeightedMAELoss, self).__init__()
        self.base_weight = base_weight
        self.scaling_factor = scaling_factor
        
    def forward(self, pred, target, mask, orig_range=None):
        """
        Compute MAE loss STRICTLY in lesion areas only with dynamic weighting based on lesion size.
        Smaller lesions receive higher weights to ensure they're properly inpainted.
        
        Args:
            pred: Predicted tensor
            target: Target tensor  
            mask: Lesion mask tensor
            orig_range: Original intensity range as tuple (min, max) before normalization
            
        Returns:
            Dynamically weighted MAE loss (calculated ONLY in lesion voxels)
        """
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Zjistit relativní velikost léze v patchi
        lesion_voxels = binary_mask.sum()
        total_voxels = torch.numel(binary_mask)
        lesion_ratio = lesion_voxels / total_voxels
        
        # Zkontrolujeme, zda maska obsahuje nějaké léze
        if lesion_voxels <= 0:
            # Pokud nejsou léze, vrátíme nulovou ztrátu
            return torch.tensor(0.0, device=pred.device)
            
        # Dynamicky zvýšit váhu při malém podílu léze (inverzní vztah)
        # Čím menší léze, tím větší váha (ale zachovává minimální váhu base_weight)
        dynamic_weight = self.base_weight * (1.0 + self.scaling_factor * (1.0 - lesion_ratio))
        
        # DŮLEŽITÉ: Aplikujeme masku, aby se počítala ztráta POUZE v oblasti léze
        masked_pred = pred * binary_mask
        masked_target = target * binary_mask
        
        # Calculate absolute difference POUZE v oblasti léze
        abs_diff = torch.abs(masked_pred - masked_target)
        
        # Calculate mean only over masked regions - POUZE v oblasti léze
        mae = abs_diff.sum() / lesion_voxels
        
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
            
            # Scale the loss by the original range for more intuitive results
            range_factor = torch.abs(orig_max - orig_min)
            mae = mae * range_factor
            
            # Pro lepší debugging
            if torch.isnan(mae).any() or torch.isinf(mae).any():
                print(f"Warning: NaN/Inf in Dynamic MAE loss! Range: {orig_min.item()} to {orig_max.item()}")
                return torch.tensor(1.0, device=pred.device) * self.base_weight
        
        # Pro debugování - výpis aktuální váhy
        # print(f"Lesion ratio: {lesion_ratio.item():.4f}, Dynamic weight: {dynamic_weight.item():.2f}")
        
        # Apply dynamic weight
        return dynamic_weight * mae

class WGANLoss:
    """
    Implementace WGAN-GP loss pro stabilní trénink GAN.
    Používá Wasserstein vzdálenost a gradient penalty.
    """
    def __init__(self, lambda_gp=10.0):
        """
        Inicializace WGAN-GP loss.
        
        Args:
            lambda_gp: Váha gradient penalty
        """
        self.lambda_gp = lambda_gp
    
    def discriminator_loss(self, real_validity, fake_validity):
        """
        Wasserstein loss pro diskriminátor.
        
        Args:
            real_validity: Skóre diskriminátoru pro reálné vzorky
            fake_validity: Skóre diskriminátoru pro vygenerované vzorky
            
        Returns:
            Loss pro diskriminátor
        """
        # Kritérium Wasserstein distance: max E[D(real)] - E[D(fake)]
        # Implementováno jako minimalizace: E[D(fake)] - E[D(real)]
        return torch.mean(fake_validity) - torch.mean(real_validity)
    
    def generator_loss(self, fake_validity):
        """
        Wasserstein loss pro generátor.
        
        Args:
            fake_validity: Skóre diskriminátoru pro vygenerované vzorky
            
        Returns:
            Loss pro generátor
        """
        # Kritérium Wasserstein distance: min -E[D(fake)]
        # Implementováno jako minimalizace: -E[D(fake)]
        return -torch.mean(fake_validity)
    
    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples, pseudo_healthy, label):
        """
        Výpočet gradient penalty pro WGAN-GP.
        
        Args:
            discriminator: Diskriminátor model
            real_samples: Reálné ADC mapy
            fake_samples: Vygenerované ADC mapy
            pseudo_healthy: Pseudo-zdravé vstupy
            label: Masky lézí
            
        Returns:
            Gradient penalty term
        """
        # Náhodná váha pro interpolaci mezi reálnými a generovanými vzorky
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=real_samples.device)
        
        # Získáme náhodné body na přímce mezi reálnými a generovanými vzorky
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Vyhodnotíme diskriminátor v interpolovaných bodech
        d_interpolates = discriminator(pseudo_healthy, label, interpolates)
        
        # Automatický výpočet gradientů
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Spočítáme normu gradientu pro každý batch
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # Gradient penalty: (||∇D(x)||_2 - 1)^2
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty * self.lambda_gp

class BiasedMAELoss(nn.Module):
    """
    MAE loss s preferencí nižších hodnot v lézích, založená na medicínské znalosti.
    Penalizuje více případy, kdy předpověď má vyšší hodnoty než okolí (což je proti očekávání).
    """
    def __init__(self, weight=5.0, high_penalty_factor=2.0, neighbor_radius=2):
        super(BiasedMAELoss, self).__init__()
        self.weight = weight
        self.high_penalty_factor = high_penalty_factor  # Faktor, o kolik víc penalizujeme vyšší hodnoty
        self.neighbor_radius = neighbor_radius  # Radius pro výpočet průměrných hodnot okolí
        
    def forward(self, pred, target, mask, orig_range=None):
        """
        Args:
            pred: Predikované hodnoty (B, 1, D, H, W)
            target: Skutečné hodnoty (B, 1, D, H, W)
            mask: Maska léze (B, 1, D, H, W)
            orig_range: Původní rozsah hodnot pro škálování
            
        Returns:
            Weighted biased MAE loss
        """
        # Vytvoření binární masky
        binary_mask = (mask > 0.5).float()
        
        # Ověření, že maska obsahuje nenulové hodnoty
        mask_sum = binary_mask.sum()
        if mask_sum < 1:
            return torch.tensor(0.0, device=pred.device)
        
        # Absolutní rozdíl
        abs_diff = torch.abs(pred - target) * binary_mask
        
        # Výpočet průměrných hodnot okolí pro každý pixel léze
        # Nejprve rozšíříme masku pro zahrnutí okolí
        kernel_size = 2 * self.neighbor_radius + 1
        padding = self.neighbor_radius
        
        # Použijeme průměrový pooling pro výpočet okolních hodnot
        neighbor_values = F.avg_pool3d(
            target * (1 - binary_mask),  # Pouze hodnoty mimo lézi
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # Identifikace míst, kde predikce je vyšší než okolí
        # Což je proti medicínskému očekávání (léze mají nižší ADC)
        higher_than_neighbors = (pred > neighbor_values) * binary_mask
        
        # Aplikace vyšší penalizace na tyto případy
        biased_diff = abs_diff * (1 + (self.high_penalty_factor - 1) * higher_than_neighbors)
        
        # Výpočet celkové loss
        loss = biased_diff.sum() / mask_sum
        
        # Aplikace původního rozsahu hodnot, pokud je k dispozici
        if orig_range is not None:
            orig_min, orig_max = orig_range
            if isinstance(orig_min, torch.Tensor) and isinstance(orig_max, torch.Tensor):
                try:
                    range_width = orig_max.item() - orig_min.item()
                    loss = loss * range_width
                except (RuntimeError, ValueError) as e:
                    if torch.isnan(orig_min).any() or torch.isnan(orig_max).any() or \
                       torch.isinf(orig_min).any() or torch.isinf(orig_max).any():
                        print(f"Warning: NaN/Inf in BiasedMAE loss! Range: {orig_min.item()} to {orig_max.item()}")
                    else:
                        print(f"Error in BiasedMAE loss calculation: {e}")
                    loss = loss  # Use unnormalized loss as fallback
            else:
                range_width = orig_max - orig_min
                loss = loss * range_width
                
        return self.weight * loss

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
    
    # Inicializace diskriminátoru pro CGAN
    discriminator = LesionDiscriminator(in_channels=3).to(device)
    
    # Define loss functions based on the selected loss type
    loss_type = args.loss_type if hasattr(args, 'loss_type') else 'mae'
    print(f"Using loss type: {loss_type}")
    
    # Initialize loss functions - všechny rekonstrukční ztráty mají nižší váhy pro dominanci adversariální složky
    # All reconstruction losses have reduced weights to prioritize the adversarial component
    masked_mae_loss = MaskedMAELoss(weight=5.0).to(device)
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0, weight=5.0).to(device)
    dynamic_mae_loss = DynamicWeightedMAELoss(base_weight=7.5, scaling_factor=8.0).to(device)
    biased_mae_loss = BiasedMAELoss(weight=7.5, high_penalty_factor=3.0, neighbor_radius=2).to(device)
    gradient_loss = GradientSmoothingLoss(weight=0.01).to(device)
    
    # Inicializace perceptuálního lossu - drasticky snížená váha, protože měla příliš vysoké hodnoty
    perceptual_loss = PerceptualLoss(weight=0.05).to(device)
    
    # Inicializace WGAN lossu pro adversariální trénink
    wgan_loss = WGANLoss(lambda_gp=10.0)
    
    # Váha adversariální loss - hlavní komponent GAN architektury
    # Higher weight emphasizes the adversarial aspect of the model
    # Váha 20.0 dává adversariální složce dominantní roli oproti rekonstrukčním ztrátám
    adv_weight = args.adv_weight if hasattr(args, 'adv_weight') else 20.0
    print(f"Adversarial weight: {adv_weight}")
    
    # Initialize optimizers
    optimizer_G = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Initialize patch extractor
    patch_extractor = PatchExtractor(patch_size=(16, 16, 16))
    
    # Define max patches per volume for training - this prevents excessive patch extraction
    max_patches_per_volume = args.max_patches if hasattr(args, 'max_patches') else 32
    max_val_patches_per_volume = 16  # Can use fewer patches for validation
    
    # Training loop
    best_val_lesion_loss = float('inf')
    best_val_lesion_dice = 0.0
    best_val_lesion_mae = float('inf')
    best_val_lesion_ssim = 0.0
    
    # Inicializace start_epoch pro případy, kdy není použit --resume
    args.start_epoch = 0
    
    # Načíst checkpoint, pokud je k dispozici
    if hasattr(args, 'resume') and args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        discriminator.load_state_dict(checkpoint.get('discriminator_state_dict', {}))  # Bude fungovat i pro starší checkpointy
        optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'discriminator_optimizer_state_dict' in checkpoint:
            optimizer_D.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        
        args.start_epoch = checkpoint.get('epoch', 0)
        best_val_lesion_loss = checkpoint.get('best_val_lesion_loss', float('inf'))
        best_val_lesion_mae = checkpoint.get('best_val_lesion_mae', float('inf'))
        best_val_lesion_dice = checkpoint.get('best_val_lesion_dice', 0.0)
        best_val_lesion_ssim = checkpoint.get('best_val_lesion_ssim', 0.0)
        print(f"Resuming from epoch {args.start_epoch}")
    
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        discriminator.train()
        epoch_loss = 0
        epoch_mae_loss = 0
        epoch_focal_loss = 0
        epoch_dynamic_mae_loss = 0
        epoch_biased_mae_loss = 0
        epoch_gradient_loss = 0
        epoch_perceptual_loss = 0
        epoch_adv_loss = 0
        epoch_d_loss = 0
        epoch_g_raw_loss = 0  # Čistá generátor loss bez váhy
        epoch_d_raw_loss = 0  # Diskriminátor loss bez GP
        train_samples = 0
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            pseudo_healthy = batch_data["pseudo_healthy"].to(device)
            adc = batch_data["adc"].to(device)
            label = batch_data["label"].to(device)
            
            # Get the original ranges from the dataset for loss calculation
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
                    # Also ensure the patch actually contains lesions
                    if patches_label[i].sum() > 0:
                        valid_patches.append(i)
            
            # Skip if no valid patches with lesions
            if not valid_patches:
                print(f"Warning: No valid patches with lesions found for training sample {batch_idx}")
                continue
                
            # Limit the number of patches to prevent excessive memory usage and imbalanced training
            if len(valid_patches) > args.max_patches:
                # Use random sampling
                random.seed(epoch * 1000 + batch_idx)  # For reproducibility but different each epoch
                valid_patches = random.sample(valid_patches, args.max_patches)
            
            # Filter patches based on valid_patches
            filtered_patches_ph = [patches_ph[i] for i in valid_patches]
            filtered_patches_label = [patches_label[i] for i in valid_patches]
            filtered_patches_adc = [patches_adc[i] for i in valid_patches]
            filtered_patches_coords = [patch_coords[i] for i in valid_patches]
            
            # Debug info about lesion content
            if len(filtered_patches_label) > 0:
                avg_lesion_voxels = sum(patch.sum().item() for patch in filtered_patches_label) / len(filtered_patches_label)
            
            batch_loss = 0
            for ph_patch, label_patch, adc_patch in zip(filtered_patches_ph, filtered_patches_label, filtered_patches_adc):
                # Add batch dimension
                ph_patch = ph_patch.unsqueeze(0).to(device)
                label_patch = label_patch.unsqueeze(0).to(device)
                adc_patch = adc_patch.unsqueeze(0).to(device)
                
                # ------
                # KROK 1: Trénink diskriminátoru (WGAN-GP)
                # ------
                for _ in range(1):  # Train discriminator once per generator update
                    optimizer_D.zero_grad()
                    
                    # Generovat fake výstup
                    with torch.no_grad():
                        fake_adc = model(ph_patch, label_patch)
                    
                    # Spočítat validitu pro real a fake
                    real_validity = discriminator(ph_patch, label_patch, adc_patch)
                    fake_validity = discriminator(ph_patch, label_patch, fake_adc.detach())
                    
                    # Wasserstein loss pro diskriminátor
                    d_loss = wgan_loss.discriminator_loss(real_validity, fake_validity)
                    
                    # Gradient penalty
                    gp = wgan_loss.compute_gradient_penalty(
                        discriminator, adc_patch, fake_adc.detach(), ph_patch, label_patch
                    )
                    
                    # Celková loss pro diskriminátor
                    d_total_loss = d_loss + gp
                    
                    # Backpropagation pro diskriminátor
                    d_total_loss.backward()
                    optimizer_D.step()
                
                # ------
                # KROK 2: Trénink generátoru
                # ------
                optimizer_G.zero_grad()
                
                # Generovat fake výstup
                output = model(ph_patch, label_patch)
                
                # Adversariální loss - WGAN
                fake_validity = discriminator(ph_patch, label_patch, output)
                g_adv_loss = wgan_loss.generator_loss(fake_validity)
                
                # Calculate losses based on selected loss type
                if loss_type == 'mae':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    focal_value = torch.tensor(0.0, device=device)  # Not used
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = mae_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                    
                elif loss_type == 'focal':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)  # For tracking
                    focal_value = focal_loss(output, adc_patch, label_patch, adc_orig_range)
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = focal_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                    
                elif loss_type == 'dynamic_mae':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)  # For tracking
                    focal_value = torch.tensor(0.0, device=device)  # Not used
                    dynamic_mae_value = dynamic_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = dynamic_mae_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                    
                elif loss_type == 'combined':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    focal_value = focal_loss(output, adc_patch, label_patch, adc_orig_range)
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = mae_value + focal_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                    
                elif loss_type == 'perceptual':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)  # For tracking
                    focal_value = torch.tensor(0.0, device=device)  # Not used
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = perceptual_value + gradient_value
                    loss = recon_loss + adv_weight * g_adv_loss
                    
                elif loss_type == 'biased_mae':
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)  # For tracking
                    focal_value = torch.tensor(0.0, device=device)  # Not used
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    biased_mae_value = biased_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = biased_mae_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                
                else:  # Default to MAE
                    mae_value = masked_mae_loss(output, adc_patch, label_patch, adc_orig_range)
                    focal_value = torch.tensor(0.0, device=device)  # Not used
                    dynamic_mae_value = torch.tensor(0.0, device=device)  # Not used
                    gradient_value = gradient_loss(output, label_patch)
                    perceptual_value = perceptual_loss(output, adc_patch, label_patch, adc_orig_range)
                    
                    # Kombinace rekonstrukční a adversariální loss
                    recon_loss = mae_value + gradient_value + perceptual_value
                    loss = recon_loss + adv_weight * g_adv_loss
                
                # Backpropagation pro generátor
                loss.backward()
                optimizer_G.step()
                
                # Přidání hodnot lossu pro monitoring
                batch_loss += loss.item()
                epoch_loss += loss.item()
                epoch_mae_loss += mae_value.item() if loss_type in ['mae', 'combined'] else 0
                epoch_focal_loss += focal_value.item() if loss_type in ['focal', 'combined'] else 0
                epoch_dynamic_mae_loss += dynamic_mae_value.item() if loss_type == 'dynamic_mae' else 0
                epoch_biased_mae_loss += biased_mae_value.item() if loss_type == 'biased_mae' else 0
                epoch_gradient_loss += gradient_value.item()
                epoch_perceptual_loss += perceptual_value.item()
                epoch_adv_loss += g_adv_loss.item()  # Přidání adversariální ztráty pro monitoring
                epoch_d_loss += d_total_loss.item()  # Přidání diskriminátor ztráty pro monitoring
                epoch_g_raw_loss += g_adv_loss.item()  # Čistá WGAN generátor loss
                epoch_d_raw_loss += d_loss.item()  # Čistá WGAN diskriminátor loss
                train_samples += 1
            
            # Average loss over patches
            if len(filtered_patches_ph) > 0:
                avg_batch_loss = batch_loss / len(filtered_patches_ph)
                epoch_loss += avg_batch_loss
                
        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        avg_epoch_mae_loss = epoch_mae_loss / max(1, train_samples) if loss_type not in ['focal', 'dynamic_mae'] else 0
        avg_epoch_focal_loss = epoch_focal_loss / max(1, train_samples) if loss_type not in ['mae', 'dynamic_mae'] else 0
        avg_epoch_dynamic_mae_loss = epoch_dynamic_mae_loss / max(1, train_samples) if loss_type == 'dynamic_mae' else 0
        avg_epoch_biased_mae_loss = epoch_biased_mae_loss / max(1, train_samples) if loss_type == 'biased_mae' else 0
        avg_epoch_gradient_loss = epoch_gradient_loss / max(1, train_samples)
        avg_epoch_perceptual_loss = epoch_perceptual_loss / max(1, train_samples)
        avg_epoch_adv_loss = epoch_adv_loss / max(1, train_samples)  # Průměrná adversariální ztráta
        avg_epoch_d_loss = epoch_d_loss / max(1, train_samples)  # Průměrná diskriminátor ztráta
        avg_epoch_g_raw_loss = epoch_g_raw_loss / max(1, train_samples)  # Průměrná čistá generátor loss
        avg_epoch_d_raw_loss = epoch_d_raw_loss / max(1, train_samples)  # Průměrná čistá diskriminátor loss
        
        # Get actual values by dividing by the weight for display purposes only
        actual_mae = avg_epoch_mae_loss / 5.0 if loss_type not in ['focal', 'dynamic_mae'] else 0
        actual_focal = avg_epoch_focal_loss / 5.0 if loss_type not in ['mae', 'dynamic_mae'] else 0
        actual_dynamic_mae = avg_epoch_dynamic_mae_loss / 7.5 if loss_type == 'dynamic_mae' else 0
        actual_biased_mae = avg_epoch_biased_mae_loss / 7.5 if loss_type == 'biased_mae' else 0
        
        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")
        if loss_type not in ['focal', 'dynamic_mae', 'biased_mae']:
            print(f"  - MAE: {actual_mae:.4f} (unweighted)")
        if loss_type not in ['mae', 'dynamic_mae', 'biased_mae']:
            print(f"  - Focal: {actual_focal:.4f} (unweighted)")
        if loss_type == 'dynamic_mae':
            print(f"  - Dynamic MAE: {actual_dynamic_mae:.4f} (unweighted)")
        if loss_type == 'biased_mae':
            print(f"  - Biased MAE: {actual_biased_mae:.4f} (unweighted)")
        print(f"  - Gradient Loss: {avg_epoch_gradient_loss:.4f}")
        print(f"  - Perceptual Loss: {avg_epoch_perceptual_loss:.4f}")
        
        # Výpis GAN komponent
        print(f"\n  ===== GAN PERFORMANCE =====")
        print(f"  - Generator Loss (weighted): {adv_weight * avg_epoch_g_raw_loss:.4f}")
        print(f"  - Discriminator Loss (total): {avg_epoch_d_loss:.4f}")
        print(f"  -----------------------")
        print(f"  - Generator WGAN Raw:   {avg_epoch_g_raw_loss:.4f}")
        print(f"  - Discriminator WGAN Raw: {avg_epoch_d_raw_loss:.4f}")
        print(f"  ===========================\n")
        
        # Validation
        model.eval()
        discriminator.eval()
        val_loss = 0.0
        val_lesion_mae = 0.0
        val_lesion_focal = 0.0
        val_lesion_dynamic_mae = 0.0
        val_lesion_biased_mae = 0.0
        val_lesion_ssim = 0.0
        val_lesion_dice = 0.0
        val_lesion_perceptual = 0.0
        val_examples = 0
        
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
                    
                    # Filter out patches with invalid coordinates (negative values)
                    valid_patches = []
                    for i, (patch, coords) in enumerate(zip(patches_ph, patch_coords)):
                        if all(c >= 0 for c in coords):
                            # Only include patches that contain lesions
                            if patches_label[i].sum() > 0:
                                valid_patches.append(i)
                    
                    if not valid_patches:
                        print(f"Warning: No valid patches with lesions found for validation sample {val_batch_idx}")
                        continue
                        
                    # Limit the number of patches to prevent excessive validation on large lesions
                    total_valid_patches = len(valid_patches)
                    max_val_patches = 16  # Lower limit for validation to speed it up
                    if total_valid_patches > max_val_patches:
                        # Use deterministic sampling for validation (fixed seed for reproducibility)
                        seed = 42 + val_batch_idx
                        random.seed(seed)
                        valid_patches = random.sample(valid_patches, max_val_patches)
                    
                    # Filter patches based on valid_patches list
                    filtered_patches_ph = [patches_ph[i] for i in valid_patches]
                    filtered_patches_label = [patches_label[i] for i in valid_patches]
                    filtered_patches_adc = [patches_adc[i] for i in valid_patches]
                    filtered_patch_coords = [patch_coords[i] for i in valid_patches]
                    
                    # Debug info about lesion content
                    if len(filtered_patches_label) > 0:
                        avg_lesion_voxels = sum(patch.sum().item() for patch in filtered_patches_label) / len(filtered_patches_label)
                    
                    # Convert patches to tensors and move to device
                    filtered_patches_ph = [p.to(device) for p in filtered_patches_ph]
                    filtered_patches_label = [p.to(device) for p in filtered_patches_label]
                    filtered_patches_adc = [p.to(device) for p in filtered_patches_adc]
                    # DO NOT move patch coordinates to device - they are tuples not tensors
                    # filtered_patch_coords are left as tuples of integers
                    
                    # Process patches one by one
                    output_patches = []
                    valid_patch_coords = []
                    
                    # Collect metrics for this batch
                    batch_mae = 0.0
                    batch_ssim = 0.0
                    batch_dice = 0.0
                    batch_focal = 0.0
                    batch_dynamic_mae = 0.0
                    batch_biased_mae = 0.0
                    batch_perceptual = 0.0
                    num_valid_patches = 0
                    
                    for patch_idx, (ph_patch, label_patch, adc_patch, coords) in enumerate(zip(
                            filtered_patches_ph, filtered_patches_label, 
                            filtered_patches_adc, filtered_patch_coords)):
                        try:
                            # Add batch dimension
                            ph_patch = ph_patch.unsqueeze(0)  
                            label_patch = label_patch.unsqueeze(0)
                            adc_patch = adc_patch.unsqueeze(0)
                            
                            # Generate inpainted output
                            output_patch = model(ph_patch, label_patch)
                            
                            # Check output shape 
                            expected_shape = ph_patch.shape[2:]
                            output_shape = output_patch.shape[2:]
                            
                            if expected_shape != output_shape:
                                print(f"Warning: Output patch dimensions {output_shape} don't match expected dimensions {expected_shape}")
                                # Try to fix dimensions using interpolation
                                if len(output_shape) == len(expected_shape):
                                    try:
                                        # Use interpolate to resize to expected dimensions
                                        output_patch = torch.nn.functional.interpolate(
                                            output_patch,
                                            size=expected_shape,
                                            mode='trilinear',
                                            align_corners=False
                                        )
                                    except Exception as resize_err:
                                        print(f"Error resizing patch: {resize_err}")
                                        continue
                                else:
                                    # Skip this patch if dimensions don't match and can't be fixed
                                    continue
                            
                            # Calculate metrics for this patch
                            binary_mask = (label_patch > 0.5).float()
                            
                            # Always calculate MAE for validation regardless of loss type
                            lesion_voxels = binary_mask.sum()
                            if lesion_voxels > 0:
                                # Calculate MAE only in lesion areas (masked)
                                masked_pred = output_patch * binary_mask
                                masked_target = adc_patch * binary_mask
                                abs_diff = torch.abs(masked_pred - masked_target)
                                mae_patch = abs_diff.sum() / lesion_voxels  # Average only over lesion voxels
                                
                                # Apply scaling if original range is provided
                                if adc_orig_range is not None:
                                    mae_patch = mae_patch * (adc_orig_range[1] - adc_orig_range[0])
                            else:
                                mae_patch = torch.tensor(0.0, device=device)
                            
                            # Pro SSIM metriku - pouze pro zpětnou kompatibilitu
                            if binary_mask.sum() > 0:
                                try:
                                    from pytorch_msssim import ssim
                                    # Upravit dimenze pro SSIM, který očekává 4D nebo 5D tenzor
                                    if output_patch.dim() == 5:  # [B,C,D,H,W]
                                        # Pro 3D data vezmeme průměr přes D dimenzi
                                        ssim_val = 0
                                        depth = output_patch.shape[2]
                                        for d in range(depth):
                                            # Použijeme ssim pro každý 2D řez
                                            slice_ssim = ssim(
                                                output_patch[:,:,d],
                                                adc_patch[:,:,d],
                                                data_range=1.0,
                                                size_average=True
                                            )
                                            ssim_val += slice_ssim
                                        ssim_val /= depth  # Průměr přes všechny řezy
                                    else:
                                        # Pokud není 5D, nastavíme defaultní hodnotu
                                        ssim_val = 0.5
                                        
                                    ssim_patch_val = ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val
                                except:
                                    # Pokud pytorch_msssim není k dispozici, použijeme placeholder
                                    ssim_patch_val = 0.5
                            else:
                                ssim_patch_val = 0.0
                            
                            # Calculate Dice coefficient 
                            if binary_mask.sum() > 0:
                                masked_pred = output_patch * binary_mask
                                masked_target = adc_patch * binary_mask
                                
                                # Normalize values for thresholding
                                norm_pred = (masked_pred - masked_pred.min()) / (masked_pred.max() - masked_pred.min() + 1e-8)
                                norm_target = (masked_target - masked_target.min()) / (masked_target.max() - masked_target.min() + 1e-8)
                                
                                # Apply threshold (0.5) to get binary masks
                                pred_binary = (norm_pred > 0.5).float()
                                target_binary = (norm_target > 0.5).float()
                                
                                # Calculate Dice
                                intersection = (pred_binary * target_binary * binary_mask).sum()
                                dice_patch = (2. * intersection) / (
                                    (pred_binary * binary_mask).sum() + (target_binary * binary_mask).sum() + 1e-8
                                )
                            else:
                                dice_patch = torch.tensor(0.0, device=device)
                            
                            # Calculate Focal Loss for validation if using focal loss
                            if loss_type in ['focal', 'combined']:
                                try:
                                    focal_patch = focal_loss(output_patch, adc_patch, label_patch, adc_orig_range).item() / 25.0  # Divide by weight for raw value
                                except Exception as e:
                                    print(f"Error calculating focal loss: {e}")
                                    focal_patch = 0.0
                            else:
                                focal_patch = 0.0
                            
                            # Calculate Dynamic MAE for validation if using dynamic_mae loss
                            if loss_type in ['dynamic_mae']:
                                try:
                                    dynamic_mae_patch = dynamic_mae_loss(output_patch, adc_patch, label_patch, adc_orig_range).item() / 7.5  # Divide by base_weight for raw value
                                except Exception as e:
                                    print(f"Error calculating dynamic mae loss: {e}")
                                    dynamic_mae_patch = 0.0
                            else:
                                dynamic_mae_patch = 0.0
                                
                            # Calculate Biased MAE for validation if using biased_mae loss
                            if loss_type in ['biased_mae']:
                                try:
                                    biased_mae_patch = biased_mae_loss(output_patch, adc_patch, label_patch, adc_orig_range).item() / 7.5  # Divide by weight for raw value
                                except Exception as e:
                                    print(f"Error calculating biased mae loss: {e}")
                                    biased_mae_patch = 0.0
                            else:
                                biased_mae_patch = 0.0
                            
                            # Calculate Perceptual Loss for validation if using perceptual loss
                            if loss_type in ['combined', 'perceptual']:
                                try:
                                    perceptual_patch = perceptual_loss(output_patch, adc_patch, label_patch, adc_orig_range).item() / 0.05  # Divide by weight for raw value
                                except Exception as e:
                                    print(f"Error calculating perceptual loss: {e}")
                                    perceptual_patch = 0.0
                            else:
                                perceptual_patch = 0.0
                            
                            # Add metrics to batch totals
                            # Always add MAE regardless of loss type for validation (for comparison)
                            batch_mae += mae_patch.item()
                            batch_ssim += ssim_patch_val
                            batch_dice += dice_patch.item()
                            if loss_type in ['focal', 'combined']:
                                batch_focal += focal_patch
                            if loss_type in ['dynamic_mae']:
                                batch_dynamic_mae += dynamic_mae_patch
                            if loss_type in ['biased_mae']:
                                batch_biased_mae += biased_mae_patch
                            if loss_type in ['combined', 'perceptual']:
                                batch_perceptual += perceptual_patch
                            num_valid_patches += 1
                            
                            # Store patch for reconstruction
                            output_patches.append(output_patch[0])
                            valid_patch_coords.append(coords)
                            
                        except Exception as patch_error:
                            print(f"Error processing patch {patch_idx}: {patch_error}")
                            continue
                    
                    # Average metrics for this batch
                    if num_valid_patches > 0:
                        # Always calculate average MAE for validation
                        batch_mae /= num_valid_patches
                        batch_ssim /= num_valid_patches
                        batch_dice /= num_valid_patches
                        if loss_type in ['focal', 'combined']:
                            batch_focal /= num_valid_patches
                        if loss_type in ['dynamic_mae']:
                            batch_dynamic_mae /= num_valid_patches
                        if loss_type in ['biased_mae']:
                            batch_biased_mae /= num_valid_patches
                        if loss_type in ['combined', 'perceptual']:
                            batch_perceptual /= num_valid_patches
                        
                        # Add to validation totals
                        # Always add MAE to validation totals for comparison
                        val_lesion_mae += batch_mae 
                        val_lesion_ssim += batch_ssim
                        val_lesion_dice += batch_dice
                        if loss_type in ['focal', 'combined']:
                            val_lesion_focal += batch_focal
                        if loss_type in ['dynamic_mae']:
                            val_lesion_dynamic_mae += batch_dynamic_mae
                        if loss_type in ['biased_mae']:
                            val_lesion_biased_mae += batch_biased_mae
                        if loss_type in ['combined', 'perceptual']:
                            val_lesion_perceptual += batch_perceptual
                        val_examples += 1
                        
                        # Calculate total validation loss based on loss_type
                        if loss_type == 'mae':
                            batch_val_loss = batch_mae
                        elif loss_type == 'focal':
                            batch_val_loss = batch_focal
                        elif loss_type == 'dynamic_mae':
                            batch_val_loss = batch_dynamic_mae
                        elif loss_type == 'combined':
                            batch_val_loss = batch_mae * 0.5 + batch_focal * 0.5 + batch_perceptual * 0.5
                        elif loss_type == 'perceptual':
                            batch_val_loss = batch_perceptual
                        else:  # Default to MAE
                            batch_val_loss = batch_mae
                            
                        val_loss += batch_val_loss
                    
                    # Reconstruct full volume
                    if output_patches:
                        try:
                            # Ensure all patches have the same number of dimensions
                            patch_dims = [p.dim() for p in output_patches]
                            if len(set(patch_dims)) > 1:
                                print(f"Warning: Inconsistent patch dimensions: {patch_dims}")
                                # Try to normalize dimensions
                                for i in range(len(output_patches)):
                                    while output_patches[i].dim() < 4:  # Ensure 4D [C, Z, Y, X]
                                        output_patches[i] = output_patches[i].unsqueeze(0)
                                    while output_patches[i].dim() > 4:
                                        output_patches[i] = output_patches[i].squeeze(0)
                            
                            # Reconstruct validation volume
                            reconstructed = patch_extractor.reconstruct_from_patches(
                                output_patches, filtered_patch_coords, pseudo_healthy.shape
                            )
                            
                            # Apply mask to show inpainted regions
                            # Rest of reconstruction code...
                            
                        except Exception as e:
                            print(f"Error during validation reconstruction: {e}")
                except Exception as batch_error:
                    print(f"Error processing validation batch {val_batch_idx}: {batch_error}")
                    continue
        
        if val_examples > 0:
            # Calculate average metrics
            avg_val_loss = val_loss / val_examples
            
            # Always calculate MAE for validation regardless of loss type
            avg_val_lesion_mae = val_lesion_mae / val_examples
            # Divide by the weight factor (25.0) to match training MAE display
            avg_val_lesion_mae = avg_val_lesion_mae
                
            if loss_type in ['focal', 'combined']:
                avg_val_lesion_focal = val_lesion_focal / val_examples
            else:
                # Set to zero if not using this metric
                avg_val_lesion_focal = 0.0
                
            if loss_type in ['dynamic_mae']:
                avg_val_lesion_dynamic_mae = val_lesion_dynamic_mae / val_examples
            else:
                # Set to zero if not using this metric
                avg_val_lesion_dynamic_mae = 0.0
                
            avg_val_lesion_ssim = val_lesion_ssim / val_examples
            avg_val_lesion_dice = val_lesion_dice / val_examples
            avg_val_lesion_perceptual = val_lesion_perceptual / val_examples
            
            # Divide overall loss by the weight factor (25.0) to match training display
            avg_val_loss = avg_val_loss / 25.0
            
            # Print validation metrics summary
            print("\nValidation Results:")
            
            # Make overall loss label more specific based on loss type but always show MAE
            if loss_type == 'mae':
                print(f"Overall MAE Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f}")
            elif loss_type == 'focal':
                print(f"Overall Focal Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f} (for comparison)")
                print(f"Lesion Focal: {avg_val_lesion_focal:.4f}")
            elif loss_type == 'dynamic_mae':
                print(f"Overall Dynamic MAE Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f} (for comparison)")
                print(f"Lesion Dynamic MAE: {avg_val_lesion_dynamic_mae:.4f}")
            elif loss_type == 'combined':
                print(f"Overall Combined Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f}")
                print(f"Lesion Focal: {avg_val_lesion_focal:.4f}")
            elif loss_type == 'perceptual':
                print(f"Overall Perceptual Loss: {avg_val_loss:.4f}")
                print(f"Lesion Perceptual: {avg_val_lesion_perceptual:.4f}")
            else:
                print(f"Overall Loss: {avg_val_loss:.4f}")
                print(f"Lesion MAE: {avg_val_lesion_mae:.4f}")
                
            print(f"Lesion Dice: {avg_val_lesion_dice:.4f}")
            print(f"Lesion SSIM: {avg_val_lesion_ssim:.4f} (informační metrika, neoptimalizováno)")
            
            # Generate and save validation visualizations based on vis_freq
            if (epoch + 1) % args.vis_freq == 0:
                # Sample a few validation cases for visualization
                vis_indices = random.sample(range(len(val_loader)), min(3, len(val_loader)))
                
                for vis_idx in vis_indices:
                    try:
                        # Get a sample from validation set
                        vis_data = list(val_loader)[vis_idx]
                        
                        vis_ph = vis_data["pseudo_healthy"].to(device)
                        vis_adc = vis_data["adc"].to(device)
                        vis_label = vis_data["label"].to(device)
                        
                        # Generate reconstruction for visualization
                        with torch.no_grad():
                            # Extract patches
                            vis_patches_ph, vis_patches_label, vis_patches_adc, vis_patch_coords = patch_extractor.extract_patches_with_lesions(
                                vis_ph[0], vis_label[0], vis_adc[0]
                            )
                            
                            # Filter and limit patches for visualization
                            valid_vis_patches = []
                            for i, (patch, coords) in enumerate(zip(vis_patches_ph, vis_patch_coords)):
                                if all(c >= 0 for c in coords) and vis_patches_label[i].sum() > 0:
                                    valid_vis_patches.append(i)
                            
                            # Limit number of visualization patches - use even fewer to keep visualization manageable
                            max_vis_patches = 8
                            if len(valid_vis_patches) > max_vis_patches:
                                # For visualization, use a deterministic sample
                                random.seed(vis_idx * 10000)  # Fixed seed for reproducibility
                                valid_vis_patches = random.sample(valid_vis_patches, max_vis_patches)
                            
                            # Only use patches that contain lesions
                            vis_out_patches = []
                            filtered_vis_coords = []
                            
                            for p_idx in valid_vis_patches:
                                # Add batch dimension
                                p_ph = vis_patches_ph[p_idx].unsqueeze(0).to(device)
                                p_label = vis_patches_label[p_idx].unsqueeze(0).to(device)
                                
                                # Generate output
                                p_out = model(p_ph, p_label)
                                vis_out_patches.append(p_out[0])
                                filtered_vis_coords.append(vis_patch_coords[p_idx])
                        
                            # Skip if no patches with lesions
                            if len(vis_out_patches) == 0:
                                continue
                                
                            # Reconstruct full volume
                            reconstructed = patch_extractor.reconstruct_from_patches(
                                vis_out_patches, filtered_vis_coords, vis_ph.shape
                            )
                            
                            # Apply the lesion mask to only modify lesion regions
                            # Oprava: Zajistíme, že tenzor má správné rozměry pro max_pool3d (5D tenzor [B,C,D,H,W])
                            # Zkontrolujeme rozměry
                            if vis_label.dim() == 4:  # [C,D,H,W]
                                vis_label_5d = vis_label.unsqueeze(0)  # Přidáme batch dimenzi -> [1,C,D,H,W]
                            elif vis_label.dim() == 5:  # [B,C,D,H,W]
                                vis_label_5d = vis_label
                            else:
                                raise ValueError(f"Neočekávaný rozměr vis_label: {vis_label.shape}")
                                
                            dilated_mask = torch.nn.functional.max_pool3d(
                                vis_label_5d, kernel_size=3, stride=1, padding=1
                            )
                            
                            # Upravíme také tvar rekonstruovaného výstupu a pseudozdravého vstupu
                            if reconstructed.dim() < 5:
                                reconstructed = reconstructed.unsqueeze(0)
                            
                            if vis_ph.dim() == 4:  # [C,D,H,W]
                                vis_ph_5d = vis_ph.unsqueeze(0)  # [1,C,D,H,W]
                            elif vis_ph.dim() == 5:  # [B,C,D,H,W]
                                vis_ph_5d = vis_ph
                            else:
                                raise ValueError(f"Neočekávaný rozměr vis_ph: {vis_ph.shape}")
                            
                            # Create final output - copy pseudo-healthy and replace only in lesion regions
                            final_output = vis_ph_5d.clone()
                            final_output = final_output * (1 - dilated_mask) + reconstructed * dilated_mask
                            
                            # Get the original ranges for metrics
                            adc_orig_range_str = None
                            if "original_ranges" in vis_data and "adc" in vis_data["original_ranges"]:
                                orig_range = vis_data["original_ranges"]["adc"]
                                adc_orig_range_str = f"{float(orig_range[0]):.4f} to {float(orig_range[1]):.4f}"
                            
                            # Create metrics dictionary for visualization
                            vis_metrics = {
                                "Epoch": epoch + 1,
                                "Dataset": "Validation",
                                "Loss Type": loss_type,
                                "Sample": f"{vis_idx+1}/{len(val_loader)}"
                            }
                            
                            # Always add MAE to metrics regardless of loss type for comparison
                            vis_metrics["Lesion MAE"] = avg_val_lesion_mae
                                
                            if loss_type in ['focal', 'combined']:
                                vis_metrics["Lesion Focal"] = avg_val_lesion_focal
                                
                            # Add perceptual loss to visualization metrics
                            if loss_type in ['perceptual', 'combined']:
                                vis_metrics["Lesion Perceptual"] = avg_val_lesion_perceptual
                                
                            # Add remaining metrics
                            vis_metrics["Lesion Dice"] = avg_val_lesion_dice
                            vis_metrics["Lesion SSIM (info)"] = avg_val_lesion_ssim
                            vis_metrics["Original ADC Range"] = adc_orig_range_str if adc_orig_range_str else "Not available"
                            
                            # Save visualization
                            vis_output_file = os.path.join(
                                args.output_dir, 
                                f"val_visualization_epoch{epoch+1}_sample{vis_idx+1}.pdf"
                            )
                            
                            visualize_results(
                                vis_ph, 
                                vis_label, 
                                final_output.squeeze(0) if final_output.dim() > 4 else final_output, 
                                vis_adc, 
                                vis_output_file,
                                metrics=vis_metrics
                            )
                            
                            print(f"Saved validation visualization to {vis_output_file}")
                    except Exception as vis_error:
                        print(f"Error creating visualization for validation sample {vis_idx}: {vis_error}")
            
            # Save best models based on metrics
            # Track the best value for each metric type
            best_val_lesion_focal = getattr(train, 'best_val_lesion_focal', float('inf')) if loss_type in ['focal', 'combined'] else float('inf')
            
            # Prioritize metric based on loss type
            if loss_type == 'mae':
                # Prioritize MAE (lower is better)
                if avg_val_lesion_mae < best_val_lesion_mae:
                    best_val_lesion_mae = avg_val_lesion_mae
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lesion_mae_model.pth"))
                    print(f"Saved new best model with lesion MAE: {best_val_lesion_mae:.4f}")
                    
            elif loss_type == 'focal':
                # Prioritize Focal Loss (lower is better)
                if avg_val_lesion_focal < best_val_lesion_focal:
                    best_val_lesion_focal = avg_val_lesion_focal
                    train.best_val_lesion_focal = best_val_lesion_focal  # Store for future comparisons
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lesion_focal_model.pth"))
                    print(f"Saved new best model with lesion Focal: {best_val_lesion_focal:.4f}")
                    
            elif loss_type == 'combined':
                # Save both metrics
                if avg_val_lesion_mae < best_val_lesion_mae:
                    best_val_lesion_mae = avg_val_lesion_mae
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lesion_mae_model.pth"))
                    print(f"Saved new best model with lesion MAE: {best_val_lesion_mae:.4f}")
                    
                if avg_val_lesion_focal < best_val_lesion_focal:
                    best_val_lesion_focal = avg_val_lesion_focal
                    train.best_val_lesion_focal = best_val_lesion_focal  # Store for future comparisons
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lesion_focal_model.pth"))
                    print(f"Saved new best model with lesion Focal: {best_val_lesion_focal:.4f}")
            
            # Always save Dice as a secondary metric (higher is better)
            if avg_val_lesion_dice > best_val_lesion_dice:
                best_val_lesion_dice = avg_val_lesion_dice
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_lesion_dice_model.pth"))
                print(f"Saved new best model with lesion Dice: {best_val_lesion_dice:.4f}")
            
            # SSIM is just informational now
            if avg_val_lesion_ssim > best_val_lesion_ssim:
                best_val_lesion_ssim = avg_val_lesion_ssim
                # Save with experimental suffix
                torch.save(model.state_dict(), os.path.join(args.output_dir, "experimental_lesion_ssim_model.pth"))
                print(f"Saved experimental model with lesion SSIM: {best_val_lesion_ssim:.4f} (neoptimalizovaná metrika)")
            
            # Save checkpoint with all metrics
            if (epoch + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'loss': avg_epoch_loss,
                    'mae_loss': avg_epoch_mae_loss,
                    'focal_loss': avg_epoch_focal_loss,
                    'gradient_loss': avg_epoch_gradient_loss,
                    'perceptual_loss': avg_epoch_perceptual_loss,
                    'val_loss': avg_val_loss,
                    'val_lesion_mae': avg_val_lesion_mae,
                    'val_lesion_focal': avg_val_lesion_focal,
                    'val_lesion_ssim': avg_val_lesion_ssim,  # Still save for tracking
                    'val_lesion_dice': avg_val_lesion_dice
                }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        else:
            print("Warning: No valid validation examples found")

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
    patch_extractor = PatchExtractor(patch_size=(16, 16, 16))
    
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
        
        # Filter out patches with invalid coordinates (negative values)
        valid_patches = []
        for i, (patch, coords) in enumerate(zip(patches_ph, patch_coords)):
            if all(c >= 0 for c in coords):
                # Also ensure the patch actually contains lesions
                if patches_label[i].sum() > 0:
                    valid_patches.append(i)
        
        # Skip if no valid patches with lesions
        if not valid_patches:
            print(f"Warning: No valid patches with lesions found in the input image")
            return
            
        # Limit the number of patches to prevent excessive memory usage
        max_inference_patches = 32  # You can adjust this or make it a parameter
        if len(valid_patches) > max_inference_patches:
            # Use deterministic sampling for inference
            random.seed(42)  # Fixed seed for reproducibility
            valid_patches = random.sample(valid_patches, max_inference_patches)
            print(f"Limiting from {len(patches_ph)} to {max_inference_patches} patches for inference")
        
        # Filter patches based on valid_patches list
        filtered_patches_ph = [patches_ph[i] for i in valid_patches]
        filtered_patches_label = [patches_label[i] for i in valid_patches]
        filtered_patch_coords = [patch_coords[i] for i in valid_patches]
        
        # Debug info about lesion content
        if len(filtered_patches_label) > 0:
            avg_lesion_voxels = sum(patch.sum().item() for patch in filtered_patches_label) / len(filtered_patches_label)
            print(f"Inference: {len(filtered_patches_label)} patches, avg lesion voxels per patch: {avg_lesion_voxels:.1f}")
        
        output_patches = []
        for ph_patch, label_patch in zip(filtered_patches_ph, filtered_patches_label):
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
                output_patches, filtered_patch_coords, pseudo_healthy.shape
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
    # Create argument parser
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
    train_parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch number")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    train_parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint frequency")
    train_parser.add_argument("--vis_freq", type=int, default=1, help="Visualization frequency")
    train_parser.add_argument("--loss_type", type=str, default="mae", 
                        choices=["mae", "focal", "combined", "dynamic_mae", "perceptual", "biased_mae"], 
                        help="Loss type (mae, focal, combined, dynamic_mae, perceptual, biased_mae)")
    train_parser.add_argument("--max_patches", type=int, default=32, help="Maximum number of patches per volume")
    train_parser.add_argument("--adv_weight", type=float, default=20.0, help="Adversarial weight for WGAN-GP")
    train_parser.add_argument("--resume", type=str, help="Path to checkpoint for resuming training")
    
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
