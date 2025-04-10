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
    LoadImaged,
    ScaleIntensityd,
    RandRotated,
    RandAffined,
    RandScaleIntensityd,
    EnsureTyped,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd
)
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import SimpleITK as sitk
import glob
import random
from tqdm import tqdm

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
                    "pseudo_healthy": ph_file,
                    "adc": adc_file,
                    "label": label_file
                })
        
        print(f"{'Training' if train else 'Validation'} dataset contains {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        
        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict

class PatchExtractor:
    def __init__(self, patch_size=(64, 64, 32), overlap=0.5):
        self.patch_size = patch_size
        self.overlap = overlap
    
    def extract_patches_with_lesions(self, pseudo_healthy, label, adc):
        """
        Extract patches that contain lesions
        
        Args:
            pseudo_healthy: tensor of shape [C, D, H, W]
            label: tensor of shape [C, D, H, W]
            adc: tensor of shape [C, D, H, W]
            
        Returns:
            patches_ph: list of pseudo_healthy patches
            patches_label: list of label patches
            patches_adc: list of adc patches
            patch_coords: list of patch coordinates (for reconstruction)
        """
        # Get non-zero indices from the label
        non_zero_indices = torch.nonzero(label[0])
        
        if len(non_zero_indices) == 0:
            # If no lesion, return a single random patch
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
        
        # Cluster lesion voxels to find lesion centers
        centers = self._cluster_lesion_voxels(non_zero_indices)
        
        patches_ph, patches_label, patches_adc, patch_coords = [], [], [], []
        
        # Extract patches around each lesion center
        for center in centers:
            z, y, x = center
            
            # Calculate patch start coordinates
            z_start = max(0, z - self.patch_size[0] // 2)
            y_start = max(0, y - self.patch_size[1] // 2)
            x_start = max(0, x - self.patch_size[2] // 2)
            
            # Adjust if patch exceeds volume boundaries
            z_start = min(z_start, pseudo_healthy.shape[1] - self.patch_size[0])
            y_start = min(y_start, pseudo_healthy.shape[2] - self.patch_size[1])
            x_start = min(x_start, pseudo_healthy.shape[3] - self.patch_size[2])
            
            # Extract patches
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
            
            # Only add patch if it contains lesion
            if label_patch.sum() > 0:
                patches_ph.append(ph_patch)
                patches_label.append(label_patch)
                patches_adc.append(adc_patch)
                patch_coords.append((z_start, y_start, x_start))
        
        # If no patches with lesions were found, take a random patch
        if len(patches_ph) == 0:
            return self.extract_patches_with_lesions(pseudo_healthy, label, adc)
        
        return patches_ph, patches_label, patches_adc, patch_coords
    
    def _cluster_lesion_voxels(self, non_zero_indices, min_distance=16):
        """Simple clustering of lesion voxels to find centers"""
        if len(non_zero_indices) == 0:
            return []
        
        centers = [non_zero_indices[0].tolist()]
        
        for idx in non_zero_indices:
            idx = idx.tolist()
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
        """Pad patch to match required patch size"""
        c, d, h, w = patch.shape
        pad_d = max(0, self.patch_size[0] - d)
        pad_h = max(0, self.patch_size[1] - h)
        pad_w = max(0, self.patch_size[2] - w)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            return torch.nn.functional.pad(
                patch, 
                (0, pad_w, 0, pad_h, 0, pad_d), 
                mode='constant', 
                value=0
            )
        return patch

    def reconstruct_from_patches(self, patches, patch_coords, output_shape):
        """
        Reconstruct full volume from patches
        """
        reconstructed = torch.zeros(output_shape, device=patches[0].device)
        count = torch.zeros(output_shape, device=patches[0].device)
        
        for patch, (z, y, x) in zip(patches, patch_coords):
            z_end = min(z + self.patch_size[0], output_shape[1])
            y_end = min(y + self.patch_size[1], output_shape[2])
            x_end = min(x + self.patch_size[2], output_shape[3])
            
            patch_z = z_end - z
            patch_y = y_end - y
            patch_x = x_end - x
            
            reconstructed[:, z:z_end, y:y_end, x:x_end] += patch[:, :patch_z, :patch_y, :patch_x]
            count[:, z:z_end, y:y_end, x:x_end] += 1
        
        # Average overlapping regions
        count[count == 0] = 1  # Avoid division by zero
        reconstructed = reconstructed / count
        
        return reconstructed

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

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction='none')
        
    def forward(self, pred, target, mask):
        # Create a slightly dilated mask to include the perimeter
        dilated_mask = self._dilate_mask(mask)
        
        # Calculate L1 loss
        loss = self.criterion(pred, target)
        
        # Apply mask - only compute loss in lesion regions and their perimeter
        masked_loss = loss * dilated_mask
        
        # Average over non-zero elements
        non_zero_elements = dilated_mask.sum()
        if non_zero_elements > 0:
            return masked_loss.sum() / non_zero_elements
        else:
            return torch.tensor(0.0, device=pred.device)
    
    def _dilate_mask(self, mask, kernel_size=3):
        """Dilate the mask by a small amount to include perimeter"""
        dilated_mask = torch.nn.functional.max_pool3d(
            mask, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2
        )
        return dilated_mask

class GradientSmoothingLoss(nn.Module):
    def __init__(self, weight=0.1):
        super(GradientSmoothingLoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred, mask):
        # Create a slightly dilated mask to include the perimeter
        dilated_mask = self._dilate_mask(mask)
        
        # Calculate gradients in 3 dimensions
        grad_z = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        grad_y = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        grad_x = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        # Apply mask to gradients
        mask_z = dilated_mask[:, :, 1:, :, :]
        mask_y = dilated_mask[:, :, :, 1:, :]
        mask_x = dilated_mask[:, :, :, :, 1:]
        
        # Calculate gradient smoothing loss
        loss_z = (grad_z**2 * mask_z).sum() / (mask_z.sum() + 1e-8)
        loss_y = (grad_y**2 * mask_y).sum() / (mask_y.sum() + 1e-8)
        loss_x = (grad_x**2 * mask_x).sum() / (mask_x.sum() + 1e-8)
        
        return self.weight * (loss_z + loss_y + loss_x)
    
    def _dilate_mask(self, mask, kernel_size=3):
        """Dilate the mask by a small amount to include perimeter"""
        dilated_mask = torch.nn.functional.max_pool3d(
            mask, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size//2
        )
        return dilated_mask

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms - specify SimpleITK reader explicitly
    train_transforms = Compose([
        LoadImaged(keys=["pseudo_healthy", "adc", "label"], reader="SimpleITK"),
        EnsureChannelFirstd(keys=["pseudo_healthy", "adc", "label"]),
        Orientationd(keys=["pseudo_healthy", "adc", "label"], axcodes="RAS"),
        Spacingd(keys=["pseudo_healthy", "adc", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "nearest")),
        ScaleIntensityd(keys=["pseudo_healthy", "adc"], minv=0.0, maxv=1.0),
        RandAffined(
            keys=["pseudo_healthy", "adc", "label"],
            prob=0.15,
            rotate_range=(0.05, 0.05, 0.05),
            scale_range=(0.05, 0.05, 0.05),
            mode=("bilinear", "bilinear", "nearest"),
            padding_mode="zeros"
        ),
        RandScaleIntensityd(keys=["pseudo_healthy", "adc"], factors=0.1, prob=0.15),
        EnsureTyped(keys=["pseudo_healthy", "adc", "label"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["pseudo_healthy", "adc", "label"], reader="SimpleITK"),
        EnsureChannelFirstd(keys=["pseudo_healthy", "adc", "label"]),
        Orientationd(keys=["pseudo_healthy", "adc", "label"], axcodes="RAS"),
        Spacingd(keys=["pseudo_healthy", "adc", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "nearest")),
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
    
    # Define loss functions
    masked_l1_loss = MaskedL1Loss().to(device)
    gradient_loss = GradientSmoothingLoss(weight=0.1).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize patch extractor
    patch_extractor = PatchExtractor(patch_size=(64, 64, 32))
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        train_samples = 0
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            pseudo_healthy = batch_data["pseudo_healthy"].to(device)
            adc = batch_data["adc"].to(device)
            label = batch_data["label"].to(device)
            
            # Extract patches containing lesions
            patches_ph, patches_label, patches_adc, _ = patch_extractor.extract_patches_with_lesions(
                pseudo_healthy[0], label[0], adc[0]
            )
            
            batch_loss = 0
            for ph_patch, label_patch, adc_patch in zip(patches_ph, patches_label, patches_adc):
                # Add batch dimension
                ph_patch = ph_patch.unsqueeze(0)
                label_patch = label_patch.unsqueeze(0)
                adc_patch = adc_patch.unsqueeze(0)
                
                # Generate inpainted output
                output = model(ph_patch, label_patch)
                
                # Calculate losses
                l1_loss = masked_l1_loss(output, adc_patch, label_patch)
                smooth_loss = gradient_loss(output, label_patch)
                loss = l1_loss + smooth_loss
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                train_samples += 1
            
            # Average loss over patches
            if len(patches_ph) > 0:
                avg_batch_loss = batch_loss / len(patches_ph)
                epoch_loss += avg_batch_loss
                
                if batch_idx % 5 == 0:
                    print(f"Batch {batch_idx}, Loss: {avg_batch_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for val_batch_idx, val_batch_data in enumerate(val_loader):
                pseudo_healthy = val_batch_data["pseudo_healthy"].to(device)
                adc = val_batch_data["adc"].to(device)
                label = val_batch_data["label"].to(device)
                
                # Extract patches containing lesions
                patches_ph, patches_label, patches_adc, patch_coords = patch_extractor.extract_patches_with_lesions(
                    pseudo_healthy[0], label[0], adc[0]
                )
                
                output_patches = []
                for ph_patch, label_patch in zip(patches_ph, patches_label):
                    ph_patch = ph_patch.unsqueeze(0)
                    label_patch = label_patch.unsqueeze(0)
                    
                    # Generate inpainted output
                    output_patch = model(ph_patch, label_patch)
                    output_patches.append(output_patch[0])
                
                # Reconstruct full volume
                if output_patches:
                    reconstructed = patch_extractor.reconstruct_from_patches(
                        output_patches, patch_coords, pseudo_healthy.shape
                    )
                    
                    # Apply the lesion mask to only modify lesion regions
                    dilated_mask = torch.nn.functional.max_pool3d(
                        label, kernel_size=3, stride=1, padding=1
                    )
                    
                    # Create final inpainted image - copy pseudo-healthy and replace only in lesion regions
                    final_output = pseudo_healthy.clone()
                    final_output = final_output * (1 - dilated_mask) + reconstructed * dilated_mask
                    
                    # Calculate validation loss
                    l1_loss = masked_l1_loss(final_output, adc, label)
                    smooth_loss = gradient_loss(final_output, label)
                    loss = l1_loss + smooth_loss
                    val_loss += loss.item()
                    
                    # Save validation visualization for the first batch
                    if val_batch_idx == 0 and (epoch + 1) % args.vis_freq == 0:
                        visualize_results(
                            pseudo_healthy[0], 
                            label[0], 
                            final_output[0], 
                            adc[0],
                            os.path.join(args.output_dir, f"epoch_{epoch+1}_validation.pdf")
                        )
        
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'val_loss': avg_val_loss
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"))

def visualize_results(pseudo_healthy, label, output, target, output_path):
    """
    Visualize results by creating a PDF with slice-by-slice comparisons
    """
    # Convert tensors to numpy arrays
    pseudo_healthy = pseudo_healthy.detach().cpu().numpy()[0]  # Remove channel dim
    label = label.detach().cpu().numpy()[0]
    output = output.detach().cpu().numpy()[0]
    target = target.detach().cpu().numpy()[0]
    
    # Create PDF
    with PdfPages(output_path) as pdf:
        # Only include slices that have lesions
        lesion_slices = np.where(np.sum(label, axis=(1, 2)) > 0)[0]
        
        if len(lesion_slices) == 0:
            # If no lesion slices, show a few random slices
            lesion_slices = np.linspace(0, label.shape[0]-1, 5, dtype=int)
        
        for z in lesion_slices:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            
            # Plot pseudo-healthy
            im0 = axs[0].imshow(pseudo_healthy[z], cmap='gray')
            axs[0].set_title(f'Pseudo-healthy (z={z})')
            plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
            
            # Plot label
            im1 = axs[1].imshow(label[z], cmap='Reds')
            axs[1].set_title(f'Lesion Mask')
            plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            
            # Plot output
            im2 = axs[2].imshow(output[z], cmap='gray')
            axs[2].set_title(f'Inpainted Output')
            plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
            
            # Plot target
            im3 = axs[3].imshow(target[z], cmap='gray')
            axs[3].set_title(f'Target ADC')
            plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

def inference(args):
    """
    Run inference on new data
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms - specify SimpleITK reader explicitly
    infer_transforms = Compose([
        LoadImaged(keys=["pseudo_healthy", "label"], reader="SimpleITK"),
        EnsureChannelFirstd(keys=["pseudo_healthy", "label"]),
        Orientationd(keys=["pseudo_healthy", "label"], axcodes="RAS"),
        Spacingd(keys=["pseudo_healthy", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["pseudo_healthy"], minv=0.0, maxv=1.0),
        EnsureTyped(keys=["pseudo_healthy", "label"]),
    ])
    
    # Load pseudo-healthy and lesion files
    data = [{
        "pseudo_healthy": args.input_pseudo_healthy,
        "label": args.input_label
    }]
    
    # Create dataset
    dataset = monai.data.Dataset(data=data, transform=infer_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = LesionInpaintingModel(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Initialize patch extractor
    patch_extractor = PatchExtractor(patch_size=(64, 64, 32))
    
    # Perform inference
    with torch.no_grad():
        for batch_data in loader:
            pseudo_healthy = batch_data["pseudo_healthy"].to(device)
            label = batch_data["label"].to(device)
            
            # Get original shape and metadata for saving output
            original_data = sitk.ReadImage(args.input_pseudo_healthy)
            
            # Extract patches with lesions
            patches_ph, patches_label, _, patch_coords = patch_extractor.extract_patches_with_lesions(
                pseudo_healthy[0], label[0], pseudo_healthy[0]  # Use pseudo_healthy as placeholder for adc
            )
            
            output_patches = []
            for ph_patch, label_patch in zip(patches_ph, patches_label):
                ph_patch = ph_patch.unsqueeze(0)
                label_patch = label_patch.unsqueeze(0)
                
                # Generate inpainted output
                output_patch = model(ph_patch, label_patch)
                output_patches.append(output_patch[0])
            
            # Reconstruct full volume
            if output_patches:
                reconstructed = patch_extractor.reconstruct_from_patches(
                    output_patches, patch_coords, pseudo_healthy.shape
                )
                
                # Apply the lesion mask to only modify lesion regions
                dilated_mask = torch.nn.functional.max_pool3d(
                    label, kernel_size=3, stride=1, padding=1
                )
                
                # Create final inpainted image - copy pseudo-healthy and replace only in lesion regions
                final_output = pseudo_healthy.clone()
                final_output = final_output * (1 - dilated_mask) + reconstructed * dilated_mask
                
                # Convert to numpy and rescale if needed
                output_np = final_output[0, 0].cpu().numpy()
                
                # Create SimpleITK image with same properties as input
                output_image = sitk.GetImageFromArray(output_np)
                output_image.CopyInformation(original_data)
                
                # Save output
                sitk.WriteImage(output_image, args.output_file)
                print(f"Saved inpainted result to {args.output_file}")
                
                # Visualize result if requested
                if args.visualize:
                    visualize_results(
                        pseudo_healthy[0], 
                        label[0], 
                        final_output[0], 
                        pseudo_healthy[0],  # Use pseudo-healthy as placeholder for target
                        args.output_file.replace(".mha", ".pdf")
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
    train_parser.add_argument("--vis_freq", type=int, default=5, help="Visualization frequency")
    
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
