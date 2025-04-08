import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from monai.networks.nets import SwinUNETR
from tqdm import tqdm
import random
import glob
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import argparse
from torchvision.utils import make_grid


class LesionInpaintingDataset(Dataset):
    """
    Dataset for HIE lesion inpainting.
    
    It pairs:
    1. ADC maps without lesions in a specific area (truly healthy brain)
    2. Binary masks for synthetic lesions (where to place new lesions)
    3. Ground truth ADC maps with real lesions (for learning what lesions look like)
    """
    def __init__(self, 
                 zadc_dir,
                 label_dir, 
                 synthetic_lesions_dir,
                 patch_size=(96, 96, 96),
                 mode='train',
                 transform=None):
        """
        Args:
            zadc_dir: Directory with ZADC maps
            label_dir: Directory with lesion masks
            synthetic_lesions_dir: Directory with synthetic lesions
            patch_size: Size of the patches to extract
            mode: 'train' or 'val'
            transform: Optional transform to be applied on a sample
        """
        self.zadc_dir = zadc_dir
        self.label_dir = label_dir
        self.synthetic_lesions_dir = synthetic_lesions_dir
        self.patch_size = patch_size
        self.mode = mode
        self.transform = transform
        
        # Debugging info
        print(f"\n----- Dataset Initialization ({mode}) -----")
        print(f"ZADC directory: {zadc_dir}")
        print(f"Label directory: {label_dir}")
        print(f"Synthetic lesions directory: {synthetic_lesions_dir}")
        
        # Verify directories exist
        if not os.path.exists(zadc_dir):
            print(f"WARNING: ZADC directory {zadc_dir} does not exist!")
        if not os.path.exists(label_dir):
            print(f"WARNING: Label directory {label_dir} does not exist!")
        if not os.path.exists(synthetic_lesions_dir):
            print(f"WARNING: Synthetic lesions directory {synthetic_lesions_dir} does not exist!")
        
        # Get all ZADC files
        self.zadc_files = sorted(glob.glob(os.path.join(zadc_dir, "*.mha")))
        print(f"Found {len(self.zadc_files)} ZADC files")
        
        if len(self.zadc_files) == 0:
            print(f"Contents of ZADC directory: {os.listdir(zadc_dir) if os.path.exists(zadc_dir) else 'Directory not found'}")
        
        # Filter for patients with corresponding synthetic lesions
        valid_patients = []
        for zadc_file in self.zadc_files:
            # Extract base filename
            base_filename = os.path.basename(zadc_file)
            
            # Check if the filename starts with "Zmap_" and remove it
            if base_filename.startswith("Zmap_"):
                # Remove "Zmap_" prefix
                base_without_prefix = base_filename[5:]  # Skip the first 5 characters "Zmap_"
            else:
                base_without_prefix = base_filename
            
            # Extract the patient ID from the first two segments
            parts = base_without_prefix.split('-')
            if len(parts) >= 2:
                patient_id = parts[0] + '-' + parts[1]
                
                # Check if corresponding synthetic lesion directory exists
                synthetic_dir = os.path.join(synthetic_lesions_dir, patient_id)
                if os.path.exists(synthetic_dir):
                    valid_patients.append(patient_id)
                else:
                    print(f"No synthetic lesions found for patient {patient_id} at {synthetic_dir}")
            else:
                print(f"Could not extract patient ID from filename: {base_filename}")
        
        self.valid_patients = valid_patients
        print(f"Found {len(self.valid_patients)} valid patients with synthetic lesions")
        
        if len(self.valid_patients) == 0:
            # Print the first few ZADC filenames to help diagnose the issue
            if len(self.zadc_files) > 0:
                print(f"Sample ZADC files (first 5):")
            
            print(f"Contents of synthetic lesions dir: {os.listdir(synthetic_lesions_dir) if os.path.exists(synthetic_lesions_dir) else 'Directory not found'}")
        
        # Split into train/val
        if mode == 'train':
            self.patients = self.valid_patients[:int(0.8 * len(self.valid_patients))]
        else:
            self.patients = self.valid_patients[int(0.8 * len(self.valid_patients)):]
        
        print(f"Using {len(self.patients)} patients for {mode} mode")
        
        # Create patient-synthetic lesion pairs
        self.samples = []
        for patient in self.patients:
            print(f"\nProcessing patient: {patient}")
            
            # Check for ZADC files (format: Zmap_MGHNICU_xxx-VISIT_xx-ADC_smooth2mm_clipped10.mha)
            zadc_pattern = os.path.join(zadc_dir, f"Zmap_{patient}*.mha")
            zadc_files = glob.glob(zadc_pattern)
            if not zadc_files:
                print(f"WARNING: No ZADC files found for pattern: {zadc_pattern}")
                continue
            zadc_file = zadc_files[0]
            print(f"Found ZADC file: {os.path.basename(zadc_file)}")
            
            # Check for label files (format: MGHNICU_xxx-VISIT_xx_lesion.mha)
            label_pattern = os.path.join(label_dir, f"{patient}_lesion.mha")
            label_files = glob.glob(label_pattern)
            if not label_files:
                print(f"WARNING: No label files found for pattern: {label_pattern}")
                # Try alternative pattern with wildcard
                alt_label_pattern = os.path.join(label_dir, f"{patient}*lesion.mha")
                label_files = glob.glob(alt_label_pattern)
                if not label_files:
                    print(f"WARNING: No label files found for alternative pattern: {alt_label_pattern}")
                    continue
            label_file = label_files[0]
            print(f"Found label file: {os.path.basename(label_file)}")
            
            # Check for synthetic lesions (format: .../MGHNICU_xxx-VISIT_xx/registered_lesion_sampleXX.mha)
            syn_pattern = os.path.join(synthetic_lesions_dir, patient, "registered_lesion_*.mha")
            synthetic_lesions = glob.glob(syn_pattern)
            if not synthetic_lesions:
                print(f"WARNING: No synthetic lesions found for pattern: {syn_pattern}")
                print(f"Does directory exist? {os.path.exists(os.path.join(synthetic_lesions_dir, patient))}")
                if os.path.exists(os.path.join(synthetic_lesions_dir, patient)):
                    print(f"Contents of patient directory: {os.listdir(os.path.join(synthetic_lesions_dir, patient))}")
                continue
            
            print(f"Found {len(synthetic_lesions)} synthetic lesions")
            
            # Create sample pairs
            for syn_lesion in synthetic_lesions:
                self.samples.append({
                    'zadc': zadc_file,
                    'label': label_file,
                    'synthetic_lesion': syn_lesion
                })
        
        print(f"Created {len(self.samples)} samples for {mode} mode")
        
        if len(self.samples) == 0:
            print(f"ERROR: No valid samples could be created for {mode} mode!")
            print("Please check data paths and ensure the directory structure matches expected patterns")
        
        print(f"----- End Dataset Initialization ({mode}) -----\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load ZADC map (may contain real lesions)
        zadc_img = sitk.ReadImage(sample['zadc'])
        zadc_array = sitk.GetArrayFromImage(zadc_img)
        
        # Load original lesion mask (real lesions)
        label_img = sitk.ReadImage(sample['label'])
        label_array = sitk.GetArrayFromImage(label_img)
        
        # Load synthetic lesion (where we want to place a lesion)
        syn_lesion_img = sitk.ReadImage(sample['synthetic_lesion'])
        syn_lesion_array = sitk.GetArrayFromImage(syn_lesion_img)
        
        # Normalize ZADC to [0, 1]
        zadc_array = (zadc_array - zadc_array.min()) / (zadc_array.max() - zadc_array.min() + 1e-8)
        
        # Convert label and synthetic lesion to binary
        label_array = (label_array > 0).astype(np.float32)
        syn_lesion_array = (syn_lesion_array > 0).astype(np.float32)
        
        # Create truly healthy brain by removing existing lesions
        # This is a critical step: we replace lesion areas with estimated healthy tissue values
        healthy_brain = zadc_array.copy()
        
        # Find non-zero regions in the original lesion mask (existing lesions)
        real_lesion_indices = np.where(label_array > 0)
        
        if len(real_lesion_indices[0]) > 0:
            # Calculate average intensity of surrounding non-lesion tissue
            # First, dilate the lesion mask to create a border region
            from scipy import ndimage
            dilated_mask = ndimage.binary_dilation(label_array, iterations=2)
            border_mask = dilated_mask & ~label_array  # Border around lesion but not lesion itself
            
            # Get average intensity value in the border region
            if np.any(border_mask):
                border_values = zadc_array[border_mask]
                healthy_tissue_value = np.mean(border_values)
            else:
                # Fallback - use global average of non-lesion areas
                healthy_tissue_value = np.mean(zadc_array[~label_array])
            
            # Replace lesion areas with healthy tissue values
            healthy_brain[label_array > 0] = healthy_tissue_value
            
            print(f"Removed real lesions from input brain. Avg healthy value: {healthy_tissue_value:.4f}")
        
        # Find non-zero region in the synthetic lesion
        z_indices, y_indices, x_indices = np.where(syn_lesion_array > 0)
        
        if len(z_indices) > 0:
            # Extract a patch centered on the synthetic lesion
            center_z = (z_indices.min() + z_indices.max()) // 2
            center_y = (y_indices.min() + y_indices.max()) // 2
            center_x = (x_indices.min() + x_indices.max()) // 2
            
            # Calculate the patch boundaries
            z_start = max(0, center_z - self.patch_size[0] // 2)
            y_start = max(0, center_y - self.patch_size[1] // 2)
            x_start = max(0, center_x - self.patch_size[2] // 2)
            
            z_end = min(zadc_array.shape[0], z_start + self.patch_size[0])
            y_end = min(zadc_array.shape[1], y_start + self.patch_size[1])
            x_end = min(zadc_array.shape[2], x_start + self.patch_size[2])
            
            # Extract patches
            zadc_patch = zadc_array[z_start:z_end, y_start:y_end, x_start:x_end]
            healthy_patch = healthy_brain[z_start:z_end, y_start:y_end, x_start:x_end]  # Now truly healthy
            label_patch = label_array[z_start:z_end, y_start:y_end, x_start:x_end]
            syn_lesion_patch = syn_lesion_array[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Pad if necessary to match patch_size
            z_pad = max(0, self.patch_size[0] - zadc_patch.shape[0])
            y_pad = max(0, self.patch_size[1] - zadc_patch.shape[1])
            x_pad = max(0, self.patch_size[2] - zadc_patch.shape[2])
            
            if z_pad > 0 or y_pad > 0 or x_pad > 0:
                zadc_patch = np.pad(zadc_patch, ((0, z_pad), (0, y_pad), (0, x_pad)), mode='constant')
                healthy_patch = np.pad(healthy_patch, ((0, z_pad), (0, y_pad), (0, x_pad)), mode='constant')
                label_patch = np.pad(label_patch, ((0, z_pad), (0, y_pad), (0, x_pad)), mode='constant')
                syn_lesion_patch = np.pad(syn_lesion_patch, ((0, z_pad), (0, y_pad), (0, x_pad)), mode='constant')
            
            # Convert to tensors
            zadc_tensor = torch.from_numpy(zadc_patch).float().unsqueeze(0)  # Real brain with lesions
            healthy_tensor = torch.from_numpy(healthy_patch).float().unsqueeze(0)  # Truly healthy brain
            label_tensor = torch.from_numpy(label_patch).float().unsqueeze(0)  # Real lesion mask
            syn_lesion_tensor = torch.from_numpy(syn_lesion_patch).float().unsqueeze(0)  # Synthetic lesion mask
            
            # Combined input: TRULY healthy brain and synthetic lesion mask
            input_tensor = torch.cat([healthy_tensor, syn_lesion_tensor], dim=0)
            
            # If there's a substantial overlap between the synthetic lesion and real lesion,
            # we can use the real brain as a target. Otherwise, we need to create a hybrid target.
            synthetic_real_overlap = np.sum(syn_lesion_patch * label_patch) / np.sum(syn_lesion_patch)
            
            if synthetic_real_overlap > 0.5:
                # More than 50% overlap - we can use the real brain with its lesion as target
                target_tensor = zadc_tensor
                print(f"Using real lesion as target (overlap: {synthetic_real_overlap:.2f})")
            else:
                # Less overlap - create a hybrid target where we add "lesion-like" values to the healthy brain
                # in the area of the synthetic mask
                hybrid_target = healthy_patch.copy()
                
                # Calculate average intensity in real lesion areas
                if np.sum(label_patch) > 0:
                    lesion_value = np.mean(zadc_patch[label_patch > 0])
                else:
                    # No real lesions in this patch, use a typical lesion intensity reduction
                    # (ADC typically shows hypointensity in acute stroke)
                    non_zero_mask = healthy_patch > 0.1  # Avoid background
                    avg_healthy = np.mean(healthy_patch[non_zero_mask])
                    lesion_value = avg_healthy * 0.6  # Typical reduction in ADC values
                
                # Apply lesion-like values to the synthetic mask area
                hybrid_target[syn_lesion_patch > 0] = lesion_value
                target_tensor = torch.from_numpy(hybrid_target).float().unsqueeze(0)
                print(f"Created hybrid target (low overlap: {synthetic_real_overlap:.2f}, lesion value: {lesion_value:.4f})")
            
            return {
                'input': input_tensor,  # Truly healthy brain + synthetic lesion mask
                'target': target_tensor,  # Brain with real or simulated lesion
                'original_mask': label_tensor,  # Original lesion mask
                'synthetic_mask': syn_lesion_tensor  # Synthetic lesion mask
            }
        else:
            # Fallback if no lesion is found (should be rare)
            dummy_tensor = torch.zeros((1, *self.patch_size))
            return {
                'input': torch.cat([dummy_tensor, dummy_tensor], dim=0),
                'target': dummy_tensor,
                'original_mask': dummy_tensor,
                'synthetic_mask': dummy_tensor
            }


class PatchGANDiscriminator(nn.Module):
    """
    3D PatchGAN discriminator for lesion inpainting.
    """
    def __init__(self, in_channels=2, ndf=64, n_layers=3):
        """
        Args:
            in_channels: Number of input channels (ZADC + mask)
            ndf: Number of filters in the first conv layer
        """
        super(PatchGANDiscriminator, self).__init__()
        
        sequence = [
            nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, 
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1),
            nn.BatchNorm3d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x, mask):
        """
        Args:
            x: Input image
            mask: Lesion mask to focus attention on specific regions
        """
        # Concatenate image and mask
        x_and_mask = torch.cat([x, mask], dim=1)
        return self.model(x_and_mask)


class LesionInpaintingGAN:
    """
    GAN model for HIE lesion inpainting.
    """
    def __init__(self, 
                 img_shape=(96, 96, 96),
                 device=None):
        """
        Args:
            img_shape: Shape of the input image
            device: Device to run the model on
        """
        self.img_shape = img_shape
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generator - SwinUNETR for high-quality 3D inpainting
        self.generator = SwinUNETR(
            img_size=img_shape,
            in_channels=2,  # Healthy brain + synthetic lesion mask
            out_channels=1,  # Inpainted ZADC
            feature_size=24,
            use_checkpoint=True
        ).to(self.device)
        
        # Discriminator - PatchGAN for assessing realism
        self.discriminator = PatchGANDiscriminator(
            in_channels=2  # ZADC + lesion mask
        ).to(self.device)
        
        # Optimizers
        self.optimizer_G = Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_D = Adam(self.discriminator.parameters(), lr=4e-4, betas=(0.5, 0.999))
        
        # Learning rate schedulers
        self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=200)
        self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=200)
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
    def train_step(self, data):
        """
        Execute one training step.
        
        Args:
            data: Dictionary containing 'input', 'target', 'synthetic_mask'
        
        Returns:
            Dictionary of losses
        """
        # Set models to training mode
        self.generator.train()
        self.discriminator.train()
        
        # Get data
        real_brain = data['target'].to(self.device)
        input_data = data['input'].to(self.device)
        synthetic_mask = data['synthetic_mask'].to(self.device)
        
        # -----------------
        # Train Generator
        # -----------------
        self.optimizer_G.zero_grad()
        
        # Generate inpainted image
        fake_brain = self.generator(input_data)
        
        # Mask predictions from discriminator (only care about lesion area)
        pred_fake = self.discriminator(fake_brain, synthetic_mask)
        
        # Calculate generator losses
        valid = torch.ones_like(pred_fake).to(self.device)
        adv_loss = self.adversarial_loss(pred_fake, valid)
        
        # Pixel-wise loss (only for lesion area)
        mask_expanded = synthetic_mask.expand_as(real_brain)
        l1_loss = self.l1_loss(fake_brain * mask_expanded, real_brain * mask_expanded) * 100
        
        # Context loss (for surrounding area)
        inv_mask = 1 - synthetic_mask
        inv_mask_expanded = inv_mask.expand_as(real_brain)
        context_loss = self.l1_loss(fake_brain * inv_mask_expanded, real_brain * inv_mask_expanded) * 50
        
        # Total generator loss
        g_loss = adv_loss + l1_loss + context_loss
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # -----------------
        # Train Discriminator
        # -----------------
        self.optimizer_D.zero_grad()
        
        # Real brain with real lesions
        pred_real = self.discriminator(real_brain, synthetic_mask)
        real_labels = torch.ones_like(pred_real).to(self.device)
        real_loss = self.adversarial_loss(pred_real, real_labels)
        
        # Generated brain with fake lesions
        pred_fake = self.discriminator(fake_brain.detach(), synthetic_mask)
        fake_labels = torch.zeros_like(pred_fake).to(self.device)
        fake_loss = self.adversarial_loss(pred_fake, fake_labels)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        self.optimizer_D.step()
        
        return {
            'g_loss': g_loss.item(),
            'adv_loss': adv_loss.item(),
            'l1_loss': l1_loss.item(),
            'context_loss': context_loss.item(),
            'd_loss': d_loss.item()
        }
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Dictionary of validation metrics
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {
            'g_loss': 0,
            'adv_loss': 0,
            'l1_loss': 0,
            'context_loss': 0,
            'd_loss': 0
        }
        
        with torch.no_grad():
            for batch in dataloader:
                real_brain = batch['target'].to(self.device)
                input_data = batch['input'].to(self.device)
                synthetic_mask = batch['synthetic_mask'].to(self.device)
                
                # Generate inpainted image
                fake_brain = self.generator(input_data)
                
                # Discriminator predictions
                pred_fake = self.discriminator(fake_brain, synthetic_mask)
                pred_real = self.discriminator(real_brain, synthetic_mask)
                
                # Adversarial loss
                valid = torch.ones_like(pred_fake).to(self.device)
                adv_loss = self.adversarial_loss(pred_fake, valid)
                
                # Pixel-wise loss for lesion area
                mask_expanded = synthetic_mask.expand_as(real_brain)
                l1_loss = self.l1_loss(fake_brain * mask_expanded, real_brain * mask_expanded) * 100
                
                # Context loss for surrounding area
                inv_mask = 1 - synthetic_mask
                inv_mask_expanded = inv_mask.expand_as(real_brain)
                context_loss = self.l1_loss(fake_brain * inv_mask_expanded, real_brain * inv_mask_expanded) * 50
                
                # Total generator loss
                g_loss = adv_loss + l1_loss + context_loss
                
                # Discriminator loss
                real_labels = torch.ones_like(pred_real).to(self.device)
                fake_labels = torch.zeros_like(pred_fake).to(self.device)
                real_loss = self.adversarial_loss(pred_real, real_labels)
                fake_loss = self.adversarial_loss(pred_fake, fake_labels)
                d_loss = (real_loss + fake_loss) / 2
                
                # Accumulate losses
                val_losses['g_loss'] += g_loss.item()
                val_losses['adv_loss'] += adv_loss.item()
                val_losses['l1_loss'] += l1_loss.item()
                val_losses['context_loss'] += context_loss.item()
                val_losses['d_loss'] += d_loss.item()
                
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(dataloader)
            
        return val_losses
    
    def save_models(self, save_dir, epoch):
        """
        Save model checkpoints.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'scheduler_G': self.scheduler_G.state_dict(),
            'scheduler_D': self.scheduler_D.state_dict(),
            'epoch': epoch
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    def load_models(self, checkpoint_path):
        """
        Load model checkpoints.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])
        return checkpoint['epoch']
    
    def inference(self, healthy_brain, synthetic_lesion_mask):
        """
        Perform inference on a single sample.
        
        Args:
            healthy_brain: Healthy brain scan (tensor of shape [1, D, H, W])
            synthetic_lesion_mask: Synthetic lesion mask (tensor of shape [1, D, H, W])
        
        Returns:
            Inpainted brain scan with lesion
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Ensure inputs are on the correct device
            healthy_brain = healthy_brain.to(self.device)
            synthetic_lesion_mask = synthetic_lesion_mask.to(self.device)
            
            # Combine inputs
            combined_input = torch.cat([healthy_brain, synthetic_lesion_mask], dim=1)
            
            # Generate inpainted image
            inpainted_brain = self.generator(combined_input)
            
        return inpainted_brain


def train(zadc_dir, label_dir, synthetic_lesions_dir, output_dir, num_epochs=200, batch_size=4, save_interval=5):
    """
    Train the HIE lesion inpainting GAN.
    
    Args:
        zadc_dir: Directory containing ZADC maps
        label_dir: Directory containing lesion labels
        synthetic_lesions_dir: Directory containing synthetic lesions
        output_dir: Directory to save model checkpoints and results
        num_epochs: Number of epochs to train
        batch_size: Batch size
        save_interval: Interval (in epochs) for saving model checkpoints
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n===== VALIDATING DATASET PATHS =====")
    print(f"ZADC directory: {zadc_dir} - Exists: {os.path.exists(zadc_dir)}")
    print(f"Label directory: {label_dir} - Exists: {os.path.exists(label_dir)}")
    print(f"Synthetic lesions directory: {synthetic_lesions_dir} - Exists: {os.path.exists(synthetic_lesions_dir)}")
    if os.path.exists(synthetic_lesions_dir):
        print(f"Contents of synthetic_lesions_dir: {os.listdir(synthetic_lesions_dir)[:10]}")
    print("=====================================\n")
    
    # Create datasets
    train_dataset = LesionInpaintingDataset(
        zadc_dir=zadc_dir,
        label_dir=label_dir,
        synthetic_lesions_dir=synthetic_lesions_dir,
        patch_size=(96, 96, 96),
        mode='train'
    )
    
    val_dataset = LesionInpaintingDataset(
        zadc_dir=zadc_dir,
        label_dir=label_dir,
        synthetic_lesions_dir=synthetic_lesions_dir,
        patch_size=(96, 96, 96),
        mode='val'
    )
    
    # Check if datasets have samples
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty! Cannot continue with training.\n"
            "Please check the following:\n"
            "1. The directories exist and contain the expected files\n"
            "2. The file naming conventions match those expected in the code\n"
            "3. The synthetic_lesions_dir has subdirectories named after patients\n"
            "4. Each patient directory contains synthetic lesion MHA files"
        )
    
    if len(val_dataset) == 0:
        print("WARNING: Validation dataset is empty. Training will continue without validation.")
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Initialize model
    gan_model = LesionInpaintingGAN(img_shape=(96, 96, 96))
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        epoch_losses = {
            'g_loss': 0,
            'adv_loss': 0,
            'l1_loss': 0,
            'context_loss': 0,
            'd_loss': 0
        }
        
        pbar = tqdm(train_loader)
        for batch in pbar:
            losses = gan_model.train_step(batch)
            
            # Update progress bar
            pbar.set_description(f"G: {losses['g_loss']:.4f}, D: {losses['d_loss']:.4f}")
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        train_losses.append(epoch_losses)
        
        # Validation
        if val_loader is not None:
            print("Validating...")
            val_epoch_losses = gan_model.validate(val_loader)
            val_losses.append(val_epoch_losses)
            print(f"Validation - G: {val_epoch_losses['g_loss']:.4f}, D: {val_epoch_losses['d_loss']:.4f}")
        
        # Update learning rates
        gan_model.scheduler_G.step()
        gan_model.scheduler_D.step()
        
        # Save checkpoint based on specified save_interval
        if (epoch + 1) % save_interval == 0:
            gan_model.save_models(output_dir, epoch + 1)
        
        # Generate visualizations after each epoch
        if val_loader is not None:
            with torch.no_grad():
                # Get a sample from validation set
                val_sample = next(iter(val_loader))
                input_data = val_sample['input'].to(gan_model.device)
                real_brain = val_sample['target'].to(gan_model.device)
                synthetic_mask = val_sample['synthetic_mask'].to(gan_model.device)
                
                # Generate inpainted image
                fake_brain = gan_model.generator(input_data)
                
                # Create detailed visualizations for each sample
                for i in range(min(2, input_data.shape[0])):
                    # Get current sample data
                    input_vol = input_data[i, 0].cpu().numpy()
                    mask_vol = synthetic_mask[i, 0].cpu().numpy()
                    real_vol = real_brain[i, 0].cpu().numpy()
                    fake_vol = fake_brain[i, 0].cpu().numpy()
                    
                    # Find slices containing the lesion
                    lesion_slices = []
                    for z in range(mask_vol.shape[0]):
                        if np.any(mask_vol[z] > 0):
                            lesion_slices.append(z)
                    
                    if not lesion_slices:
                        # If no lesion found, use middle slices
                        mid_slice = mask_vol.shape[0] // 2
                        lesion_slices = list(range(max(0, mid_slice-5), min(mask_vol.shape[0], mid_slice+6)))
                    
                    # We want ALL slices with lesion content
                    print(f"Visualizing all {len(lesion_slices)} slices with lesion content")
                    
                    # Calculate overall change in lesion area
                    lesion_area_values_before = input_vol[mask_vol > 0]
                    lesion_area_values_after = fake_vol[mask_vol > 0]
                    mean_value_before = np.mean(lesion_area_values_before) if len(lesion_area_values_before) > 0 else 0
                    mean_value_after = np.mean(lesion_area_values_after) if len(lesion_area_values_after) > 0 else 0
                    mean_abs_change = np.mean(np.abs(lesion_area_values_after - lesion_area_values_before)) if len(lesion_area_values_before) > 0 else 0
                    
                    # Create multi-page PDF to store all slices if there are too many
                    use_pdf = len(lesion_slices) > 15
                    if use_pdf:
                        from matplotlib.backends.backend_pdf import PdfPages
                        pdf_filename = os.path.join(output_dir, f'epoch_{epoch+1:03d}_sample_{i}_all_slices.pdf')
                        pdf = PdfPages(pdf_filename)
                        
                        # Create multiple figures with max 15 slices per figure
                        for slice_batch_idx in range(0, len(lesion_slices), 15):
                            batch_slices = lesion_slices[slice_batch_idx:slice_batch_idx+15]
                            fig, axes = plt.subplots(4, len(batch_slices), figsize=(4*len(batch_slices), 16))
                            
                            # If only one slice, reshape axes for indexing
                            if len(batch_slices) == 1:
                                axes = axes.reshape(4, 1)
                                
                            for j, slice_idx in enumerate(batch_slices):
                                # Process slice as before...
                                # Create overlay of input with lesion mask for visualization
                                input_with_mask = np.stack([input_vol[slice_idx], input_vol[slice_idx], input_vol[slice_idx]], axis=2)
                                # Add red overlay for lesion
                                mask_overlay = mask_vol[slice_idx] > 0
                                input_with_mask[mask_overlay, 0] = 1.0  # Red channel
                                input_with_mask[mask_overlay, 1] = 0.0  # Green channel
                                input_with_mask[mask_overlay, 2] = 0.0  # Blue channel
                                
                                # Calculate changes for this particular slice
                                slice_mask = mask_vol[slice_idx] > 0
                                if np.any(slice_mask):
                                    slice_before = input_vol[slice_idx][slice_mask]
                                    slice_after = fake_vol[slice_idx][slice_mask]
                                    slice_mean_change = np.mean(np.abs(slice_after - slice_before))
                                    slice_title = f'Slice {slice_idx} (Δ={slice_mean_change:.4f})'
                                else:
                                    slice_title = f'Slice {slice_idx}'
                                
                                # Create difference map between fake and input to show changes
                                diff_map = np.abs(fake_vol[slice_idx] - input_vol[slice_idx])
                                
                                # Plot each slice
                                axes[0, j].imshow(input_vol[slice_idx], cmap='gray')
                                axes[0, j].set_title(slice_title)
                                axes[0, j].axis('off')
                                
                                axes[1, j].imshow(input_with_mask)
                                axes[1, j].set_title(f'Lesion Overlay')
                                axes[1, j].axis('off')
                                
                                axes[2, j].imshow(fake_vol[slice_idx], cmap='gray')
                                axes[2, j].set_title(f'Generated')
                                axes[2, j].axis('off')
                                
                                # Show difference map - where changes were made
                                axes[3, j].imshow(diff_map, cmap='hot')
                                axes[3, j].set_title(f'Change Map')
                                axes[3, j].axis('off')
                            
                            # Add row labels
                            axes[0, 0].set_ylabel('Input Volume')
                            axes[1, 0].set_ylabel('Lesion Location')
                            axes[2, 0].set_ylabel('Generated Result')
                            axes[3, 0].set_ylabel('Change Heatmap')
                            
                            # Add overall statistics to the figure
                            plt.suptitle(f'Epoch {epoch+1} - G:{epoch_losses["g_loss"]:.4f}, D:{epoch_losses["d_loss"]:.4f}\n'
                                         f'Mean value in lesion area: Before={mean_value_before:.4f}, After={mean_value_after:.4f}, Change={mean_abs_change:.4f}')
                            
                            plt.tight_layout()
                            pdf.savefig(fig)
                            plt.close()
                            
                        pdf.close()
                        print(f"Saved all {len(lesion_slices)} slices to {pdf_filename}")
                    else:
                        # If few enough slices, just create one image
                        num_slices = len(lesion_slices)
                        fig, axes = plt.subplots(4, num_slices, figsize=(4*num_slices, 16))
                        
                        # If only one slice, reshape axes for indexing
                        if num_slices == 1:
                            axes = axes.reshape(4, 1)
                        
                        for j, slice_idx in enumerate(lesion_slices):
                            # Create overlay of input with lesion mask for visualization
                            input_with_mask = np.stack([input_vol[slice_idx], input_vol[slice_idx], input_vol[slice_idx]], axis=2)
                            # Add red overlay for lesion
                            mask_overlay = mask_vol[slice_idx] > 0
                            input_with_mask[mask_overlay, 0] = 1.0  # Red channel
                            input_with_mask[mask_overlay, 1] = 0.0  # Green channel
                            input_with_mask[mask_overlay, 2] = 0.0  # Blue channel
                            
                            # Calculate changes for this particular slice
                            slice_mask = mask_vol[slice_idx] > 0
                            if np.any(slice_mask):
                                slice_before = input_vol[slice_idx][slice_mask]
                                slice_after = fake_vol[slice_idx][slice_mask]
                                slice_mean_change = np.mean(np.abs(slice_after - slice_before))
                                slice_title = f'Slice {slice_idx} (Δ={slice_mean_change:.4f})'
                            else:
                                slice_title = f'Slice {slice_idx}'
                            
                            # Create difference map between fake and input to show changes
                            diff_map = np.abs(fake_vol[slice_idx] - input_vol[slice_idx])
                            
                            # Plot each slice
                            axes[0, j].imshow(input_vol[slice_idx], cmap='gray')
                            axes[0, j].set_title(slice_title)
                            axes[0, j].axis('off')
                            
                            axes[1, j].imshow(input_with_mask)
                            axes[1, j].set_title(f'Lesion Overlay')
                            axes[1, j].axis('off')
                            
                            axes[2, j].imshow(fake_vol[slice_idx], cmap='gray')
                            axes[2, j].set_title(f'Generated')
                            axes[2, j].axis('off')
                            
                            # Show difference map - where changes were made
                            axes[3, j].imshow(diff_map, cmap='hot')
                            axes[3, j].set_title(f'Change Map')
                            axes[3, j].axis('off')
                        
                        # Add row labels
                        if num_slices > 0:
                            axes[0, 0].set_ylabel('Input Volume')
                            axes[1, 0].set_ylabel('Lesion Location')
                            axes[2, 0].set_ylabel('Generated Result')
                            axes[3, 0].set_ylabel('Change Heatmap')
                        
                        # Add overall statistics to the figure
                        plt.suptitle(f'Epoch {epoch+1} - G:{epoch_losses["g_loss"]:.4f}, D:{epoch_losses["d_loss"]:.4f}\n'
                                     f'Mean value in lesion area: Before={mean_value_before:.4f}, After={mean_value_after:.4f}, Change={mean_abs_change:.4f}')
                        
                        # Save with epoch number and sample number
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'epoch_{epoch+1:03d}_sample_{i}_full_volume.png'), dpi=150)
                        plt.close()
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    # Generator losses
    plt.subplot(2, 1, 1)
    plt.plot([loss['g_loss'] for loss in train_losses], label='G Total Loss (Train)')
    plt.plot([loss['adv_loss'] for loss in train_losses], label='G Adversarial Loss (Train)')
    plt.plot([loss['l1_loss'] for loss in train_losses], label='G L1 Loss (Train)')
    plt.plot([loss['context_loss'] for loss in train_losses], label='G Context Loss (Train)')
    
    if val_losses:
        plt.plot([loss['g_loss'] for loss in val_losses], '--', label='G Total Loss (Val)')
    
    plt.legend()
    plt.title('Generator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Discriminator losses
    plt.subplot(2, 1, 2)
    plt.plot([loss['d_loss'] for loss in train_losses], label='D Loss (Train)')
    
    if val_losses:
        plt.plot([loss['d_loss'] for loss in val_losses], '--', label='D Loss (Val)')
    
    plt.legend()
    plt.title('Discriminator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    return gan_model


def apply_inpainting(gan_model, zadc_file, synthetic_lesion_file, output_file):
    """
    Apply the trained model to inpaint a lesion into a brain scan.
    
    Args:
        gan_model: Trained LesionInpaintingGAN model
        zadc_file: Path to the ZADC file (.mha)
        synthetic_lesion_file: Path to the synthetic lesion file (.mha)
        output_file: Path to save the output (.mha)
    """
    # Load ZADC map
    zadc_img = sitk.ReadImage(zadc_file)
    zadc_array = sitk.GetArrayFromImage(zadc_img)
    
    # Load synthetic lesion
    syn_lesion_img = sitk.ReadImage(synthetic_lesion_file)
    syn_lesion_array = sitk.GetArrayFromImage(syn_lesion_img)
    
    # Normalize ZADC to [0, 1]
    zadc_min = zadc_array.min()
    zadc_max = zadc_array.max()
    zadc_array_norm = (zadc_array - zadc_min) / (zadc_max - zadc_min + 1e-8)
    
    # Convert synthetic lesion to binary
    syn_lesion_array = (syn_lesion_array > 0).astype(np.float32)
    
    # Convert to tensors
    zadc_tensor = torch.from_numpy(zadc_array_norm).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    syn_lesion_tensor = torch.from_numpy(syn_lesion_array).float().unsqueeze(0).unsqueeze(0)
    
    # Generate inpainted image
    inpainted_brain = gan_model.inference(zadc_tensor, syn_lesion_tensor)
    
    # Convert back to numpy and denormalize
    inpainted_array = inpainted_brain.squeeze().cpu().numpy()
    inpainted_array = inpainted_array * (zadc_max - zadc_min) + zadc_min
    
    # Create SimpleITK image (using original image as reference)
    output_img = sitk.GetImageFromArray(inpainted_array)
    output_img.CopyInformation(zadc_img)  # Copy metadata
    
    # Save the output
    sitk.WriteImage(output_img, output_file)


if __name__ == "__main__":
    import argparse
    
    # Vytvoříme hlavní parser pro příkazovou řádku
    parser = argparse.ArgumentParser(description='HIE Lesion Inpainting GAN')
    
    # Vytvoříme podparsery pro různé příkazy (train, generate)
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parser pro příkaz "train"
    train_parser = subparsers.add_parser('train', help='Train the HIE lesion inpainting GAN model')
    train_parser.add_argument('--zadc_dir', type=str, default="data/BONBID2023_Train/2Z_ADC",
                       help='Directory containing ZADC maps')
    train_parser.add_argument('--label_dir', type=str, default="data/BONBID2023_Train/3LABEL",
                       help='Directory containing lesion label masks')
    train_parser.add_argument('--synthetic_lesions_dir', type=str, default="data/registered_lesions",
                       help='Directory containing synthetic lesions')
    train_parser.add_argument('--output_dir', type=str, default="output/lesion_inpainting",
                       help='Directory to save model checkpoints and results')
    train_parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of epochs to train')
    train_parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    train_parser.add_argument('--save_interval', type=int, default=5,
                       help='Interval (in epochs) for saving model checkpoints')
    train_parser.add_argument('--test_patient', type=str, default="MGHNICU_445-VISIT_01",
                       help='Patient ID to use for test inference after training')
    train_parser.add_argument('--test_lesion', type=str, default="registered_lesion_sample33.mha",
                       help='Synthetic lesion file to use for test inference')
    train_parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training (cuda or cpu). Uses cuda if available by default.')
    
    # Parser pro příkaz "generate"
    generate_parser = subparsers.add_parser('generate', help='Generate inpainted brain images using a trained model')
    generate_parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    generate_parser.add_argument('--zadc_file', type=str, required=True,
                       help='Path to ZADC MHA file to inpaint')
    generate_parser.add_argument('--lesion_file', type=str, required=True,
                       help='Path to synthetic lesion mask MHA file')
    generate_parser.add_argument('--output_file', type=str, required=True,
                       help='Path where to save output inpainted MHA file')
    generate_parser.add_argument('--device', type=str, default=None,
                       help='Device to use for inference (cuda or cpu). Uses cuda if available by default.')
    
    # Zpracujeme argumenty
    args = parser.parse_args()
    
    # Zpracování příkazu "train"
    if args.command == 'train':
        print(f"Starting training with the following parameters:")
        print(f"ZADC directory: {args.zadc_dir}")
        print(f"Label directory: {args.label_dir}")
        print(f"Synthetic lesions directory: {args.synthetic_lesions_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Save interval: Every {args.save_interval} epochs")
        
        # Trénujeme model s argumenty z příkazové řádky
        gan_model = train(
            zadc_dir=args.zadc_dir,
            label_dir=args.label_dir,
            synthetic_lesions_dir=args.synthetic_lesions_dir,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            save_interval=args.save_interval
        )
        
        print(f"Training completed. Running inference on test patient {args.test_patient}")
        
        # Provádění inference s natrénovaným modelem na testovacím pacientovi
        zadc_file = os.path.join(args.zadc_dir, f"Zmap_{args.test_patient}-ADC_smooth2mm_clipped10.mha")
        synthetic_lesion_file = os.path.join(args.synthetic_lesions_dir, args.test_patient, args.test_lesion)
        output_file = os.path.join(args.output_dir, f"{args.test_patient}_inpainted.mha")
        
        apply_inpainting(gan_model, zadc_file, synthetic_lesion_file, output_file)
        print(f"Inference completed. Output saved to {output_file}")
    
    # Zpracování příkazu "generate"
    elif args.command == 'generate':
        print(f"Running inference with the following parameters:")
        print(f"Model checkpoint: {args.model_path}")
        print(f"ZADC file: {args.zadc_file}")
        print(f"Synthetic lesion file: {args.lesion_file}")
        print(f"Output file: {args.output_file}")
        
        # Inicializace modelu
        device = args.device
        model = LesionInpaintingGAN(device=device)
        
        # Načtení modelu z checkpointu
        print(f"Loading model from checkpoint: {args.model_path}")
        epoch = model.load_models(args.model_path)
        print(f"Loaded model from epoch {epoch}")
        
        # Inference - aplikace inpainting
        print(f"Running inference...")
        apply_inpainting(model, args.zadc_file, args.lesion_file, args.output_file)
        print(f"Inference completed. Output saved to {args.output_file}")
    
    else:
        parser.print_help()
