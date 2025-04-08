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
    1. ADC maps without lesions in a specific area
    2. Binary masks for synthetic lesions
    3. Ground truth ADC maps with lesions (for validation)
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
        
        # Get all ZADC files
        self.zadc_files = sorted(glob.glob(os.path.join(zadc_dir, "*.mha")))
        
        # Filter for patients with corresponding synthetic lesions
        valid_patients = []
        for zadc_file in self.zadc_files:
            patient_id = os.path.basename(zadc_file).split('-')[0] + '-' + os.path.basename(zadc_file).split('-')[1]
            if os.path.exists(os.path.join(synthetic_lesions_dir, patient_id)):
                valid_patients.append(patient_id)
        
        self.valid_patients = valid_patients
        
        # Split into train/val
        if mode == 'train':
            self.patients = self.valid_patients[:int(0.8 * len(self.valid_patients))]
        else:
            self.patients = self.valid_patients[int(0.8 * len(self.valid_patients)):]
        
        # Create patient-synthetic lesion pairs
        self.samples = []
        for patient in self.patients:
            zadc_file = glob.glob(os.path.join(zadc_dir, f"Zmap_{patient}*.mha"))[0]
            label_file = glob.glob(os.path.join(label_dir, f"{patient}*lesion.mha"))[0]
            synthetic_lesions = glob.glob(os.path.join(synthetic_lesions_dir, patient, "*.mha"))
            
            for syn_lesion in synthetic_lesions:
                self.samples.append({
                    'zadc': zadc_file,
                    'label': label_file,
                    'synthetic_lesion': syn_lesion
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load ZADC map
        zadc_img = sitk.ReadImage(sample['zadc'])
        zadc_array = sitk.GetArrayFromImage(zadc_img)
        
        # Load original lesion mask
        label_img = sitk.ReadImage(sample['label'])
        label_array = sitk.GetArrayFromImage(label_img)
        
        # Load synthetic lesion
        syn_lesion_img = sitk.ReadImage(sample['synthetic_lesion'])
        syn_lesion_array = sitk.GetArrayFromImage(syn_lesion_img)
        
        # Normalize ZADC to [0, 1]
        zadc_array = (zadc_array - zadc_array.min()) / (zadc_array.max() - zadc_array.min() + 1e-8)
        
        # Convert label and synthetic lesion to binary
        label_array = (label_array > 0).astype(np.float32)
        syn_lesion_array = (syn_lesion_array > 0).astype(np.float32)
        
        # Create "healthy" brain - masking out any lesions
        # We'll create a version of the brain where lesions are "erased" through inpainting
        healthy_brain = zadc_array.copy()
        
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
            healthy_patch = healthy_brain[z_start:z_end, y_start:y_end, x_start:x_end]
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
            zadc_tensor = torch.from_numpy(zadc_patch).float().unsqueeze(0)  # Add channel dimension
            healthy_tensor = torch.from_numpy(healthy_patch).float().unsqueeze(0)
            label_tensor = torch.from_numpy(label_patch).float().unsqueeze(0)
            syn_lesion_tensor = torch.from_numpy(syn_lesion_patch).float().unsqueeze(0)
            
            # Combined input: healthy brain and synthetic lesion mask
            input_tensor = torch.cat([healthy_tensor, syn_lesion_tensor], dim=0)
            
            return {
                'input': input_tensor,  # Healthy brain + synthetic lesion mask
                'target': zadc_tensor,   # Original ZADC with real lesions
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
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
            
            # Visualize some results
            # Get a sample from validation set
            if val_loader is not None:
                with torch.no_grad():
                    val_sample = next(iter(val_loader))
                    input_data = val_sample['input'].to(gan_model.device)
                    real_brain = val_sample['target'].to(gan_model.device)
                    synthetic_mask = val_sample['synthetic_mask'].to(gan_model.device)
                    
                    # Generate inpainted image
                    fake_brain = gan_model.generator(input_data)
                    
                    # Save middle slices
                    for i in range(min(4, batch_size)):
                        slice_idx = fake_brain.shape[2] // 2
                        
                        # Extract middle slices
                        input_slice = input_data[i, 0, slice_idx].cpu().numpy()
                        mask_slice = synthetic_mask[i, 0, slice_idx].cpu().numpy()
                        real_slice = real_brain[i, 0, slice_idx].cpu().numpy()
                        fake_slice = fake_brain[i, 0, slice_idx].cpu().numpy()
                        
                        # Create figure
                        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                        
                        # Plot images
                        axes[0, 0].imshow(input_slice, cmap='gray')
                        axes[0, 0].set_title('Input (Healthy Brain)')
                        axes[0, 0].axis('off')
                        
                        axes[0, 1].imshow(mask_slice, cmap='gray')
                        axes[0, 1].set_title('Synthetic Lesion Mask')
                        axes[0, 1].axis('off')
                        
                        axes[1, 0].imshow(real_slice, cmap='gray')
                        axes[1, 0].set_title('Real Brain')
                        axes[1, 0].axis('off')
                        
                        axes[1, 1].imshow(fake_slice, cmap='gray')
                        axes[1, 1].set_title('Generated Brain with Lesion')
                        axes[1, 1].axis('off')
                        
                        plt.savefig(os.path.join(output_dir, f'visualization_epoch{epoch+1}_sample{i}.png'))
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
