import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from monai.networks.nets import SwinUNETR
from monai.losses import FocalLoss
import glob
from collections import defaultdict
from torch.amp import GradScaler, autocast
import argparse


class HieLesionDataset(Dataset):
    def __init__(self, 
                 lesion_dir, 
                 lesion_atlas_path, 
                 transform=None, 
                 filter_empty=True,
                 min_non_zero_percentage=0.00001):
        """
        Dataset for HIE lesion synthesis
        
        Args:
            lesion_dir (str): Directory with lesion .nii files
            lesion_atlas_path (str): Path to the lesion frequency atlas
            transform (callable, optional): Transform to apply to the data
            filter_empty (bool): Whether to filter out images with no lesions
            min_non_zero_percentage (float): Minimum percentage of non-zero voxels to include sample
        """
        self.lesion_dir = lesion_dir
        self.transform = transform
        self.filter_empty = filter_empty
        self.min_non_zero_percentage = min_non_zero_percentage
        
        # Load the lesion atlas (frequency atlas)
        self.lesion_atlas = nib.load(lesion_atlas_path).get_fdata()
        assert self.lesion_atlas.shape == (128, 128, 64), f"Lesion atlas shape is {self.lesion_atlas.shape}, expected (128, 128, 64)"
        
        # Normalize the lesion atlas to [0, 1]
        self.lesion_atlas = self.lesion_atlas / np.max(self.lesion_atlas)
        
        # Get all lesion files
        self.lesion_files = sorted(glob.glob(os.path.join(lesion_dir, "*lesion.nii*")))
        print(f"Found {len(self.lesion_files)} lesion files")
        
        # Filter out empty lesion files if needed
        if self.filter_empty:
            self._filter_empty_lesions()
            print(f"After filtering, {len(self.lesion_files)} lesion files remain")
    
    def _filter_empty_lesions(self):
        """Filter out lesion files that are completely empty or have too few non-zero voxels"""
        valid_files = []
        
        for lesion_file in self.lesion_files:
            # Load the lesion
            lesion = nib.load(lesion_file).get_fdata()
            
            # Check if the lesion has any non-zero voxels
            non_zero_percentage = np.count_nonzero(lesion) / lesion.size
            
            if non_zero_percentage >= self.min_non_zero_percentage:
                valid_files.append(lesion_file)
            else:
                print(f"Filtering out {lesion_file} with {non_zero_percentage*100:.4f}% non-zero voxels")
        
        self.lesion_files = valid_files
    
    def __len__(self):
        return len(self.lesion_files)
    
    def __getitem__(self, idx):
        # Load the lesion
        lesion_file = self.lesion_files[idx]
        lesion = nib.load(lesion_file).get_fdata()
        
        # Binarize the lesion if it's not already binary (just in case)
        lesion = (lesion > 0).astype(np.float32)
        
        # Convert to torch tensors
        lesion_tensor = torch.from_numpy(lesion).float()
        atlas_tensor = torch.from_numpy(self.lesion_atlas).float()
        
        # Add channel dimension
        lesion_tensor = lesion_tensor.unsqueeze(0)
        atlas_tensor = atlas_tensor.unsqueeze(0)
        
        # Apply transform if provided
        if self.transform:
            lesion_tensor, atlas_tensor = self.transform(lesion_tensor, atlas_tensor)
        
        return {
            'atlas': atlas_tensor,  # Input: frequency atlas
            'lesion': lesion_tensor  # Ground truth: binary lesion mask
        }


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_size=24, dropout_rate=0.0):
        """
        Generator based on SwinUNETR for lesion synthesis
        
        Args:
            in_channels (int): Number of input channels (1 for lesion atlas)
            out_channels (int): Number of output channels (1 for binary lesion mask)
            feature_size (int): Feature size for SwinUNETR
            dropout_rate (float): Dropout rate
        """
        super(Generator, self).__init__()
        
        # SwinUNETR as the backbone
        self.swin_unetr = SwinUNETR(
            img_size=(128, 128, 64),
            in_channels=in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            drop_rate=dropout_rate,
            use_checkpoint=True,
        )
        
        # Additional layers to get binary output
        self.final_conv = nn.Conv3d(feature_size, out_channels, kernel_size=1)
    
    def forward(self, x):
        # SwinUNETR features
        features = self.swin_unetr(x)
        
        # Final layers
        x = self.final_conv(features)
        
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=2, feature_size=24, depth=4):
        """
        Discriminator for lesion GAN
        
        Args:
            in_channels (int): Number of input channels (atlas + lesion = 2)
            feature_size (int): Initial feature size
            depth (int): Number of downsampling layers
        """
        super(Discriminator, self).__init__()
        
        # Initial convolution
        layers = [
            nn.Conv3d(in_channels, feature_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Downsampling layers
        current_feature_size = feature_size
        for _ in range(depth - 1):
            layers += [
                nn.Conv3d(current_feature_size, current_feature_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(current_feature_size * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            current_feature_size *= 2
        
        # Final layers
        layers += [
            nn.Conv3d(current_feature_size, 1, kernel_size=4, stride=1, padding=1, bias=False)
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, lesion_atlas, lesion):
        # Concatenate atlas and lesion along channel dimension
        x = torch.cat([lesion_atlas, lesion], dim=1)
        return self.model(x)


class SwinGAN(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 feature_size=24, 
                 dropout_rate=0.0,
                 lambda_focal=10.0,
                 lambda_l1=5.0,
                 focal_alpha=0.75,
                 focal_gamma=2.0):
        """
        SwinGAN for lesion synthesis
        
        Args:
            in_channels (int): Number of input channels for generator
            out_channels (int): Number of output channels for generator
            feature_size (int): Feature size for both generator and discriminator
            dropout_rate (float): Dropout rate for generator
            lambda_focal (float): Weight for focal loss
            lambda_l1 (float): Weight for L1 loss
            focal_alpha (float): Alpha parameter for focal loss
            focal_gamma (float): Gamma parameter for focal loss
        """
        super(SwinGAN, self).__init__()
        
        self.generator = Generator(in_channels, out_channels, feature_size, dropout_rate)
        self.discriminator = Discriminator(in_channels + out_channels, feature_size)
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # Loss weights
        self.lambda_focal = lambda_focal
        self.lambda_l1 = lambda_l1
    
    def generator_loss(self, fake_lesion, real_lesion, atlas, fake_pred):
        """
        Generator loss computation
        
        Args:
            fake_lesion: Generated lesion
            real_lesion: Ground truth lesion
            atlas: Lesion atlas (frequency map)
            fake_pred: Discriminator prediction on fake lesion
        """
        # Adversarial loss (BCEWithLogitsLoss automatically applies sigmoid)
        adv_loss = self.bce_loss(fake_pred, torch.ones_like(fake_pred))
        
        # Apply sigmoid to fake_lesion for other losses since we removed it from generator
        fake_lesion_sigmoid = torch.sigmoid(fake_lesion)
        
        # Focal loss for sparse lesion segmentation
        focal_loss = self.focal_loss(fake_lesion, real_lesion)
        
        # L1 loss (can help with spatial consistency)
        l1_loss = self.l1_loss(fake_lesion_sigmoid, real_lesion)
        
        # Constraint loss: ensure lesions only appear in regions with non-zero atlas values
        constraint_mask = (atlas > 0).float()
        lesion_outside_mask = fake_lesion_sigmoid * (1 - constraint_mask)
        constraint_loss = torch.mean(lesion_outside_mask) * 100.0  # Heavy penalty
        
        # Total generator loss
        total_loss = adv_loss + self.lambda_focal * focal_loss + self.lambda_l1 * l1_loss + constraint_loss
        
        return total_loss, {
            'adv_loss': adv_loss.item(),
            'focal_loss': focal_loss.item(),
            'l1_loss': l1_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'total_g_loss': total_loss.item()
        }
    
    def discriminator_loss(self, real_pred, fake_pred):
        """
        Discriminator loss computation
        
        Args:
            real_pred: Discriminator prediction on real lesion
            fake_pred: Discriminator prediction on fake lesion
        """
        # Loss on real samples (BCEWithLogitsLoss automatically applies sigmoid)
        real_loss = self.bce_loss(real_pred, torch.ones_like(real_pred))
        
        # Loss on fake samples
        fake_loss = self.bce_loss(fake_pred, torch.zeros_like(fake_pred))
        
        # Total discriminator loss
        total_loss = (real_loss + fake_loss) * 0.5
        
        return total_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'total_d_loss': total_loss.item()
        }


class SwinGANTrainer:
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader=None,
                 lr_g=0.0002,
                 lr_d=0.0001,
                 betas=(0.5, 0.999),
                 device='cuda',
                 output_dir='./output',
                 use_amp=True,
                 generator_save_interval=4):
        """
        Trainer for SwinGAN
        
        Args:
            model: SwinGAN model
            train_loader: Training data loader
            val_loader: Validation data loader
            lr_g: Learning rate for generator
            lr_d: Learning rate for discriminator
            betas: Adam optimizer betas
            device: Device to use
            output_dir: Output directory for saving models and samples
            use_amp: Whether to use automatic mixed precision
            generator_save_interval: Interval for saving generator-only checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.generator_save_interval = generator_save_interval
        
        # Initialize optimizers
        self.optimizer_g = torch.optim.Adam(
            self.model.generator.parameters(), lr=lr_g, betas=betas
        )
        self.optimizer_d = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr_d, betas=betas
        )
        
        # For mixed precision training
        if self.use_amp:
            self.scaler_g = GradScaler('cuda' if device == 'cuda' else 'cpu')
            self.scaler_d = GradScaler('cuda' if device == 'cuda' else 'cpu')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create generator checkpoints directory
        self.generator_dir = os.path.join(self.output_dir, 'generator_checkpoints')
        os.makedirs(self.generator_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
    
    def train(self, epochs, save_interval=5, val_interval=1):
        """
        Train the SwinGAN model
        
        Args:
            epochs: Number of epochs to train
            save_interval: Interval for saving full model checkpoints
            val_interval: Interval for running validation
        """
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = defaultdict(float)
            
            # Training loop
            for batch_idx, batch in enumerate(self.train_loader):
                atlas = batch['atlas'].to(self.device)
                real_lesion = batch['lesion'].to(self.device)
                
                # Train discriminator
                self.optimizer_d.zero_grad()
                
                with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_amp):
                    # Generate fake lesions
                    fake_lesion = self.model.generator(atlas)
                    
                    # Compute discriminator predictions
                    real_pred = self.model.discriminator(atlas, real_lesion)
                    fake_pred = self.model.discriminator(atlas, fake_lesion.detach())
                    
                    # Compute discriminator loss
                    d_loss, d_losses_dict = self.model.discriminator_loss(real_pred, fake_pred)
                
                # Update discriminator
                if self.use_amp:
                    self.scaler_d.scale(d_loss).backward()
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                else:
                    d_loss.backward()
                    self.optimizer_d.step()
                
                # Train generator
                self.optimizer_g.zero_grad()
                
                with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_amp):
                    # Generate fake lesions again (since gradients were detached)
                    fake_lesion = self.model.generator(atlas)
                    
                    # Compute discriminator prediction on fake lesions
                    fake_pred = self.model.discriminator(atlas, fake_lesion)
                    
                    # Compute generator loss
                    g_loss, g_losses_dict = self.model.generator_loss(
                        fake_lesion, real_lesion, atlas, fake_pred
                    )
                
                # Update generator
                if self.use_amp:
                    self.scaler_g.scale(g_loss).backward()
                    self.scaler_g.step(self.optimizer_g)
                    self.scaler_g.update()
                else:
                    g_loss.backward()
                    self.optimizer_g.step()
                
                # Update epoch losses
                for k, v in d_losses_dict.items():
                    epoch_losses[k] += v
                for k, v in g_losses_dict.items():
                    epoch_losses[k] += v
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(self.train_loader)} | "
                          f"G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f}")
            
            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= len(self.train_loader)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"G Loss: {epoch_losses['total_g_loss']:.4f} | "
                  f"D Loss: {epoch_losses['total_d_loss']:.4f}")
            
            # Save full model checkpoint
            if (epoch + 1) % save_interval == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                    'optimizer_d_state_dict': self.optimizer_d.state_dict(),
                    'losses': epoch_losses
                }, os.path.join(self.output_dir, f'swin_gan_epoch_{epoch+1}.pt'))
            
            # Save generator-only checkpoint every generator_save_interval epochs
            if (epoch + 1) % self.generator_save_interval == 0:
                generator_path = os.path.join(self.generator_dir, f'generator_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'generator_state_dict': self.model.generator.state_dict(),
                    'focal_loss_weight': self.model.lambda_focal,
                    'l1_loss_weight': self.model.lambda_l1,
                    'focal_alpha': self.model.focal_loss.alpha,
                    'focal_gamma': self.model.focal_loss.gamma
                }, generator_path)
                print(f"Saved generator-only checkpoint to {generator_path}")
            
            # Validation
            if self.val_loader is not None and (epoch + 1) % val_interval == 0:
                self.validate(epoch)
    
    def validate(self, epoch):
        """
        Validate the model
        
        Args:
            epoch: Current epoch
        """
        self.model.eval()
        val_losses = defaultdict(float)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                atlas = batch['atlas'].to(self.device)
                real_lesion = batch['lesion'].to(self.device)
                
                # Use autocast for validation as well for consistency
                with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_amp):
                    # Generate fake lesions
                    fake_lesion = self.model.generator(atlas)
                    
                    # Compute discriminator predictions
                    real_pred = self.model.discriminator(atlas, real_lesion)
                    fake_pred = self.model.discriminator(atlas, fake_lesion)
                    
                    # Compute losses
                    g_loss, g_losses_dict = self.model.generator_loss(
                        fake_lesion, real_lesion, atlas, fake_pred
                    )
                    d_loss, d_losses_dict = self.model.discriminator_loss(real_pred, fake_pred)
                
                # Update validation losses
                for k, v in d_losses_dict.items():
                    val_losses[k] += v
                for k, v in g_losses_dict.items():
                    val_losses[k] += v
        
        # Average losses
        for k in val_losses:
            val_losses[k] /= len(self.val_loader)
        
        # Print validation summary
        print(f"Validation | Epoch {epoch+1} | "
              f"G Loss: {val_losses['total_g_loss']:.4f} | "
              f"D Loss: {val_losses['total_d_loss']:.4f}")


def train_model(args):
    """
    Train the SwinGAN model with the provided arguments
    
    Args:
        args: Command line arguments
    """
    print("=== Training SwinGAN model ===")
    print(f"Lesion directory: {args.lesion_dir}")
    print(f"Lesion atlas: {args.lesion_atlas}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate (generator): {args.lr_g}")
    print(f"Learning rate (discriminator): {args.lr_d}")
    print(f"Feature size: {args.feature_size}")
    print(f"Non-zero threshold: {args.min_non_zero}")
    print(f"Device: {args.device}")
    print(f"Generator save interval: {args.generator_save_interval}")
    
    # Create dataset
    train_dataset = HieLesionDataset(
        lesion_dir=args.lesion_dir,
        lesion_atlas_path=args.lesion_atlas,
        filter_empty=True,
        min_non_zero_percentage=args.min_non_zero
    )
    
    # Create validation dataset if provided
    val_loader = None
    if args.val_lesion_dir:
        print(f"Validation lesion directory: {args.val_lesion_dir}")
        val_dataset = HieLesionDataset(
            lesion_dir=args.val_lesion_dir,
            lesion_atlas_path=args.lesion_atlas,
            filter_empty=True,
            min_non_zero_percentage=args.min_non_zero
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = SwinGAN(
        in_channels=1,
        out_channels=1,
        feature_size=args.feature_size,
        dropout_rate=args.dropout_rate,
        lambda_focal=args.lambda_focal,
        lambda_l1=args.lambda_l1,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    
    # Create trainer
    trainer = SwinGANTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        device=args.device,
        output_dir=args.output_dir,
        use_amp=not args.disable_amp,
        generator_save_interval=args.generator_save_interval
    )
    
    # Train model
    trainer.train(epochs=args.epochs, save_interval=args.save_interval)
    
    print(f"Training completed. Model saved to {args.output_dir}")
    print(f"Generator checkpoints saved to {trainer.generator_dir}")


def generate_lesions(args):
    """
    Generate synthetic lesions with the trained model
    
    Args:
        args: Command line arguments
    """
    print("=== Generating synthetic lesions ===")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Lesion atlas: {args.lesion_atlas}")
    print(f"Output file: {args.output_file}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")
    
    # Load atlas
    atlas_nii = nib.load(args.lesion_atlas)
    atlas = atlas_nii.get_fdata()
    
    # Normalize the atlas to [0, 1]
    atlas = atlas / np.max(atlas)
    
    # Create model
    model = SwinGAN(
        in_channels=1,
        out_channels=1,
        feature_size=args.feature_size
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_checkpoint, map_location=args.device)
    
    # Check if it's a full model checkpoint or a generator-only checkpoint
    if 'model_state_dict' in checkpoint:
        # Full model checkpoint
        print("Loading full model checkpoint...")
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'generator_state_dict' in checkpoint:
        # Generator-only checkpoint
        print("Loading generator-only checkpoint...")
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        # Optionally restore loss parameters if available
        if 'focal_loss_weight' in checkpoint:
            model.lambda_focal = checkpoint['focal_loss_weight']
            print(f"  - Restored focal loss weight: {model.lambda_focal}")
        if 'l1_loss_weight' in checkpoint:
            model.lambda_l1 = checkpoint['l1_loss_weight']
            print(f"  - Restored L1 loss weight: {model.lambda_l1}")
    else:
        raise ValueError("Invalid checkpoint format. Doesn't contain model_state_dict or generator_state_dict.")
    
    model.to(args.device)
    model.eval()
    
    # Convert atlas to tensor
    atlas_tensor = torch.from_numpy(atlas).float().unsqueeze(0).unsqueeze(0).to(args.device)
    
    # Generate lesions
    with torch.no_grad():
        # Get raw outputs from generator
        logits = model.generator(atlas_tensor)
        
        # Apply sigmoid to get probability map
        fake_lesion = torch.sigmoid(logits)
        
        # Binarize the output
        binary_lesion = (fake_lesion > args.threshold).float()
        
        # Ensure lesions only appear in regions with non-zero atlas values
        constraint_mask = (atlas_tensor > 0).float()
        binary_lesion = binary_lesion * constraint_mask
    
    # Convert to numpy and save
    lesion_np = binary_lesion.cpu().squeeze().numpy()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Save as NIfTI
    lesion_nii = nib.Nifti1Image(lesion_np.astype(np.uint8), atlas_nii.affine)
    nib.save(lesion_nii, args.output_file)
    
    print(f"Synthetic lesion generated and saved to {args.output_file}")
    
    # Print statistics
    non_zero_percentage = np.count_nonzero(lesion_np) / lesion_np.size * 100
    print(f"Lesion statistics:")
    print(f"  - Non-zero voxels: {np.count_nonzero(lesion_np)}")
    print(f"  - Non-zero percentage: {non_zero_percentage:.4f}%")
    
    # Calculate number of connected components (distinct lesions)
    try:
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(lesion_np)
        print(f"  - Number of distinct lesions: {num_features}")
    except ImportError:
        print("  - Install scipy to calculate the number of distinct lesions")


def main():
    parser = argparse.ArgumentParser(description='SwinGAN for HIE Lesion Synthesis')
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')
    
    # Training subparser
    train_parser = subparsers.add_parser('train', help='Train the SwinGAN model')
    train_parser.add_argument('--lesion_dir', type=str, required=True, 
                             help='Directory containing lesion .nii files for training')
    train_parser.add_argument('--lesion_atlas', type=str, required=True, 
                             help='Path to the lesion frequency atlas .nii file')
    train_parser.add_argument('--output_dir', type=str, default='./swin_gan_output', 
                             help='Output directory for saving model checkpoints')
    train_parser.add_argument('--val_lesion_dir', type=str, default=None, 
                             help='Directory containing lesion .nii files for validation')
    train_parser.add_argument('--batch_size', type=int, default=2, 
                             help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=100, 
                             help='Number of epochs to train')
    train_parser.add_argument('--lr_g', type=float, default=0.0002, 
                             help='Learning rate for generator')
    train_parser.add_argument('--lr_d', type=float, default=0.0001, 
                             help='Learning rate for discriminator')
    train_parser.add_argument('--feature_size', type=int, default=24, 
                             help='Feature size for SwinUNETR')
    train_parser.add_argument('--dropout_rate', type=float, default=0.0, 
                             help='Dropout rate for generator')
    train_parser.add_argument('--lambda_focal', type=float, default=10.0, 
                             help='Weight for focal loss')
    train_parser.add_argument('--lambda_l1', type=float, default=5.0, 
                             help='Weight for L1 loss')
    train_parser.add_argument('--focal_alpha', type=float, default=0.75, 
                             help='Alpha parameter for focal loss')
    train_parser.add_argument('--focal_gamma', type=float, default=2.0, 
                             help='Gamma parameter for focal loss')
    train_parser.add_argument('--min_non_zero', type=float, default=0.000001, 
                             help='Minimum percentage of non-zero voxels to include sample')
    train_parser.add_argument('--save_interval', type=int, default=5, 
                             help='Interval for saving full model checkpoints')
    train_parser.add_argument('--generator_save_interval', type=int, default=4, 
                             help='Interval for saving generator-only checkpoints')
    train_parser.add_argument('--num_workers', type=int, default=4, 
                             help='Number of workers for data loading')
    train_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                             help='Device to use (cuda or cpu)')
    train_parser.add_argument('--disable_amp', action='store_true', 
                             help='Disable automatic mixed precision training')
    
    # Generation subparser
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic lesions')
    gen_parser.add_argument('--model_checkpoint', type=str, required=True, 
                           help='Path to the trained model checkpoint')
    gen_parser.add_argument('--lesion_atlas', type=str, required=True, 
                           help='Path to the lesion frequency atlas .nii file')
    gen_parser.add_argument('--output_file', type=str, required=True, 
                           help='Path to save the generated lesion .nii file')
    gen_parser.add_argument('--threshold', type=float, default=0.5, 
                           help='Threshold for binarizing the generated lesions')
    gen_parser.add_argument('--feature_size', type=int, default=24, 
                           help='Feature size for SwinUNETR (must match training)')
    gen_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                           help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Run the appropriate mode
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'generate':
        generate_lesions(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
