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
from scipy import ndimage as measure
from torchvision import transforms


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
    def __init__(self, in_channels=1, out_channels=1, feature_size=24, dropout_rate=0.0, use_noise=True, noise_dim=16):
        """
        Generator based on SwinUNETR for lesion synthesis
        
        Args:
            in_channels (int): Number of input channels (1 for lesion atlas)
            out_channels (int): Number of output channels (1 for binary lesion mask)
            feature_size (int): Feature size for SwinUNETR
            dropout_rate (float): Dropout rate
            use_noise (bool): Whether to use noise injection for diversity
            noise_dim (int): Dimension of the noise vector to inject
        """
        super(Generator, self).__init__()
        
        self.use_noise = use_noise
        self.noise_dim = noise_dim
        self.feature_size = feature_size  # Uložíme hodnotu i jako atribut
        self.dropout_rate = dropout_rate  # Uložíme hodnotu i jako atribut
        
        # Adjust input channels if using noise
        actual_in_channels = in_channels
        if use_noise:
            actual_in_channels += 1  # Add one channel for noise
            
            # Noise processing network
            self.noise_processor = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(8, 1, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # SwinUNETR as the backbone
        self.swin_unetr = SwinUNETR(
            img_size=(128, 128, 64),
            in_channels=actual_in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            drop_rate=dropout_rate,
            use_checkpoint=True,
        )
        
        # Additional layers to get binary output
        self.final_conv = nn.Conv3d(feature_size, out_channels, kernel_size=1)
    
    def forward(self, x, noise=None):
        """
        Forward pass with optional noise input
        
        Args:
            x: Input tensor (lesion atlas)
            noise: Optional noise tensor. If None and use_noise is True, random noise will be generated
        
        Returns:
            Generated lesion mask
        """
        # Process noise if using it
        if self.use_noise:
            batch_size, _, D, H, W = x.shape
            
            # Generate random noise if not provided
            if noise is None:
                # Generujeme šum jako 1D vektor
                noise = torch.randn(batch_size, self.noise_dim, device=x.device)
                
            # Reshape noise to 5D tensor [batch_size, 1, D, H, W]
            # Nejprve rozšíříme šum na správnou délku
            if noise.dim() == 2:  # Pokud má noise tvar [batch_size, noise_dim]
                # Rozšíříme noise_dim na D*H*W
                expanded_noise = noise.view(batch_size, self.noise_dim, 1, 1, 1).expand(batch_size, self.noise_dim, D, H, W)
                # Vezmeme pouze jednu dimenzi (kanál) pro noise_processor
                noise_3d = expanded_noise[:, 0:1, :, :, :]  # [batch_size, 1, D, H, W]
            else:
                # Předpokládáme, že noise už má správný tvar
                noise_3d = noise
            
            # Process noise through a small network to make it more structured
            processed_noise = self.noise_processor(noise_3d)
            
            # Concatenate processed noise with input along channel dimension
            x = torch.cat([x, processed_noise], dim=1)
        
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
                 lambda_fragmentation=50.0,
                 focal_alpha=0.75,
                 focal_gamma=2.0,
                 use_noise=True,
                 noise_dim=16,
                 fragmentation_kernel_size=5):
        """
        SwinGAN for lesion synthesis
        
        Args:
            in_channels (int): Number of input channels for generator
            out_channels (int): Number of output channels for generator
            feature_size (int): Feature size for both generator and discriminator
            dropout_rate (float): Dropout rate for generator
            lambda_focal (float): Weight for focal loss
            lambda_l1 (float): Weight for L1 loss
            lambda_fragmentation (float): Weight for fragmentation loss - higher values promote more coherent lesions
            focal_alpha (float): Alpha parameter for focal loss
            focal_gamma (float): Gamma parameter for focal loss
            use_noise (bool): Whether to use noise injection for diversity
            noise_dim (int): Dimension of the noise vector to inject
            fragmentation_kernel_size (int): Size of kernel used for fragmentation loss
        """
        super(SwinGAN, self).__init__()
        
        self.generator = Generator(
            in_channels=in_channels, 
            out_channels=out_channels, 
            feature_size=feature_size, 
            dropout_rate=dropout_rate,
            use_noise=use_noise,
            noise_dim=noise_dim
        )
        self.discriminator = Discriminator(in_channels + out_channels, feature_size)
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # Loss weights
        self.lambda_focal = lambda_focal
        self.lambda_l1 = lambda_l1
        self.lambda_fragmentation = lambda_fragmentation
        
        # For generating diverse samples
        self.use_noise = use_noise
        self.noise_dim = noise_dim
        
        # For fragmentation loss
        self.fragmentation_kernel_size = fragmentation_kernel_size
    
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
        
        # Nová komponenta ztráty pro podporu celistvosti lézí
        # Použijeme 3D konvoluci s gaussovským kernelem pro vyhlazení lézí
        # Toto penalizuje fragmentaci a podporuje generování souvislých lézí
        batch_size = fake_lesion_sigmoid.size(0)
        
        # Vytvoření 3D gaussovského kernelu pro vyhlazení
        kernel_size = self.fragmentation_kernel_size
        sigma = 1.0
        
        # Středový bod kernelu
        center = kernel_size // 2
        
        # Vytvoříme 3D kernel
        kernel = torch.zeros((1, 1, kernel_size, kernel_size, kernel_size), device=fake_lesion.device)
        
        # Naplníme kernel gaussovskými hodnotami
        for x in range(kernel_size):
            for y in range(kernel_size):
                for z in range(kernel_size):
                    # 3D Gaussovská funkce - převedeme výpočet na tensor
                    exponent = -((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) / (2 * sigma ** 2)
                    kernel[0, 0, x, y, z] = torch.exp(torch.tensor(exponent, device=fake_lesion.device))
        
        # Normalizace kernelu
        kernel = kernel / kernel.sum()
        
        # Aplikace kernelu na fake_lesion pro získání vyhlazeného obrazu
        # padding=center zajistí, že výstup bude mít stejnou velikost jako vstup
        smoothed_lesion = F.conv3d(
            fake_lesion_sigmoid, kernel, padding=center
        )
        
        # Ztráta fragmentace - chceme, aby model preferoval podobné hodnoty sousedních voxelů
        # Tím podporujeme vytváření celistvých struktur
        fragmentation_loss = torch.mean(
            torch.abs(fake_lesion_sigmoid - smoothed_lesion)
        ) * self.lambda_fragmentation  # Použijeme parametr lambda_fragmentation
        
        # Total generator loss včetně nové komponenty
        total_loss = adv_loss + self.lambda_focal * focal_loss + self.lambda_l1 * l1_loss + constraint_loss + fragmentation_loss
        
        return total_loss, {
            'adv_loss': adv_loss.item(),
            'focal_loss': focal_loss.item(),
            'l1_loss': l1_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'fragmentation_loss': fragmentation_loss.item(),  # Přidání nové ztráty do statistik
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

    def forward(self, atlas, noise=None):
        """
        Forward pass of the SwinGAN
        
        Args:
            atlas: Lesion atlas (frequency map)
            noise: Optional noise tensor for generating diverse samples
            
        Returns:
            Generated lesion logits
        """
        # Forward pass through the generator
        fake_lesion = self.generator(atlas, noise)
        return fake_lesion


class SwinGANTrainer:
    def __init__(self, 
                 model,
                 optimizer_g,
                 optimizer_d,
                 device='cuda',
                 output_dir='./output',
                 use_amp=True,
                 generator_save_interval=4):
        """
        Trainer for SwinGAN
        
        Args:
            model: SwinGAN model
            optimizer_g: Optimizer for generator
            optimizer_d: Optimizer for discriminator
            device: Device to use
            output_dir: Output directory for saving models and samples
            use_amp: Whether to use automatic mixed precision
            generator_save_interval: Interval for saving generator-only checkpoints
        """
        self.model = model
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.generator_save_interval = generator_save_interval
        
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
    
    def train(self, dataloader, epochs, val_dataloader=None, save_interval=5):
        """
        Train the SwinGAN model
        
        Args:
            dataloader: Training data loader
            epochs: Number of epochs to train
            val_dataloader: Validation data loader
            save_interval: Interval for saving full model checkpoints
        """
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = defaultdict(float)
            
            # Training loop
            for batch_idx, batch in enumerate(dataloader):
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
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                          f"G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f}")
            
            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= len(dataloader)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"G Loss: {epoch_losses['total_g_loss']:.4f} | "
                  f"D Loss: {epoch_losses['total_d_loss']:.4f}")
            
            # Save full model checkpoint at save_interval
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
                    'focal_gamma': self.model.focal_loss.gamma,
                    'use_noise': self.model.use_noise,
                    'noise_dim': self.model.noise_dim,
                    'feature_size': self.model.generator.feature_size,
                    'dropout_rate': self.model.generator.dropout_rate
                }, generator_path)
                print(f"Saved generator-only checkpoint to {generator_path}")
                print(f"Checkpoint includes configuration for use_noise={self.model.use_noise}, noise_dim={self.model.noise_dim}")
            
            # Validation
            if val_dataloader is not None and (epoch + 1) % 1 == 0:
                self.validate(epoch, val_dataloader)
    
    def validate(self, epoch, val_dataloader):
        """
        Validate the model
        
        Args:
            epoch: Current epoch
            val_dataloader: Validation data loader
        """
        self.model.eval()
        val_losses = defaultdict(float)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
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
            val_losses[k] /= len(val_dataloader)
        
        # Print validation summary
        print(f"Validation | Epoch {epoch+1} | "
              f"G Loss: {val_losses['total_g_loss']:.4f} | "
              f"D Loss: {val_losses['total_d_loss']:.4f}")


def train_model(args):
    """
    Train the SwinGAN model
    
    Args:
        args: Command line arguments
    """
    print("=== Training SwinGAN model ===")
    print(f"Lesion directory: {args.lesion_dir}")
    print(f"Atlas file: {args.lesion_atlas}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num epochs: {args.epochs}")
    print(f"Learning rate (generator): {args.lr_g}")
    print(f"Learning rate (discriminator): {args.lr_d}")
    print(f"Feature size: {args.feature_size}")
    print(f"Non-zero threshold: {args.min_non_zero}")
    print(f"Device: {args.device}")
    print(f"Generator save interval: {args.generator_save_interval}")
    print(f"Fragmentation loss weight: {args.lambda_fragmentation}")
    print(f"Fragmentation kernel size: {args.fragmentation_kernel_size}")
    print(f"Using noise for generation diversity: True")
    print(f"Noise dimension: 16")
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
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
        lambda_fragmentation=args.lambda_fragmentation,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        use_noise=True,
        noise_dim=16,
        fragmentation_kernel_size=args.fragmentation_kernel_size
    )
    
    # Create the optimizer
    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=args.lr_g)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr_d)
    
    # Create the trainer
    trainer = SwinGANTrainer(
        model=model,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=args.device,
        output_dir=args.output_dir,
        use_amp=not args.disable_amp,
        generator_save_interval=args.generator_save_interval
    )
    
    # Train the model
    trainer.train(train_loader, args.epochs, val_dataloader=val_loader, save_interval=args.save_interval)
    
    print(f"Training completed. Model saved to {args.output_dir}")
    print(f"Generator checkpoints saved to {trainer.generator_dir}")


def generate_lesions(
    model_checkpoint,
    lesion_atlas,
    output_file=None,
    output_dir=None,
    threshold=0.5,
    device='cuda',
    num_samples=1
):
    """
    Generate synthetic lesions using a trained GAN model
    
    Args:
        model_checkpoint (str): Path to the model checkpoint
        lesion_atlas (str): Path to the lesion atlas (frequency map)
        output_file (str, optional): Path to save the generated lesion (for single sample)
        output_dir (str, optional): Directory to save multiple generated samples
        threshold (float): Threshold for binarizing the generated lesion probability map
        device (str): Device to use for inference ('cuda' or 'cpu')
        num_samples (int): Number of samples to generate with different noise vectors
    """
    # Validate input parameters
    if num_samples > 1 and output_dir is None:
        raise ValueError("output_dir must be specified when generating multiple samples")
    if num_samples == 1 and output_file is None and output_dir is None:
        raise ValueError("Either output_file or output_dir must be specified")
    
    print(f"Generating lesions with the following parameters:")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Lesion atlas: {lesion_atlas}")
    if output_file:
        print(f"Output file: {output_file}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print(f"Threshold: {threshold}")
    print(f"Device: {device}")
    print(f"Number of samples: {num_samples}")
    
    # Load the lesion atlas
    atlas_img = nib.load(lesion_atlas)
    atlas_data = atlas_img.get_fdata()
    
    # Store original shape for later
    orig_shape = atlas_data.shape
    
    # Normalize the atlas
    atlas_data = (atlas_data - atlas_data.min()) / (atlas_data.max() - atlas_data.min() + 1e-8)
    
    # Convert to tensor
    atlas_tensor = torch.from_numpy(atlas_data).float().unsqueeze(0).unsqueeze(0)
    atlas_tensor = atlas_tensor.to(device)
    
    # Create the model
    model = SwinGAN(in_channels=1, out_channels=1)
    
    # Load the checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)
    
    # Check if checkpoint contains the full model or just generator
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'generator_state_dict' in checkpoint:
        print("Loading generator from checkpoint with configuration...")
        
        # Získání konfiguračních parametrů z checkpointu
        use_noise = checkpoint.get('use_noise', True)  # defaultně True pro zpětnou kompatibilitu
        noise_dim = checkpoint.get('noise_dim', 16)
        feature_size = checkpoint.get('feature_size', 24)
        dropout_rate = checkpoint.get('dropout_rate', 0.0)
        
        print(f"Checkpoint configuration: use_noise={use_noise}, noise_dim={noise_dim}, feature_size={feature_size}")
        
        # Vytvoření nového modelu s načtenými parametry
        model = SwinGAN(
            in_channels=1, 
            out_channels=1,
            feature_size=feature_size,
            dropout_rate=dropout_rate,
            use_noise=use_noise,
            noise_dim=noise_dim
        )
        
        # Načtení vah generátoru
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model = model.to(device)
    elif 'generator' in checkpoint:
        model.generator.load_state_dict(checkpoint['generator'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    elif output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Generate samples with different noise vectors
    all_samples = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Create a random noise vector for this sample
            if model.use_noise:
                noise = torch.randn(1, model.noise_dim, device=device)
            else:
                noise = None
            
            # Generate the lesion
            fake_lesion_logits = model(atlas_tensor, noise)
            
            # Apply sigmoid to get probability map
            fake_lesion = torch.sigmoid(fake_lesion_logits)
            
            # Convert to numpy array
            fake_lesion_np = fake_lesion.squeeze().cpu().numpy()
            
            # Binarize the lesion
            binary_lesion = (fake_lesion_np > threshold).astype(np.float32)
            
            # Ensure lesions only appear in regions with non-zero atlas values
            atlas_mask = atlas_data > 0
            binary_lesion = binary_lesion * atlas_mask
            
            all_samples.append(binary_lesion)
            
            # Determine output path for this sample
            if num_samples == 1 and output_file:
                # Single sample with specified output file
                sample_output_file = output_file
            else:
                # Multiple samples or output_dir was specified
                if output_file:
                    # Derive filename from output_file but save in output_dir
                    base_name = os.path.basename(output_file)
                    name, ext = os.path.splitext(base_name)
                    if ext == '.gz' and os.path.splitext(name)[1] == '.nii':
                        name = os.path.splitext(name)[0]
                        ext = '.nii.gz'
                else:
                    # Create a generic filename
                    name = "lesion"
                    ext = ".nii.gz"
                
                if num_samples > 1:
                    # Add sample number for multiple samples
                    filename = f"{name}_sample{i+1}{ext}"
                else:
                    filename = f"{name}{ext}"
                
                sample_output_file = os.path.join(output_dir, filename)
            
            # Create a new NIfTI image and save it
            lesion_img = nib.Nifti1Image(binary_lesion, atlas_img.affine)
            nib.save(lesion_img, sample_output_file)
            
            # Print some statistics
            # measure.label vrací tuple (labeled_array, num_features)
            labeled_array, num_lesions = measure.label(binary_lesion)
            
            # Výpočet objemu v procentech celkového objemu mozku
            total_brain_volume = np.count_nonzero(atlas_data > 0)  # Počet voxelů, kde je atlas nenulový
            lesion_volume_voxels = binary_lesion.sum()
            lesion_volume_percentage = (lesion_volume_voxels / total_brain_volume) * 100.0
            
            # Také vypočítáme objem v ml pro úplnost
            lesion_volume_ml = lesion_volume_voxels * np.prod(atlas_img.header.get_zooms()) / 1000.0  # in ml
            
            if num_samples > 1:
                print(f"Sample {i+1}: Generated {num_lesions} distinct lesions")
                print(f"Sample {i+1}: Total lesion volume: {lesion_volume_percentage:.2f}% of brain volume ({lesion_volume_ml:.2f} ml)")
                
                # Varování pokud počet lézí je mimo očekávaný rozsah podle trénovací množiny
                if num_lesions > 75:
                    print(f"WARNING: Sample {i+1} has {num_lesions} lesions, which is much higher than expected (1-75 based on training set)")
            else:
                print(f"Generated {num_lesions} distinct lesions")
                print(f"Total lesion volume: {lesion_volume_percentage:.2f}% of brain volume ({lesion_volume_ml:.2f} ml)")
                
                # Varování pokud počet lézí je mimo očekávaný rozsah podle trénovací množiny
                if num_lesions > 75:
                    print(f"WARNING: Generated {num_lesions} lesions, which is much higher than expected (1-75 based on training set)")
    
    # If multiple samples were generated, also save a mean probability map
    if num_samples > 1:
        mean_lesion = np.mean(all_samples, axis=0)
        
        if output_file:
            base_name = os.path.basename(output_file)
            name, ext = os.path.splitext(base_name)
            if ext == '.gz' and os.path.splitext(name)[1] == '.nii':
                name = os.path.splitext(name)[0]
                ext = '.nii.gz'
        else:
            name = "lesion"
            ext = ".nii.gz"
        
        mean_filename = f"{name}_mean{ext}"
        mean_output_file = os.path.join(output_dir, mean_filename)
        
        mean_img = nib.Nifti1Image(mean_lesion, atlas_img.affine)
        nib.save(mean_img, mean_output_file)
        print(f"Saved mean probability map to {mean_output_file}")


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
    train_parser.add_argument('--lambda_fragmentation', type=float, default=50.0, 
                             help='Weight for fragmentation loss - higher values promote more coherent lesions')
    train_parser.add_argument('--fragmentation_kernel_size', type=int, default=5, 
                             help='Size of kernel used for fragmentation loss - larger values promote larger coherent structures')
    train_parser.add_argument('--focal_alpha', type=float, default=0.75, 
                             help='Alpha parameter for focal loss')
    train_parser.add_argument('--focal_gamma', type=float, default=2.0, 
                             help='Gamma parameter for focal loss')
    train_parser.add_argument('--min_non_zero', type=float, default=0.000001, 
                             help='Minimum percentage of non-zero voxels to include sample')
    train_parser.add_argument('--generator_save_interval', type=int, default=4, 
                             help='Interval for saving generator-only checkpoints')
    train_parser.add_argument('--num_workers', type=int, default=4, 
                             help='Number of workers for data loading')
    train_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                             help='Device to use (cuda or cpu)')
    train_parser.add_argument('--disable_amp', action='store_true', 
                             help='Disable automatic mixed precision training')
    train_parser.add_argument('--save_interval', type=int, default=5, 
                             help='Interval for saving full model checkpoints')
    
    # Generation subparser
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic lesions')
    gen_parser.add_argument('--model_checkpoint', type=str, required=True, 
                           help='Path to the trained model checkpoint')
    gen_parser.add_argument('--lesion_atlas', type=str, required=True, 
                           help='Path to the lesion frequency atlas .nii file')
    gen_parser.add_argument('--output_file', type=str, default=None,
                           help='Path to save the generated lesion .nii file (for single sample)')
    gen_parser.add_argument('--output_dir', type=str, default=None,
                           help='Directory to save multiple generated samples')
    gen_parser.add_argument('--threshold', type=float, default=0.5, 
                           help='Threshold for binarizing the generated lesions')
    gen_parser.add_argument('--feature_size', type=int, default=24, 
                           help='Feature size for SwinUNETR (must match training)')
    gen_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                           help='Device to use (cuda or cpu)')
    gen_parser.add_argument('--num_samples', type=int, default=1, 
                           help='Number of samples to generate with different noise vectors')
    
    args = parser.parse_args()
    
    # Run the appropriate mode
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'generate':
        generate_lesions(
            model_checkpoint=args.model_checkpoint,
            lesion_atlas=args.lesion_atlas,
            output_file=args.output_file,
            output_dir=args.output_dir,
            threshold=args.threshold,
            device=args.device,
            num_samples=args.num_samples
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
