import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, RandGaussianNoised, 
    ScaleIntensityd, RandAdjustContrastd, ToTensord, LoadImaged,
    SpatialPadd, ScaleIntensityRanged, RandSpatialCropd, RandShiftIntensityd,
    EnsureChannelFirstd, EnsureTyped, NormalizeIntensityd
)
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models


def load_mha_file(file_path):
    """
    Load an MHA file and return as a numpy array along with original image
    """
    print(f"Loading {file_path}")
    img = sitk.ReadImage(str(file_path))
    data = sitk.GetArrayFromImage(img)
    return data, img


def save_mha_file(data, reference_image, output_path):
    """
    Save a numpy array as an MHA file using the metadata from reference_image
    """
    print(f"Saving to {output_path}")
    out_img = sitk.GetImageFromArray(data)
    out_img.CopyInformation(reference_image)
    sitk.WriteImage(out_img, str(output_path))


class HIEDataset(Dataset):
    """
    Dataset for loading pseudo-healthy ADC maps, lesion maps and real ADC maps with lesions
    """
    def __init__(self, data_dir, lesion_dir, target_dir, transform=None):
        """
        Args:
            data_dir: Directory with pseudo-healthy ADC maps
            lesion_dir: Directory with lesion maps
            target_dir: Directory with real ADC maps with lesions
            transform: Optional transform to be applied on sample
        """
        self.data_dir = data_dir
        self.lesion_dir = lesion_dir
        self.target_dir = target_dir
        self.transform = transform
        
        # Get list of files that exist in all directories
        pseudo_healthy_files = set([f for f in os.listdir(data_dir) if f.endswith('.mha')])
        lesion_files = set([f for f in os.listdir(lesion_dir) if f.endswith('.mha')])
        target_files = set([f for f in os.listdir(target_dir) if f.endswith('.mha')])
        
        # Find common files (based on patient ID)
        self.file_names = list(pseudo_healthy_files.intersection(lesion_files).intersection(target_files))
        print(f"Found {len(self.file_names)} matching files")
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_name = self.file_names[idx]
        
        # Load pseudo-healthy ADC map
        pseudo_healthy_path = os.path.join(self.data_dir, file_name)
        pseudo_healthy_data, pseudo_healthy_img = load_mha_file(pseudo_healthy_path)
        
        # Load lesion map
        lesion_path = os.path.join(self.lesion_dir, file_name)
        lesion_data, lesion_img = load_mha_file(lesion_path)
        
        # Load target ADC map with lesions
        target_path = os.path.join(self.target_dir, file_name)
        target_data, target_img = load_mha_file(target_path)
        
        # Prepare sample
        sample = {
            'pseudo_healthy': pseudo_healthy_data,
            'lesion_map': lesion_data,
            'target': target_data,
            'image_meta': {
                'patient_id': file_name,
                'reference_img': target_img
            }
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class DownConvBlock(nn.Module):
    """
    Down-convolutional block for 3D U-Net Generator
    """
    def __init__(self, in_channels, out_channels, first_block=False):
        super(DownConvBlock, self).__init__()
        
        if first_block:
            # First block doesn't have batch normalization
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            # Regular blocks have batch normalization
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
    
    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    """
    Up-convolutional block for 3D U-Net Generator
    """
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super(UpConvBlock, self).__init__()
        
        layers = [
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if apply_dropout:
            layers.append(nn.Dropout3d(0.5))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    """
    Generator network for CGAN with 7 down-conv and 7 up-conv blocks
    Takes pseudo-healthy ADC map and lesion map as input
    Outputs ADC map with synthesized lesions
    """
    def __init__(self, in_channels=2, out_channels=1, base_features=64):
        super(Generator, self).__init__()
        
        # Contracting path (encoder)
        self.down1 = DownConvBlock(in_channels, base_features, first_block=True)  # No BN in first layer
        self.down2 = DownConvBlock(base_features, base_features*2)
        self.down3 = DownConvBlock(base_features*2, base_features*4)
        self.down4 = DownConvBlock(base_features*4, base_features*8)
        self.down5 = DownConvBlock(base_features*8, base_features*8)
        self.down6 = DownConvBlock(base_features*8, base_features*8)
        self.down7 = DownConvBlock(base_features*8, base_features*8)
        
        # Bottleneck
        self.bottleneck = nn.Conv3d(base_features*8, base_features*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bottleneck_activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Expanding path (decoder) with skip connections
        self.up1 = UpConvBlock(base_features*8, base_features*8, apply_dropout=True)
        self.up2 = UpConvBlock(base_features*8*2, base_features*8, apply_dropout=True)
        self.up3 = UpConvBlock(base_features*8*2, base_features*8, apply_dropout=True)
        self.up4 = UpConvBlock(base_features*8*2, base_features*8)
        self.up5 = UpConvBlock(base_features*8*2, base_features*4)
        self.up6 = UpConvBlock(base_features*4*2, base_features*2)
        self.up7 = UpConvBlock(base_features*2*2, base_features)
        
        # Final layer
        self.final = nn.Conv3d(base_features*2, out_channels, kernel_size=3, padding=1)
        self.final_activation = nn.Tanh()
    
    def forward(self, pseudo_healthy, lesion_map):
        # Concatenate inputs along channel dimension
        x = torch.cat([pseudo_healthy, lesion_map], dim=1)
        
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        # Bottleneck
        bottleneck = self.bottleneck_activation(self.bottleneck(d7))
        
        # Decoder with skip connections
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        
        # Final layer
        output = self.final(torch.cat([u7, d1], dim=1))
        return self.final_activation(output)


class Discriminator(nn.Module):
    """
    Discriminator network for CGAN with 5 convolutional layers
    Takes ADC map and lesion map as input
    Outputs scalar (real/fake) prediction
    """
    def __init__(self, in_channels=2, base_features=64):
        super(Discriminator, self).__init__()
        
        # First layer - no batch normalization
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, base_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv3d(base_features, base_features*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_features*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv3d(base_features*2, base_features*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_features*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv3d(base_features*4, base_features*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_features*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output Layer (5th layer)
        self.layer5 = nn.Sequential(
            nn.Conv3d(base_features*8, 1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, adc_map, lesion_map):
        # Concatenate inputs along channel dimension
        x = torch.cat([adc_map, lesion_map], dim=1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).eval()
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:18].eval())
        blocks.append(vgg.features[18:27].eval())
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        
        self.blocks = nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, input, target):
        """
        Compute perceptual loss between input and target
        For 3D volumes, we compute 2D perceptual loss on slices and average
        """
        # For 3D volumes, we compute perceptual loss on representative 2D slices
        batch_size, channels, depth, height, width = input.shape
        total_loss = 0.0
        
        # Select representative slices (mid slices in each dimension)
        d_slices = [depth // 4, depth // 2, 3 * depth // 4]
        
        for d in d_slices:
            # Extract 2D slices
            input_slice = input[:, :, d, :, :]
            target_slice = target[:, :, d, :, :]
            
            # Handle grayscale input (expand to 3 channels for VGG)
            if channels == 1:
                input_slice = input_slice.repeat(1, 3, 1, 1)
                target_slice = target_slice.repeat(1, 3, 1, 1)
            
            # Normalize input for VGG
            input_slice = (input_slice - self.mean) / self.std
            target_slice = (target_slice - self.mean) / self.std
            
            if self.resize:
                input_slice = F.interpolate(input_slice, mode='bilinear', size=(224, 224), align_corners=False)
                target_slice = F.interpolate(target_slice, mode='bilinear', size=(224, 224), align_corners=False)
            
            loss = 0.0
            x = input_slice
            y = target_slice
            
            for block in self.blocks:
                x = block(x)
                y = block(y)
                loss += F.l1_loss(x, y)
            
            total_loss += loss
        
        return total_loss / len(d_slices)


class LesionCGAN:
    """
    CGAN model for inpainting lesions into ADC maps
    """
    def __init__(self, 
                 generator=None, 
                 discriminator=None, 
                 alpha=1.0,     # Weight for adversarial loss
                 beta=10.0,     # Weight for MAE loss
                 gamma=0.1,     # Weight for perceptual loss
                 lr=0.0002,
                 device='cuda'):
        self.device = device
        
        # Initialize networks
        self.generator = generator if generator else Generator().to(device)
        self.discriminator = discriminator if discriminator else Discriminator().to(device)
        
        # Loss functions
        self.mae_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss().to(device)
        
        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Tensorboard
        self.writer = SummaryWriter()
    
    def train_step(self, pseudo_healthy, lesion_map, target):
        """
        Perform a single training step (for both generator and discriminator)
        """
        batch_size = pseudo_healthy.size(0)
        real_label = torch.ones(batch_size, 1, 1, 1, 1).to(self.device)
        fake_label = torch.zeros(batch_size, 1, 1, 1, 1).to(self.device)
        
        # -----------------
        # Train Generator
        # -----------------
        self.optimizer_G.zero_grad()
        
        # Generate fake ADC map
        fake_adc = self.generator(pseudo_healthy, lesion_map)
        
        # Adversarial loss
        pred_fake = self.discriminator(fake_adc, lesion_map)
        loss_G_adv = F.binary_cross_entropy(pred_fake, real_label)
        
        # MAE loss
        loss_G_mae = self.mae_loss(fake_adc, target)
        
        # Perceptual loss
        loss_G_perc = self.perceptual_loss(fake_adc, target)
        
        # Total generator loss
        loss_G = self.alpha * loss_G_adv + self.beta * loss_G_mae + self.gamma * loss_G_perc
        
        loss_G.backward()
        self.optimizer_G.step()
        
        # -----------------
        # Train Discriminator
        # -----------------
        self.optimizer_D.zero_grad()
        
        # Real loss
        pred_real = self.discriminator(target, lesion_map)
        loss_D_real = F.binary_cross_entropy(pred_real, real_label)
        
        # Fake loss (detach to avoid backprop through generator)
        pred_fake = self.discriminator(fake_adc.detach(), lesion_map)
        loss_D_fake = F.binary_cross_entropy(pred_fake, fake_label)
        
        # Total discriminator loss
        loss_D = (loss_D_real + loss_D_fake) / 2
        
        loss_D.backward()
        self.optimizer_D.step()
        
        return {
            'loss_G': loss_G.item(),
            'loss_G_adv': loss_G_adv.item(),
            'loss_G_mae': loss_G_mae.item(),
            'loss_G_perc': loss_G_perc.item(),
            'loss_D': loss_D.item(),
            'fake_adc': fake_adc.detach(),
            'target': target
        }

    def compute_lesion_mae(self, fake_adc, target, lesion_map):
        """
        Compute MAE loss only in lesion regions
        """
        # Create binary mask for lesion regions
        lesion_mask = (lesion_map > 0).float()
        
        # Calculate error only in lesion regions
        error = torch.abs(fake_adc - target) * lesion_mask
        
        # Calculate mean error over lesion voxels
        total_lesion_voxels = torch.sum(lesion_mask) + 1e-8  # Avoid division by zero
        lesion_mae = torch.sum(error) / total_lesion_voxels
        
        return lesion_mae.item()
    
    def train(self, dataloader, num_epochs=100, save_dir='checkpoints'):
        """
        Train the CGAN model
        """
        os.makedirs(save_dir, exist_ok=True)
        best_lesion_mae = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            epoch_lesion_mae = 0.0
            
            self.generator.train()
            self.discriminator.train()
            
            with tqdm(dataloader, unit='batch') as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
                
                for batch in tepoch:
                    pseudo_healthy = batch['pseudo_healthy'].to(self.device)
                    lesion_map = batch['lesion_map'].to(self.device)
                    target = batch['target'].to(self.device)
                    
                    # Train step
                    result = self.train_step(pseudo_healthy, lesion_map, target)
                    
                    # Compute lesion-specific MAE
                    lesion_mae = self.compute_lesion_mae(result['fake_adc'], target, lesion_map)
                    
                    # Update losses
                    epoch_loss_G += result['loss_G']
                    epoch_loss_D += result['loss_D']
                    epoch_lesion_mae += lesion_mae
                    
                    # Update progress bar
                    tepoch.set_postfix(loss_G=result['loss_G'], loss_D=result['loss_D'], lesion_mae=lesion_mae)
            
            # Calculate average epoch losses
            num_batches = len(dataloader)
            avg_loss_G = epoch_loss_G / num_batches
            avg_loss_D = epoch_loss_D / num_batches
            avg_lesion_mae = epoch_lesion_mae / num_batches
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Generator', avg_loss_G, epoch)
            self.writer.add_scalar('Loss/Discriminator', avg_loss_D, epoch)
            self.writer.add_scalar('Metrics/Lesion_MAE', avg_lesion_mae, epoch)
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                'loss_G': avg_loss_G,
                'loss_D': avg_loss_D,
                'lesion_mae': avg_lesion_mae
            }
            
            # Save the latest model
            torch.save(checkpoint, os.path.join(save_dir, 'latest_model.pth'))
            
            # Save the best model based on lesion MAE
            if avg_lesion_mae < best_lesion_mae:
                best_lesion_mae = avg_lesion_mae
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved with lesion MAE: {best_lesion_mae:.6f}")
            
            # Visualize sample results every 5 epochs
            if epoch % 5 == 0:
                self.visualize_results(pseudo_healthy[:4], lesion_map[:4], target[:4], 
                                      result['fake_adc'][:4], epoch, save_dir)
        
        self.writer.close()
        print("Training completed!")
    
    def visualize_results(self, pseudo_healthy, lesion_map, target, fake_adc, epoch, save_dir):
        """
        Visualize sample results
        """
        # Create directory for visualizations
        vis_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Set to evaluation mode
        self.generator.eval()
        
        with torch.no_grad():
            # Process each sample in the batch
            for i in range(min(4, pseudo_healthy.size(0))):
                # Get a mid-slice for each volume
                depth = pseudo_healthy.size(2) // 2
                
                # Get slices
                pseudo_slice = pseudo_healthy[i, 0, depth].cpu().numpy()
                lesion_slice = lesion_map[i, 0, depth].cpu().numpy()
                target_slice = target[i, 0, depth].cpu().numpy()
                fake_slice = fake_adc[i, 0, depth].cpu().numpy()
                
                # Create figure
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Plot pseudo-healthy ADC
                axes[0, 0].imshow(pseudo_slice, cmap='gray')
                axes[0, 0].set_title('Pseudo-healthy ADC')
                axes[0, 0].axis('off')
                
                # Plot lesion map
                axes[0, 1].imshow(lesion_slice, cmap='jet')
                axes[0, 1].set_title('Lesion Map')
                axes[0, 1].axis('off')
                
                # Plot target ADC
                axes[1, 0].imshow(target_slice, cmap='gray')
                axes[1, 0].set_title('Target ADC')
                axes[1, 0].axis('off')
                
                # Plot generated ADC
                axes[1, 1].imshow(fake_slice, cmap='gray')
                axes[1, 1].set_title('Generated ADC')
                axes[1, 1].axis('off')
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'epoch_{epoch}_sample_{i}.png'))
                plt.close()
        
        # Set back to training mode
        self.generator.train()
    
    def save(self, path):
        """
        Save the model
        """
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }, path)
    
    def load(self, path):
        """
        Load the model
        """
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])


def get_transforms():
    """
    Get data transforms for training
    """
    train_transforms = Compose([
        # Convert to tensor
        ToTensord(keys=['pseudo_healthy', 'lesion_map', 'target']),
        
        # Add channel dimension if needed
        EnsureChannelFirstd(keys=['pseudo_healthy', 'lesion_map', 'target']),
        
        # Normalize intensity
        ScaleIntensityd(keys=['pseudo_healthy', 'target'], minv=0.0, maxv=1.0),
        
        # Binarize lesion map
        ScaleIntensityd(keys=['lesion_map'], minv=0.0, maxv=1.0),
        
        # Random rotations
        RandRotate90d(keys=['pseudo_healthy', 'lesion_map', 'target'], 
                      prob=0.5, spatial_axes=(0, 1)),
        
        # Random flips
        RandFlipd(keys=['pseudo_healthy', 'lesion_map', 'target'], 
                  prob=0.5, spatial_axis=0),
        RandFlipd(keys=['pseudo_healthy', 'lesion_map', 'target'], 
                  prob=0.5, spatial_axis=1),
        RandFlipd(keys=['pseudo_healthy', 'lesion_map', 'target'], 
                  prob=0.5, spatial_axis=2),
        
        # Random intensity shift (only for ADC maps)
        RandShiftIntensityd(keys=['pseudo_healthy', 'target'], 
                           prob=0.5, offsets=0.1),
        
        # Ensure tensor type
        EnsureTyped(keys=['pseudo_healthy', 'lesion_map', 'target']),
    ])
    
    val_transforms = Compose([
        # Convert to tensor
        ToTensord(keys=['pseudo_healthy', 'lesion_map', 'target']),
        
        # Add channel dimension if needed
        EnsureChannelFirstd(keys=['pseudo_healthy', 'lesion_map', 'target']),
        
        # Normalize intensity
        ScaleIntensityd(keys=['pseudo_healthy', 'target'], minv=0.0, maxv=1.0),
        
        # Binarize lesion map
        ScaleIntensityd(keys=['lesion_map'], minv=0.0, maxv=1.0),
        
        # Ensure tensor type
        EnsureTyped(keys=['pseudo_healthy', 'lesion_map', 'target']),
    ])
    
    return train_transforms, val_transforms


def main(args):
    """
    Main function for training the CGAN model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    train_transforms, val_transforms = get_transforms()
    
    # Create dataset and dataloader
    train_dataset = HIEDataset(
        data_dir=args.pseudo_healthy_dir,
        lesion_dir=args.lesion_dir,
        target_dir=args.target_dir,
        transform=train_transforms
    )
    
    val_dataset = HIEDataset(
        data_dir=args.pseudo_healthy_dir,
        lesion_dir=args.lesion_dir,
        target_dir=args.target_dir,
        transform=val_transforms
    )
    
    # Split dataset into training and validation
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.val_split * dataset_size))
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=True
    )
    
    # Initialize model
    model = LesionCGAN(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lr=args.lr,
        device=device
    )
    
    # Train model
    model.train(
        dataloader=train_dataloader,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='CGAN Lesion Inpainting')
    
    # Data paths
    parser.add_argument('--pseudo_healthy_dir', type=str, required=True,
                        help='Directory with pseudo-healthy ADC maps')
    parser.add_argument('--lesion_dir', type=str, required=True,
                        help='Directory with lesion maps')
    parser.add_argument('--target_dir', type=str, required=True,
                        help='Directory with real ADC maps with lesions')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for adversarial loss')
    parser.add_argument('--beta', type=float, default=10.0,
                        help='Weight for MAE loss')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Weight for perceptual loss')
    
    # Other settings
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    
    args = parser.parse_args()
    main(args)
