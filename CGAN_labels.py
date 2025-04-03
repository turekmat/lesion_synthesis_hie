import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Define the Dataset class
class HIEDataset(Dataset):
    def __init__(self, labels_dir, lesion_atlas_path, transform=None):
        self.transform = transform
        self.lesion_atlas = self.load_nifti(lesion_atlas_path)
        
        # Load all label files
        self.label_paths = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) 
                            if f.endswith('.nii') or f.endswith('.nii.gz')]
        
        # Filter out all-black labels
        valid_labels = []
        for path in self.label_paths:
            label = self.load_nifti(path)
            if np.sum(label) > 0:  # Check if the label has any non-zero values
                valid_labels.append(path)
        
        self.label_paths = valid_labels
        print(f"Dataset loaded with {len(self.label_paths)} valid samples (removed {len(self.label_paths) - len(valid_labels)} all-black samples)")
        
    def load_nifti(self, path):
        nifti_img = nib.load(path)
        data = nifti_img.get_fdata()
        # Ensure the data is normalized and has the correct dimensions
        data = np.clip(data, 0, 1)
        return data
    
    def __len__(self):
        return len(self.label_paths)
    
    def __getitem__(self, idx):
        label_path = self.label_paths[idx]
        label = self.load_nifti(label_path)
        
        # Convert to binary
        label = (label > 0).astype(np.float32)
        
        # Convert to tensors
        label_tensor = torch.from_numpy(label).float().unsqueeze(0)
        atlas_tensor = torch.from_numpy(self.lesion_atlas).float().unsqueeze(0)
        
        # Normalize lesion atlas to [0, 1]
        atlas_tensor = atlas_tensor / 0.34
        
        # Generate random noise (shape [100, 1, 1, 1] without extra dimension)
        noise = torch.randn(100, 1, 1, 1)
        
        return {
            'label': label_tensor,
            'atlas': atlas_tensor,
            'noise': noise
        }

# Generator spatial attention block
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, padding=1)
        
    def forward(self, x, atlas):
        # Concatenate input features with atlas
        combined = torch.cat([x, atlas], dim=1)
        attention = torch.sigmoid(self.conv(combined))
        return attention * x

# Adaptive lesion gate for upsampling paths
class AdaptiveLesionGate(nn.Module):
    def __init__(self):
        super(AdaptiveLesionGate, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x, atlas):
        gate = 1 + self.alpha * atlas
        return gate * x

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Initial processing of noise vector and atlas
        self.initial = nn.Sequential(
            nn.Conv3d(101, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder blocks
        self.down1 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.down3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.down4 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Spatial attention modules
        self.attn1 = SpatialAttention()
        self.attn2 = SpatialAttention()
        self.attn3 = SpatialAttention()
        self.attn4 = SpatialAttention()
        
        # Decoder blocks with lesion gates
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(512),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive lesion gates
        self.gate1 = AdaptiveLesionGate()
        self.gate2 = AdaptiveLesionGate()
        self.gate3 = AdaptiveLesionGate()
        self.gate4 = AdaptiveLesionGate()
        
        # Output layer
        self.final = nn.Sequential(
            nn.Conv3d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, noise, atlas):
        # Resize noise to match spatial dimensions
        batch_size = noise.size(0)
        
        # Handle noise shape - reshape if necessary
        if noise.dim() > 5:  # If noise has extra dimensions
            noise = noise.squeeze(1)  # Remove the extra dimension
        
        # Expand noise to match spatial dimensions
        noise_resized = noise.expand(batch_size, 100, 
                                     atlas.size(2), atlas.size(3), atlas.size(4))
        
        # Concatenate noise with atlas
        x = torch.cat([noise_resized, atlas], dim=1)
        
        # Initial processing
        x0 = self.initial(x)
        
        # Encoder path with attention
        x1 = self.down1(x0)
        x1 = self.attn1(x1, F.interpolate(atlas, size=x1.shape[2:]))
        
        x2 = self.down2(x1)
        x2 = self.attn2(x2, F.interpolate(atlas, size=x2.shape[2:]))
        
        x3 = self.down3(x2)
        x3 = self.attn3(x3, F.interpolate(atlas, size=x3.shape[2:]))
        
        x4 = self.down4(x3)
        x4 = self.attn4(x4, F.interpolate(atlas, size=x4.shape[2:]))
        
        # Decoder path with skip connections and lesion gates
        d1 = self.up1(x4)
        d1 = self.gate1(d1, F.interpolate(atlas, size=d1.shape[2:]))
        d1 = torch.cat([d1, x3], dim=1)
        
        d2 = self.up2(d1)
        d2 = self.gate2(d2, F.interpolate(atlas, size=d2.shape[2:]))
        d2 = torch.cat([d2, x2], dim=1)
        
        d3 = self.up3(d2)
        d3 = self.gate3(d3, F.interpolate(atlas, size=d3.shape[2:]))
        d3 = torch.cat([d3, x1], dim=1)
        
        d4 = self.up4(d3)
        d4 = self.gate4(d4, F.interpolate(atlas, size=d4.shape[2:]))
        
        # Final output
        output = self.final(d4)
        
        # Mask the output with atlas (lesions only where atlas > 0)
        masked_output = output * (atlas > 0).float()
        
        return masked_output

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # PatchGAN discriminator with spectral normalization
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(2, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output layer (PatchGAN)
        self.output = nn.Conv3d(512, 1, kernel_size=3, padding=1)
        
    def forward(self, x, atlas):
        # Concatenate input with atlas
        x = torch.cat([x, atlas], dim=1)
        
        # Forward pass through layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # PatchGAN output
        x = self.output(x)
        
        return x

# Loss functions and utilities
def compute_gradient_penalty(discriminator, real_samples, fake_samples, atlas, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1).to(device)
    
    # Interpolated samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Discriminator output for interpolated samples
    d_interpolates = discriminator(interpolates, atlas)
    
    # Create fake targets
    fake = torch.ones(d_interpolates.size()).to(device)
    
    # Get gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Calculate gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_frequency_loss(generated, atlas):
    """Computes KL divergence between histograms of generated lesions and atlas"""
    # Compute histogram of generated lesions where atlas > 0
    mask = atlas > 0
    if not torch.any(mask):
        return torch.tensor(0.0, device=generated.device)
    
    gen_masked = generated[mask]
    atlas_masked = atlas[mask]
    
    # Create histogram for generated lesions (10 bins)
    gen_hist = torch.histc(gen_masked, bins=10, min=0, max=1)
    gen_hist = gen_hist / (gen_hist.sum() + 1e-8)
    
    # Create histogram for atlas (10 bins)
    atlas_hist = torch.histc(atlas_masked, bins=10, min=0, max=1)
    atlas_hist = atlas_hist / (atlas_hist.sum() + 1e-8)
    
    # KL divergence (adding small epsilon to avoid log(0))
    kl_div = F.kl_div(torch.log(gen_hist + 1e-8), atlas_hist, reduction='batchmean')
    
    # Add MSE term
    mse_loss = F.mse_loss(generated * atlas, atlas)
    
    return kl_div + mse_loss

def compute_sparsity_loss(generated, atlas):
    """L1 norm of the output weighted by atlas values"""
    return torch.mean(torch.abs(generated) * atlas)

# Training function
def train(generator, discriminator, dataloader, num_epochs, device, output_dir):
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.95)
    scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.95)
    
    # Track losses
    g_losses = []
    d_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            real_labels = batch['label'].to(device)
            atlas = batch['atlas'].to(device)
            noise = batch['noise'].to(device)
            
            batch_size = real_labels.size(0)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            # Train discriminator 3 times for each generator update
            for _ in range(3):
                optimizer_D.zero_grad()
                
                # Generate fake samples
                fake_labels = generator(noise, atlas)
                
                # Real samples
                real_validity = discriminator(real_labels, atlas)
                fake_validity = discriminator(fake_labels.detach(), atlas)
                
                # Wasserstein loss with gradient penalty
                d_real_loss = -torch.mean(real_validity)
                d_fake_loss = torch.mean(fake_validity)
                
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(
                    discriminator, real_labels, fake_labels.detach(), atlas, device
                )
                
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss + 10 * gradient_penalty
                
                d_loss.backward()
                optimizer_D.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            
            # Generate fake samples
            fake_labels = generator(noise, atlas)
            
            # Adversarial loss
            fake_validity = discriminator(fake_labels, atlas)
            g_adv_loss = -torch.mean(fake_validity)
            
            # Frequency regularization loss
            freq_loss = compute_frequency_loss(fake_labels, atlas)
            
            # Sparsity loss
            sparse_loss = compute_sparsity_loss(fake_labels, atlas)
            
            # Total generator loss (no topology loss for now due to complexity)
            g_loss = 0.5 * g_adv_loss + 10 * freq_loss + 0.1 * sparse_loss
            
            g_loss.backward()
            optimizer_G.step()
            
            # Track losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Average epoch losses
        epoch_g_loss /= len(dataloader)
        epoch_d_loss /= len(dataloader)
        
        g_losses.append(epoch_g_loss)
        d_losses.append(epoch_d_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}")
        
        # Save samples
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_samples(generator, dataloader, device, epoch, output_dir)
            
        # Save model
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict()
            }, os.path.join(output_dir, f"model_epoch_{epoch+1}.pt"))
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    
    # Save final model
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()
    }, os.path.join(output_dir, "final_model.pt"))
    
    return g_losses, d_losses

# Save generated samples
def save_samples(generator, dataloader, device, epoch, output_dir):
    generator.eval()
    with torch.no_grad():
        # Get a batch of data
        batch = next(iter(dataloader))
        real_labels = batch['label'].to(device)
        atlas = batch['atlas'].to(device)
        noise = batch['noise'].to(device)
        
        # Generate fake samples
        fake_labels = generator(noise, atlas)
        
        # Convert to numpy for visualization
        fake_np = fake_labels[0, 0].cpu().numpy()
        real_np = real_labels[0, 0].cpu().numpy()
        atlas_np = atlas[0, 0].cpu().numpy()
        
        # Create sample directory
        sample_dir = os.path.join(output_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save middle slices as images
        mid_z = fake_np.shape[2] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(atlas_np[:, :, mid_z], cmap='viridis')
        axes[0].set_title('Lesion Atlas')
        axes[0].axis('off')
        
        axes[1].imshow(real_np[:, :, mid_z], cmap='gray')
        axes[1].set_title('Real Label')
        axes[1].axis('off')
        
        axes[2].imshow(fake_np[:, :, mid_z], cmap='gray')
        axes[2].set_title('Generated Label')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f'sample_epoch_{epoch+1}.png'))
        plt.close()
        
    generator.train()

# Evaluation metrics
def calculate_dice_score(pred, target):
    """Calculate Dice score between predicted and target labels"""
    smooth = 1e-5
    pred_binary = (pred > 0.5).float()
    intersection = torch.sum(pred_binary * target)
    return (2. * intersection + smooth) / (torch.sum(pred_binary) + torch.sum(target) + smooth)

def evaluate(generator, dataloader, device):
    generator.eval()
    dice_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            real_labels = batch['label'].to(device)
            atlas = batch['atlas'].to(device)
            noise = batch['noise'].to(device)
            
            # Generate fake samples
            fake_labels = generator(noise, atlas)
            
            # Calculate Dice score
            for i in range(fake_labels.size(0)):
                dice = calculate_dice_score(fake_labels[i], real_labels[i])
                dice_scores.append(dice.item())
    
    generator.train()
    return np.mean(dice_scores)

# Main function
def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = HIEDataset(args.labels_dir, args.lesion_atlas_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Train or evaluate
    if not args.eval_only:
        # Train
        train(generator, discriminator, dataloader, args.num_epochs, device, args.output_dir)
    
    # Evaluate
    dice_score = evaluate(generator, dataloader, device)
    print(f"Average Dice Score: {dice_score:.4f}")
    
    # Generate some final samples
    save_samples(generator, dataloader, device, args.num_epochs, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D cGAN for HIE Lesion Synthesis")
    parser.add_argument("--lesion_atlas_path", type=str, required=True, help="Path to lesion atlas .nii file")
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory containing label .nii files")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation, no training")
    
    args = parser.parse_args()
    main(args)
