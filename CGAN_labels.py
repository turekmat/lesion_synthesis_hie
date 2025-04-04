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
from scipy import ndimage

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
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(in_channels + 1, 1, kernel_size=3, padding=1)
        
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
        
        # Spatial attention modules with correct channel counts
        self.attn1 = SpatialAttention(128)
        self.attn2 = SpatialAttention(256)
        self.attn3 = SpatialAttention(512)
        self.attn4 = SpatialAttention(512)
        
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
        
        # Nová vrstva pro podporu multi-lesion patterns
        self.lesion_pattern_decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 8, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Finální konvoluce pro zpracování vzorů lézí - nová vrstva namísto self.final
        self.pattern_to_lesion = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
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
        
        # Použijeme původní metodu pro základní případ
        base_output = self.final(d4)
        
        # Použijeme multi-pattern přístup pro generování vícečetných lézí
        lesion_patterns = self.lesion_pattern_decoder(d4)
        outputs = [base_output]  # Začneme se základním výstupem
        
        # Generujeme několik dílčích map lézí
        for i in range(8):
            pattern = lesion_patterns[:, i:i+1]
            # Použít novou vrstvu pro zpracování jednotlivých vzorů
            sub_output = self.pattern_to_lesion(pattern)
            outputs.append(sub_output)
        
        # Kombinujeme všechny mapy pomocí max operace
        output = outputs[0]
        for i in range(1, len(outputs)):
            output = torch.max(output, outputs[i])
        
        # Masking s atlasem
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

def compute_empty_penalty(generated, atlas):
    """Penalize empty generations (all zeros)"""
    # Calculate percentage of non-zero voxels in the atlas-masked area
    mask = atlas > 0
    if not torch.any(mask):
        return torch.tensor(0.0, device=generated.device)
    
    # Get non-zero percentage in masked area
    non_zero_percentage = torch.mean((generated[mask] > 0.1).float())
    
    # Penalize if percentage is too low (empty image)
    # Target a minimum of 0.5% non-zero voxels (can adjust based on data)
    target_percentage = 0.005
    empty_penalty = torch.relu(target_percentage - non_zero_percentage) * 100.0
    
    return empty_penalty

def compute_shape_consistency_loss(generated, real_labels, atlas):
    """Encourage the generator to create lesions with similar morphology to real ones"""
    # Skip if no real lesions
    if torch.sum(real_labels) < 1:
        return torch.tensor(0.0, device=generated.device)
    
    # Binarize generated lesions for shape comparison
    gen_binary = (generated > 0.5).float()
    
    # Calculate average lesion size in real data
    real_component_sizes = []
    batch_size = real_labels.size(0)
    for i in range(batch_size):
        # Skip empty real samples
        if torch.sum(real_labels[i]) < 1:
            continue
        
        # Get average connected component size (approximation using 2D slices for simplicity)
        # We'll use middle slices as representative
        mid_z = real_labels.size(4) // 2
        real_slice = real_labels[i, 0, :, :, mid_z].cpu().numpy()
        gen_slice = gen_binary[i, 0, :, :, mid_z].cpu().numpy()
        
        # Get connected components in real slice
        labeled_real, num_real = ndimage.label(real_slice)
        if num_real > 0:
            real_sizes = ndimage.sum(real_slice, labeled_real, range(1, num_real+1))
            real_component_sizes.extend(real_sizes)
        
        # Get connected components in generated slice
        labeled_gen, num_gen = ndimage.label(gen_slice)
        
        # Penalize if number of components is too different
        if num_real > 0 and num_gen > 0:
            # Convert back to tensor for gradient flow
            comp_diff = torch.abs(torch.tensor(num_real - num_gen, device=generated.device).float()) / max(num_real, 1)
            return comp_diff * 0.1  # Scale factor to balance with other losses
    
    # If we couldn't compute a meaningful shape loss
    return torch.tensor(0.0, device=generated.device)

def analyze_lesion_distribution(dataset, output_dir=None):
    """
    Analyze the distribution of lesion counts in the original dataset
    
    Args:
        dataset: HIEDataset object containing the training data
        output_dir: Directory to save analysis results
    
    Returns:
        counts: List of lesion counts for each sample
        sizes: List of average lesion sizes for each sample
        percentages: List of percentage of voxels that are lesions
        component_sizes_list: List of all component sizes across all samples
    """
    counts = []
    sizes = []
    percentages = []
    component_sizes_list = []
    component_sizes_by_sample = []
    max_component_sizes = []
    min_component_sizes = []
    
    print("Analyzing lesion distribution in the original dataset...")
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        label = sample['label'][0].numpy()  # Remove batch dimension
        
        # Calculate percentage of lesion voxels
        total_voxels = label.size
        lesion_voxels = np.sum(label > 0)
        percentage = (lesion_voxels / total_voxels) * 100
        percentages.append(percentage)
        
        # Find connected components
        labeled, num_components = ndimage.label(label)
        counts.append(num_components)
        
        # If there are components, calculate sizes
        if num_components > 0:
            component_sizes = ndimage.sum(label, labeled, range(1, num_components+1))
            avg_size = np.mean(component_sizes)
            sizes.append(avg_size)
            
            # Store min and max component size for this sample
            min_component_sizes.append(np.min(component_sizes) if len(component_sizes) > 0 else 0)
            max_component_sizes.append(np.max(component_sizes) if len(component_sizes) > 0 else 0)
            
            # Store all component sizes for overall distribution
            component_sizes_list.extend(component_sizes)
            component_sizes_by_sample.append(component_sizes)
        else:
            sizes.append(0)
            min_component_sizes.append(0)
            max_component_sizes.append(0)
            component_sizes_by_sample.append([])
    
    # Print detailed statistics
    print("\n===== LESION DISTRIBUTION ANALYSIS =====")
    print(f"\nLesion Count Statistics:")
    print(f"  Min: {min(counts)}, Max: {max(counts)}, Mean: {np.mean(counts):.2f}, Median: {np.median(counts)}")
    print(f"  Distribution of counts: {np.unique(counts, return_counts=True)}")
    
    print(f"\nLesion Size Statistics (voxels):")
    print(f"  Min: {min(sizes):.2f}, Max: {max(sizes):.2f}, Mean: {np.mean(sizes):.2f}, Median: {np.median(sizes):.2f}")
    
    if len(component_sizes_list) > 0:
        print(f"\nIndividual Lesion Component Size Statistics:")
        print(f"  Min: {min(component_sizes_list):.2f}, Max: {max(component_sizes_list):.2f}")
        print(f"  Mean: {np.mean(component_sizes_list):.2f}, Median: {np.median(component_sizes_list):.2f}")
        
        # Calculate percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = np.percentile(component_sizes_list, percentiles)
        print(f"\nLesion Component Size Percentiles (voxels):")
        for p, val in zip(percentiles, percentile_values):
            print(f"  {p}%: {val:.2f}")
    
    print(f"\nLesion Percentage Statistics (% of volume):")
    print(f"  Min: {min(percentages):.4f}%, Max: {max(percentages):.4f}%")
    print(f"  Mean: {np.mean(percentages):.4f}%, Median: {np.median(percentages):.4f}%")
    
    # Calculate ratio of largest component to total lesion volume
    max_component_ratio = []
    for i, components in enumerate(component_sizes_by_sample):
        if len(components) > 0 and np.sum(components) > 0:
            max_component_ratio.append(np.max(components) / np.sum(components))
    
    if len(max_component_ratio) > 0:
        print(f"\nRatio of largest component to total lesion volume:")
        print(f"  Min: {min(max_component_ratio):.4f}, Max: {max(max_component_ratio):.4f}")
        print(f"  Mean: {np.mean(max_component_ratio):.4f}, Median: {np.median(max_component_ratio):.4f}")
    
    # Save analysis results and create visualizations if output_dir is provided
    if output_dir:
        # Create distribution directory
        distribution_dir = os.path.join(output_dir, 'original_distribution')
        os.makedirs(distribution_dir, exist_ok=True)
        
        # Create histograms
        plt.figure(figsize=(15, 10))
        
        # 1. Distribution of lesion counts
        plt.subplot(2, 2, 1)
        # Calculate frequency of each count
        unique_counts, count_freq = np.unique(counts, return_counts=True)
        bars = plt.bar(unique_counts, count_freq)
        plt.title('Distribution of Lesion Counts')
        plt.xlabel('Number of Lesions per Sample')
        plt.ylabel('Frequency')
        plt.xticks(unique_counts)
        
        # Add frequency labels on top of bars
        for bar, freq in zip(bars, count_freq):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(freq), ha='center', va='bottom')
        
        # 2. Distribution of average lesion sizes
        plt.subplot(2, 2, 2)
        plt.hist(sizes, bins=20)
        plt.title('Distribution of Average Lesion Sizes')
        plt.xlabel('Average Size (voxels)')
        plt.ylabel('Frequency')
        
        # 3. Distribution of individual component sizes
        plt.subplot(2, 2, 3)
        if len(component_sizes_list) > 0:
            plt.hist(component_sizes_list, bins=20)
            plt.title('Distribution of Individual Lesion Sizes')
            plt.xlabel('Component Size (voxels)')
            plt.ylabel('Frequency')
            plt.xscale('log')  # Log scale for better visualization
        
        # 4. Distribution of lesion percentages
        plt.subplot(2, 2, 4)
        plt.hist(percentages, bins=20)
        plt.title('Distribution of Lesion Percentages')
        plt.xlabel('Percentage of Volume (%)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(distribution_dir, 'lesion_distribution.png'))
        
        # Create a more detailed visualization of component size distributions
        plt.figure(figsize=(15, 10))
        
        # 1. Box plot of component sizes by number of components
        plt.subplot(2, 2, 1)
        data_for_boxplot = []
        labels_for_boxplot = []
        
        # Group samples by number of components (up to 10)
        for count in range(1, min(11, max(counts) + 1)):
            sizes_for_count = [component_sizes_by_sample[i] for i in range(len(counts)) if counts[i] == count]
            if sizes_for_count:
                # Flatten the list
                flat_sizes = [size for sublist in sizes_for_count for size in sublist]
                if flat_sizes:
                    data_for_boxplot.append(flat_sizes)
                    labels_for_boxplot.append(str(count))
        
        if data_for_boxplot:
            plt.boxplot(data_for_boxplot, labels=labels_for_boxplot)
            plt.title('Distribution of Component Sizes by Number of Components')
            plt.xlabel('Number of Components')
            plt.ylabel('Component Size (voxels)')
            plt.yscale('log')  # Log scale for better visualization
        
        # 2. Scatter plot of number of components vs average size
        plt.subplot(2, 2, 2)
        plt.scatter(counts, sizes)
        plt.title('Number of Components vs Average Size')
        plt.xlabel('Number of Components')
        plt.ylabel('Average Component Size (voxels)')
        
        # 3. Ratio of largest component to total lesion volume
        plt.subplot(2, 2, 3)
        if max_component_ratio:
            plt.hist(max_component_ratio, bins=20)
            plt.title('Ratio of Largest Component to Total Volume')
            plt.xlabel('Ratio')
            plt.ylabel('Frequency')
        
        # 4. Component size percentiles
        plt.subplot(2, 2, 4)
        if len(component_sizes_list) > 0:
            component_percentiles = list(range(0, 101, 5))
            percentile_values = np.percentile(component_sizes_list, component_percentiles)
            plt.plot(component_percentiles, percentile_values, marker='o')
            plt.title('Component Size Percentiles')
            plt.xlabel('Percentile')
            plt.ylabel('Component Size (voxels)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(distribution_dir, 'component_size_distribution.png'))
        
        # Create a separate histogram just for lesion counts (larger, more detailed)
        plt.figure(figsize=(12, 8))
        bars = plt.bar(unique_counts, count_freq, width=0.7)
        plt.title('Distribution of Lesion Counts per Sample', fontsize=16)
        plt.xlabel('Number of Lesions', fontsize=14)
        plt.ylabel('Number of Samples', fontsize=14)
        plt.xticks(unique_counts, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add count and percentage labels on top of bars
        for bar, count, freq in zip(bars, unique_counts, count_freq):
            percentage = (freq / len(counts)) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f"{freq}\n({percentage:.1f}%)", ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(distribution_dir, 'lesion_count_histogram.png'))
        plt.close()
        
        # Save the raw data
        np.savez(os.path.join(distribution_dir, 'lesion_distribution.npz'), 
                 counts=counts, 
                 sizes=sizes, 
                 percentages=percentages,
                 component_sizes=component_sizes_list,
                 min_component_sizes=min_component_sizes,
                 max_component_sizes=max_component_sizes,
                 max_component_ratio=max_component_ratio if max_component_ratio else [])
        
        # Save a text summary
        with open(os.path.join(distribution_dir, 'summary.txt'), 'w') as f:
            f.write("===== LESION DISTRIBUTION ANALYSIS =====\n\n")
            
            f.write("Lesion Count Statistics:\n")
            f.write(f"  Min: {min(counts)}, Max: {max(counts)}, Mean: {np.mean(counts):.2f}, Median: {np.median(counts)}\n")
            
            f.write("\nCount Distribution:\n")
            for count, freq in zip(unique_counts, count_freq):
                percentage = (freq / len(counts)) * 100
                f.write(f"  {count} lesions: {freq} samples ({percentage:.1f}%)\n")
            
            f.write("\nLesion Size Statistics (voxels):\n")
            f.write(f"  Min: {min(sizes):.2f}, Max: {max(sizes):.2f}, Mean: {np.mean(sizes):.2f}, Median: {np.median(sizes):.2f}\n")
            
            if len(component_sizes_list) > 0:
                f.write("\nIndividual Lesion Component Size Statistics:\n")
                f.write(f"  Min: {min(component_sizes_list):.2f}, Max: {max(component_sizes_list):.2f}\n")
                f.write(f"  Mean: {np.mean(component_sizes_list):.2f}, Median: {np.median(component_sizes_list):.2f}\n")
                
                f.write("\nLesion Component Size Percentiles (voxels):\n")
                for p, val in zip(percentiles, percentile_values):
                    f.write(f"  {p}%: {val:.2f}\n")
            
            f.write("\nLesion Percentage Statistics (% of volume):\n")
            f.write(f"  Min: {min(percentages):.4f}%, Max: {max(percentages):.4f}%\n")
            f.write(f"  Mean: {np.mean(percentages):.4f}%, Median: {np.median(percentages):.4f}%\n")
            
            if len(max_component_ratio) > 0:
                f.write("\nRatio of largest component to total lesion volume:\n")
                f.write(f"  Min: {min(max_component_ratio):.4f}, Max: {max(max_component_ratio):.4f}\n")
                f.write(f"  Mean: {np.mean(max_component_ratio):.4f}, Median: {np.median(max_component_ratio):.4f}\n")
        
        print(f"\nDetailed analysis saved to {distribution_dir}")
    
    return counts, sizes, percentages, component_sizes_list

def compute_lesion_distribution_loss(generated, real_labels, atlas):
    """Podporuje realistickou distribuci počtu a velikosti lézí"""
    # Práh pro binární léze
    gen_binary = (generated > 0.5).float()
    loss = 0.0
    batch_size = generated.size(0)
    
    for i in range(batch_size):
        # Analýza reálného a generovaného vzorku
        real_np = real_labels[i, 0].cpu().numpy()
        gen_np = gen_binary[i, 0].cpu().numpy()
        
        # Počet komponent
        real_labeled, real_count = ndimage.label(real_np)
        gen_labeled, gen_count = ndimage.label(gen_np)
        
        if real_count == 0:  # Přeskočit prázdné vzorky
            continue
            
        # 1. Penalizace za špatný počet lézí
        # Použijeme log pro zmírnění vlivu vysokých počtů
        count_ratio = abs(np.log1p(real_count) - np.log1p(gen_count)) / np.log1p(max(real_count, 1))
        count_loss = torch.tensor(count_ratio, device=generated.device)
        
        # 2. Penalizace za nesprávnou distribuci velikostí
        if gen_count > 0 and real_count > 0:
            # Velikosti komponent
            real_sizes = ndimage.sum(real_np, real_labeled, range(1, real_count+1))
            gen_sizes = ndimage.sum(gen_np, gen_labeled, range(1, gen_count+1))
            
            # Poměr největší léze k celkovému objemu
            real_ratio = np.max(real_sizes) / np.sum(real_sizes) if np.sum(real_sizes) > 0 else 0
            gen_ratio = np.max(gen_sizes) / np.sum(gen_sizes) if np.sum(gen_sizes) > 0 else 0
            
            # Penalizace za rozdíl v dominanci
            dominance_loss = abs(real_ratio - gen_ratio)
            
            # Porovnání histogramů velikostí (zjednodušeně)
            # Normalizujeme velikosti do 10 binů
            real_hist = np.histogram(real_sizes, bins=10, range=(0, np.max(real_sizes)*1.1))[0]
            real_hist = real_hist / (np.sum(real_hist) + 1e-8)
            
            gen_hist = np.histogram(gen_sizes, bins=10, range=(0, np.max(real_sizes)*1.1))[0]
            gen_hist = gen_hist / (np.sum(gen_hist) + 1e-8)
            
            # KL divergence
            hist_loss = 0.0
            for j in range(10):
                if real_hist[j] > 0 and gen_hist[j] > 0:
                    hist_loss += real_hist[j] * np.log(real_hist[j] / (gen_hist[j] + 1e-8))
            
            size_loss = torch.tensor(dominance_loss + hist_loss, device=generated.device)
        else:
            size_loss = torch.tensor(1.0, device=generated.device)  # Vysoká penalizace při chybějících lézích
        
        loss += 0.7 * count_loss + 0.3 * size_loss
    
    return loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=generated.device)

# Training function
def train(generator, discriminator, dataloader, num_epochs, device, output_dir):
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.98)
    scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.98)
    
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
            # Train discriminator 1 time for each generator update
            for _ in range(1):  # Reduced from 3 to balance training
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
            
            # Sparsity loss (reduced weight to prevent overemphasis on sparsity)
            sparse_loss = compute_sparsity_loss(fake_labels, atlas)
            
            # Empty image penalty (new)
            empty_penalty = compute_empty_penalty(fake_labels, atlas)
            
            # Shape consistency loss (new)
            try:
                shape_loss = compute_shape_consistency_loss(fake_labels, real_labels, atlas)
            except Exception as e:
                print(f"Error computing shape loss: {e}")
                shape_loss = torch.tensor(0.0, device=device)
            
            # Přidat do výpočtu generator loss
            lesion_dist_loss = compute_lesion_distribution_loss(fake_labels, real_labels, atlas)
            
            # Upravit váhy v celkové loss funkci
            g_loss = (0.5 * g_adv_loss + 
                     15.0 * freq_loss +
                     0.05 * sparse_loss +
                     20.0 * empty_penalty +
                     5.0 * shape_loss +
                     10.0 * lesion_dist_loss)  # Vysoká váha pro loss distribuci lézí
            
            g_loss.backward()
            optimizer_G.step()
            
            # Track losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            # Print detailed losses occasionally
            if batch_size % 10 == 0:
                non_zero_percent = torch.mean((fake_labels > 0.1).float()) * 100
                print(f"  G_adv: {g_adv_loss.item():.4f}, Freq: {freq_loss.item():.4f}, "
                      f"Sparse: {sparse_loss.item():.4f}, Empty: {empty_penalty.item():.4f}, "
                      f"Shape: {shape_loss.item():.4f}, Non-zero: {non_zero_percent.item():.2f}%")
        
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
        if (epoch + 1) % 4 == 0:
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

# Generation function to create new images from a trained model
def generate_samples(model_path, lesion_atlas_path, output_dir, num_samples=10, threshold=0.5, show_raw_distribution=True, match_distribution=False, target_percentiles=None, min_threshold=0.01):
    """
    Generate new lesion samples using a trained generator model
    
    Args:
        model_path: Path to the saved model checkpoint
        lesion_atlas_path: Path to the lesion atlas .nii file
        output_dir: Directory to save the generated samples
        num_samples: Number of samples to generate
        threshold: Threshold for binarizing the output (starting value)
        show_raw_distribution: Whether to analyze and visualize raw distribution before thresholding
        match_distribution: Whether to try to match the distribution of lesion counts from the original dataset
        target_percentiles: List of target percentiles for lesion percentage (if None, defaults will be used)
        min_threshold: Minimum threshold to try if standard threshold gives empty results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a separate directory for distribution analysis
    if show_raw_distribution:
        distribution_dir = os.path.join(output_dir, 'raw_distribution')
        os.makedirs(distribution_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    generator = Generator().to(device)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Define target distribution if matching is requested
    if match_distribution and target_percentiles is None:
        # These are example values - ideally should be calculated from the original dataset
        # Format: [(percentile, min_percentage, max_percentage), ...]
        target_percentiles = [
            (0.2, 0.01, 0.1),    # 20% of samples should have 0.01-0.1% lesions
            (0.4, 0.1, 0.5),     # 40% of samples should have 0.1-0.5% lesions
            (0.3, 0.5, 1.0),     # 30% of samples should have 0.5-1.0% lesions
            (0.1, 1.0, 2.5)      # 10% of samples should have 1.0-2.5% lesions
        ]
    
    # Load lesion atlas
    nifti_img = nib.load(lesion_atlas_path)
    lesion_atlas = nifti_img.get_fdata()
    lesion_atlas = np.clip(lesion_atlas, 0, 1)
    
    # Get original affine for saving nifti files
    affine = nifti_img.affine
    
    # Convert to tensor
    atlas_tensor = torch.from_numpy(lesion_atlas).float().unsqueeze(0).unsqueeze(0)
    
    # Normalize atlas
    atlas_tensor = atlas_tensor / 0.34
    atlas_tensor = atlas_tensor.to(device)
    
    # Track distribution of generated samples
    generated_percentages = []
    
    # Vytvoříme set pro sledování jedinečnosti generovaných vzorků (zabránění duplicitám)
    generated_fingerprints = set()
    
    # Funkce pro generování rozmanitých typů šumu
    def generate_diverse_noise(batch_size=1, z_dim=100, device=device):
        # Základní gaussovský šum
        noise = torch.randn(batch_size, z_dim, 1, 1, 1, device=device)
        
        # Přidat více rozmanitosti pomocí různých škál
        scale = torch.rand(batch_size, 1, 1, 1, 1, device=device) * 2.0 + 0.5  # Škála mezi 0.5 a 2.5
        noise = noise * scale
        
        # Náhodně přidat perturbace pro další rozmanitost
        if random.random() > 0.3:
            # Přidat lokalizovanou perturbaci na náhodných pozicích
            num_perturbations = random.randint(1, 3)
            for _ in range(num_perturbations):
                pos = random.randint(0, z_dim-1)
                length = random.randint(5, 15)
                end_pos = min(pos + length, z_dim)
                
                # Vytvořit a aplikovat náhodnou perturbaci
                perturbation = torch.randn(batch_size, end_pos - pos, 1, 1, 1, device=device) * random.uniform(0.5, 3.0)
                noise[:, pos:end_pos] += perturbation
        
        # Občas přidat mixování s uniformním šumem
        if random.random() > 0.7:
            uniform_noise = (torch.rand(batch_size, z_dim, 1, 1, 1, device=device) * 2 - 1) * 2  # Uniformní šum v rozsahu [-2, 2]
            mix_ratio = random.uniform(0.1, 0.3)
            noise = noise * (1 - mix_ratio) + uniform_noise * mix_ratio
            
        # Občas přidat nenulovou střední hodnotu pro lepší podmínky
        if random.random() > 0.6:
            bias = (torch.rand(batch_size, 1, 1, 1, 1, device=device) - 0.5) * 0.4  # Bias v rozsahu [-0.2, 0.2]
            noise += bias
            
        return noise
    
    # Funkce pro výpočet "otisku" generovaného vzorku (pro porovnávání jedinečnosti)
    def compute_sample_fingerprint(sample, threshold):
        # Převést na binární podle thresholdu
        binary = (sample > threshold).astype(np.float32)
        
        # Najít a serializovat pozice lézí
        labeled, num_components = ndimage.label(binary)
        component_coords = []
        
        for i in range(1, num_components + 1):
            # Získat souřadnice pixelů pro každou lézi
            component_mask = (labeled == i)
            coords = np.where(component_mask)
            
            # Použít centroid (střední bod) jako identifikátor pozice
            if len(coords[0]) > 0:  # Ujistit se, že komponenta není prázdná
                centroid = (
                    np.mean(coords[0]), 
                    np.mean(coords[1]), 
                    np.mean(coords[2])
                )
                component_coords.append(centroid)
        
        # Setřídit podle pozice pro konzistentní porovnávání
        component_coords.sort()
        
        # Vytvořit hash otisk
        fingerprint = ""
        for c in component_coords:
            # Zaokrouhlit na jedno desetinné místo pro tolerování mírných odchylek
            fingerprint += f"{c[0]:.1f},{c[1]:.1f},{c[2]:.1f}|"
            
        return fingerprint
    
    # Generate samples with multiple threshold attempts if empty
    with torch.no_grad():
        i = 0
        max_attempts = num_samples * 3  # Maximální počet pokusů pro případ, že budeme muset přeskočit duplicity
        attempt_count = 0
        
        while i < num_samples and attempt_count < max_attempts:
            attempt_count += 1
            
            # Determine target percentage range for this sample if matching distribution
            target_min_pct = None
            target_max_pct = None
            if match_distribution and target_percentiles:
                # Randomly select a percentile range based on the distribution
                rand_val = random.random()
                cumulative_prob = 0
                for prob, min_pct, max_pct in target_percentiles:
                    cumulative_prob += prob
                    if rand_val <= cumulative_prob:
                        target_min_pct = min_pct
                        target_max_pct = max_pct
                        break
                
                print(f"Sample {i+1}: Targeting {target_min_pct:.3f}% - {target_max_pct:.3f}% lesion coverage")
            
            # Cílové distribuce počtu lézí
            lesion_count_targets = [
                (0.30, 1, 10),      # 30% vzorků: 1-10 lézí
                (0.40, 10, 30),     # 40% vzorků: 10-30 lézí
                (0.20, 30, 80),     # 20% vzorků: 30-80 lézí
                (0.10, 80, 200)     # 10% vzorků: 80-200 lézí
            ]

            # Vybrat cílový počet lézí
            rand_val = random.random()
            cumulative_prob = 0
            target_min_count = None
            target_max_count = None
            for prob, min_count, max_count in lesion_count_targets:
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    target_min_count = min_count
                    target_max_count = max_count
                    break

            print(f"Targeting {target_min_count}-{target_max_count} lesions")

            # Vygenerovat několik vzorků a vybrat ten nejbližší cílovému počtu
            best_match = None
            best_match_diff = float('inf')
            best_noise = None  # Změna: místo noise_scale si budeme pamatovat celý noise tensor
            best_coverage = 0.0
            best_fingerprint = None
            
            # Původní threshold pro začátek
            current_threshold = threshold
            
            # Nejprve se pokusíme dosáhnout cílového pokrytí
            reached_target_coverage = False
            
            # Nejlepší pokus pro každou metriku
            best_coverage_match = None
            best_coverage_threshold = threshold
            best_coverage_percentage = 0.0
            best_coverage_components = 0
            best_coverage_diff = float('inf')
            
            best_lesion_match = None
            best_lesion_threshold = threshold
            best_lesion_components = 0
            best_lesion_percentage = 0.0
            best_lesion_diff = float('inf')

            # Zkusit více šumů a vybrat ten s nejlepším pokrytím a počtem lézí
            for noise_attempt in range(30):  # Zvýšen počet pokusů
                # Používáme rozmanité šumy místo standardního Gaussovského šumu
                noise = generate_diverse_noise(batch_size=1, z_dim=100, device=device)
                
                # Generovat lesion map
                fake_label = generator(noise, atlas_tensor)
                
                # Získat raw data pro analýzu
                fake_np_raw = fake_label[0, 0].cpu().numpy()
                
                # Zkontrolujeme maximum hodnoty - pokud je nízké, snížíme threshold
                atlas_np = atlas_tensor[0, 0].cpu().numpy()
                mask = atlas_np > 0
                values_in_mask = fake_np_raw[mask]
                
                if len(values_in_mask) > 0:
                    max_val = values_in_mask.max()
                    
                    # Postupně snižujeme threshold, dokud nedosáhneme cílového pokrytí
                    # Vytvoříme několik kandidátních thresholdů
                    # Zkoušíme více různých úrovní thresholdu pro každý vygenerovaný vzorek
                    thresholds_to_try = []
                    
                    # Adaptivní threshold založený na maximální hodnotě
                    if max_val < threshold and max_val > 0:
                        adaptive_threshold = max(min_threshold, max_val * 0.8)
                        thresholds_to_try.append(adaptive_threshold)
                    
                    # Přidáme další kandidátní thresholdy - celá škála od min_threshold do max_val
                    step = (max_val - min_threshold) / 10
                    for t in range(10):
                        candidate_threshold = min_threshold + t * step
                        if candidate_threshold not in thresholds_to_try and candidate_threshold < max_val:
                            thresholds_to_try.append(candidate_threshold)
                    
                    # Vždy přidáme velmi nízký threshold pro zajištění maximálního pokrytí
                    thresholds_to_try.append(min_threshold)
                    
                    # Seřadíme thresholdy sestupně (začneme s vyššími hodnotami)
                    thresholds_to_try.sort(reverse=True)
                    
                    # Vyzkoušíme různé thresholdy a najdeme nejlepší match
                    for try_threshold in thresholds_to_try:
                        # Threshold a počet lézí s tímto thresholdem
                        fake_np = (fake_np_raw > try_threshold).astype(np.float32)
                        labeled, num_components = ndimage.label(fake_np)
                        
                        # Vypočítat procento pokrytí lézemi
                        coverage_pct = (np.sum(fake_np) / np.sum(mask)) * 100
                        
                        print(f"  Pokus {noise_attempt+1}, threshold {try_threshold:.4f}: {num_components} lézí, {coverage_pct:.4f}% pokrytí")
                        
                        # Vypočítat otisk pro kontrolu jedinečnosti
                        current_fingerprint = compute_sample_fingerprint(fake_np_raw, try_threshold)
                        
                        # Přeskočit, pokud jsme generovali podobný vzorek dříve
                        if current_fingerprint in generated_fingerprints:
                            print(f"  ⚠️ Nalezena podobná léze, zkouším jiný šum...")
                            continue
                        
                        # Kontrola, zda jsme dosáhli cílového pokrytí
                        if target_min_pct is not None and target_max_pct is not None:
                            coverage_in_range = target_min_pct <= coverage_pct <= target_max_pct
                            # Vypočítat, jak daleko jsme od cílového rozmezí
                            if coverage_pct < target_min_pct:
                                coverage_diff = target_min_pct - coverage_pct
                            elif coverage_pct > target_max_pct:
                                coverage_diff = coverage_pct - target_max_pct
                            else:
                                coverage_diff = 0.0
                                
                            # Pokud je toto nejlepší pokrytí, které jsme našli
                            if coverage_diff < best_coverage_diff:
                                best_coverage_diff = coverage_diff
                                best_coverage_match = fake_label
                                best_coverage_threshold = try_threshold
                                best_coverage_percentage = coverage_pct
                                best_coverage_components = num_components
                                best_fingerprint = current_fingerprint
                                
                            # Pokud jsme dosáhli cílového pokrytí, zkontrolujeme počet lézí
                            if coverage_in_range:
                                reached_target_coverage = True
                                lesion_diff = min(abs(num_components - target_min_count), abs(num_components - target_max_count))
                                
                                # Pokud je toto nejlepší počet lézí v cílovém rozmezí pokrytí
                                if lesion_diff < best_lesion_diff:
                                    best_lesion_diff = lesion_diff
                                    best_lesion_match = fake_label
                                    best_lesion_threshold = try_threshold
                                    best_lesion_components = num_components
                                    best_lesion_percentage = coverage_pct
                                    best_fingerprint = current_fingerprint
                                    
                                # Pokud jsme našli perfektní shodu (v rozmezí pokrytí i počtu lézí)
                                if target_min_count <= num_components <= target_max_count:
                                    best_match = fake_label
                                    best_noise = noise  # Změna: uložíme celý noise tensor místo noise_scale
                                    current_threshold = try_threshold
                                    best_coverage = coverage_pct
                                    best_fingerprint = current_fingerprint
                                    print(f"  NALEZENA PERFEKTNÍ SHODA: {num_components} lézí, {coverage_pct:.4f}% pokrytí")
                                    break
                                    
                        # Pokud nemáme cílové pokrytí, hodnotíme jen podle počtu lézí
                        else:
                            if target_min_count <= num_components <= target_max_count:
                                best_match = fake_label
                                best_noise = noise  # Změna: uložíme celý noise tensor místo noise_scale
                                current_threshold = try_threshold
                                best_coverage = coverage_pct
                                best_fingerprint = current_fingerprint
                                break
                    
                # Pokud jsme našli perfektní shodu, přerušíme cyklus pokusů        
                if best_match is not None:
                    break
            
            # Po vyzkoušení všech šumů a thresholdů, vybereme nejlepší dostupnou shodu
            if best_match is None:
                # Pokud jsme alespoň dosáhli cílového pokrytí, použijeme ten vzorek
                if best_lesion_match is not None:
                    best_match = best_lesion_match
                    current_threshold = best_lesion_threshold
                    best_fingerprint = best_fingerprint
                    print(f"  Použijeme vzorek s nejlepším počtem lézí v cílovém rozmezí pokrytí: {best_lesion_components} lézí, {best_lesion_percentage:.4f}% pokrytí")
                # Jinak použijeme vzorek s nejbližším pokrytím k cílovému rozmezí
                elif best_coverage_match is not None:
                    best_match = best_coverage_match
                    current_threshold = best_coverage_threshold
                    best_fingerprint = best_fingerprint
                    print(f"  Použijeme vzorek s nejbližším pokrytím k cílovému rozmezí: {best_coverage_components} lézí, {best_coverage_percentage:.4f}% pokrytí")
            
            # Pokud nemáme žádnou shodu nebo je vzorek duplicitní, přeskočíme a zkusíme znovu
            if best_match is None or best_fingerprint in generated_fingerprints:
                print(f"  ⚠️ Nepodařilo se najít vhodný vzorek, zkouším znovu...")
                continue
                
            # Přidáme otisk do set duplicit pro příští kontrolu
            generated_fingerprints.add(best_fingerprint)
            
            # Ještě jeden pokus s "nejlepším" šumem, ale agresivnějším snížením thresholdu
            # jen pokud jsme stále nedosáhli cílového pokrytí
            if best_match is not None and target_min_pct is not None and best_coverage < target_min_pct:
                fake_np_raw = best_match[0, 0].cpu().numpy()
                mask = atlas_np > 0
                values_in_mask = fake_np_raw[mask]
                
                if len(values_in_mask) > 0:
                    max_val = values_in_mask.max()
                    
                    # Velmi agresivní snížení, hledáme threshold který nám dá cílové pokrytí
                    print(f"  POSLEDNÍ POKUS: Hledáme threshold pro dosažení cílového pokrytí {target_min_pct:.4f}%")
                    
                    # Zkusíme sérii thresholdů a najdeme ten, který dá požadované pokrytí
                    percentile_values = np.percentile(values_in_mask, [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
                    
                    for percentile_val in percentile_values:
                        very_low_threshold = max(min_threshold, percentile_val)
                        # Pouze pokud je nižší než současný threshold
                        if very_low_threshold >= current_threshold:
                            continue
                            
                        fake_np = (fake_np_raw > very_low_threshold).astype(np.float32)
                        labeled, num_components = ndimage.label(fake_np)
                        coverage_pct = (np.sum(fake_np) / np.sum(mask)) * 100
                        
                        print(f"  Zkoušíme threshold {very_low_threshold:.6f}: {num_components} lézí, {coverage_pct:.4f}% pokrytí")
                        
                        # Pokud jsme dosáhli cílového pokrytí, použijeme tento threshold
                        if coverage_pct >= target_min_pct:
                            current_threshold = very_low_threshold
                            print(f"  Našli jsme threshold {very_low_threshold:.6f} pro dosažení pokrytí {coverage_pct:.4f}%")
                            break
                            
                    # Pokud je potřeba, vygenerujeme znovu vzorek s použitím nejlepšího šumu a nového thresholdu
                    if best_noise is not None and current_threshold != threshold:
                        # Regenerovat lesion map s nejlepším šumem
                        fake_label = generator(best_noise, atlas_tensor)
            
            # Použít vybraný vzorek a threshold pro finální zpracování
            fake_label = best_match
            fake_np_raw = fake_label[0, 0].cpu().numpy()
            fake_np = (fake_np_raw > current_threshold).astype(np.float32)
            mask = atlas_np > 0
            
            if np.sum(mask) > 0:
                coverage_percentage = (np.sum(fake_np[mask]) / np.sum(mask)) * 100
                _, final_num_components = ndimage.label(fake_np)
                print(f"Finální výsledek: {final_num_components} lézí, {coverage_percentage:.4f}% pokrytí (threshold: {current_threshold:.4f})")
                generated_percentages.append(coverage_percentage)
            else:
                generated_percentages.append(0)
            
            # Analyze distribution if requested
            if show_raw_distribution:
                # Extract values where the mask is non-zero
                values_in_mask = fake_np_raw[mask]
                
                # Calculate statistics
                if len(values_in_mask) > 0:
                    min_val = values_in_mask.min()
                    max_val = values_in_mask.max()
                    mean_val = values_in_mask.mean()
                    median_val = np.median(values_in_mask)
                    
                    # Calculate percentiles
                    percentiles = np.percentile(values_in_mask, [1, 5, 10, 25, 50, 75, 90, 95, 99])
                    
                    # Create plots
                    plt.figure(figsize=(15, 10))
                    
                    # 1. Raw value visualization (middle slice)
                    plt.subplot(2, 2, 1)
                    mid_z = fake_np_raw.shape[2] // 2
                    
                    # Find slice with highest values if middle is empty
                    if np.max(fake_np_raw[:, :, mid_z]) < current_threshold * 0.5:
                        slice_max_vals = [np.max(fake_np_raw[:, :, z]) for z in range(fake_np_raw.shape[2])]
                        if max(slice_max_vals) > current_threshold * 0.5:
                            mid_z = np.argmax(slice_max_vals)
                    
                    plt.imshow(fake_np_raw[:, :, mid_z], cmap='hot', vmin=0, vmax=max(max_val, 0.1))
                    plt.colorbar()
                    plt.title(f'Raw Values (Slice {mid_z})')
                    plt.axis('off')
                    
                    # 2. Histogram of values
                    plt.subplot(2, 2, 2)
                    plt.hist(values_in_mask, bins=50, range=(0, max(max_val*1.1, 0.1)), alpha=0.7)
                    plt.axvline(x=current_threshold, color='r', linestyle='--', label=f'Threshold={current_threshold:.4f}')
                    plt.legend()
                    plt.title('Histogram of Raw Values (in Atlas Mask)')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    
                    # 3. Percentile plot (as CDF)
                    plt.subplot(2, 2, 3)
                    values_sorted = np.sort(values_in_mask)
                    p = 1. * np.arange(len(values_in_mask)) / (len(values_in_mask) - 1)
                    plt.plot(values_sorted, p)
                    plt.axvline(x=current_threshold, color='r', linestyle='--')
                    plt.title('Cumulative Distribution Function')
                    plt.xlabel('Value')
                    plt.ylabel('Percentile')
                    plt.grid(True)
                    
                    # 4. Display statistics
                    plt.subplot(2, 2, 4)
                    plt.axis('off')
                    stats_text = f"""
Statistics for Sample {i+1}:
-------------------------
Min: {min_val:.6f}
Max: {max_val:.6f}
Mean: {mean_val:.6f}
Median: {median_val:.6f}

Percentiles:
  1%: {percentiles[0]:.6f}
  5%: {percentiles[1]:.6f}
 10%: {percentiles[2]:.6f}
 25%: {percentiles[3]:.6f}
 50%: {percentiles[4]:.6f}
 75%: {percentiles[5]:.6f}
 90%: {percentiles[6]:.6f}
 95%: {percentiles[7]:.6f}
 99%: {percentiles[8]:.6f}

Non-zero voxels (>0.001): {np.sum(values_in_mask > 0.001) / len(values_in_mask) * 100:.4f}%
Voxels above threshold ({current_threshold:.4f}): {np.sum(values_in_mask > current_threshold) / len(values_in_mask) * 100:.4f}%

Počet lézí: {final_num_components}
Pokrytí: {coverage_percentage:.4f}%
                    """
                    plt.text(0.01, 0.99, stats_text, fontsize=10, va='top', family='monospace')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(distribution_dir, f'sample_{i+1}_distribution.png'))
                    plt.close()
                    
                    # Also save the raw values as a nifti for further analysis
                    raw_nii = nib.Nifti1Image(fake_np_raw, affine)
                    nib.save(raw_nii, os.path.join(distribution_dir, f'sample_{i+1}_raw.nii.gz'))
                    
                    # Print some statistics
                    print(f"Sample {i+1} statistics in atlas mask:")
                    print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}, Mean: {mean_val:.6f}, Median: {median_val:.6f}")
                    print(f"  Above threshold ({current_threshold:.4f}): {np.sum(values_in_mask > current_threshold) / len(values_in_mask) * 100:.4f}%")
            
            # Compute percentage of non-zero voxels
            non_zero_percentage = np.sum(fake_np) / np.sum(mask) if np.sum(mask) > 0 else 0
            print(f"Sample {i+1} non-zero voxels after thresholding: {non_zero_percentage * 100:.4f}%")
            
            # Save as .nii file
            fake_nii = nib.Nifti1Image(fake_np, affine)
            nib.save(fake_nii, os.path.join(output_dir, f'sample_{i+1}.nii.gz'))
            
            # Also save middle slice as image for quick preview
            mid_z = fake_np.shape[2] // 2
            
            # Find the z-slice with the most lesions if middle slice is empty
            if np.sum(fake_np[:, :, mid_z]) == 0:
                lesion_sums = [np.sum(fake_np[:, :, z]) for z in range(fake_np.shape[2])]
                if max(lesion_sums) > 0:
                    mid_z = np.argmax(lesion_sums)
                    print(f"Middle slice empty, using slice {mid_z} with most lesions")
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(fake_np[:, :, mid_z], cmap='gray')
            plt.title(f'Sample {i+1} - Axial (Threshold: {current_threshold:.4f})')
            plt.axis('off')
            
            # Find the y-slice with the most lesions
            lesion_sums_y = [np.sum(fake_np[:, y, :]) for y in range(fake_np.shape[1])]
            mid_y = fake_np.shape[1] // 2
            if max(lesion_sums_y) > 0:
                mid_y = np.argmax(lesion_sums_y)
            
            plt.subplot(1, 2, 2)
            plt.imshow(fake_np[:, mid_y, :], cmap='gray')
            plt.title(f'Sample {i+1} - Sagittal (Threshold: {current_threshold:.4f})')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'))
            plt.close()
            
            # Pokračovat na další sample
            i += 1
    
    print(f"Generated {num_samples} samples in {output_dir}")
    if show_raw_distribution:
        print(f"Raw distribution analysis saved in {distribution_dir}")
        
    # Zobrazit výsledný přehled jedinečnosti
    print(f"\nUniqueness Report:")
    print(f"  Total generated samples: {num_samples}")
    print(f"  Total attempts needed: {attempt_count}")
    print(f"  Ratio samples/attempts: {num_samples/attempt_count:.2f}")
    print(f"  Unique sample patterns: {len(generated_fingerprints)}")

# Main function
def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = HIEDataset(args.labels_dir, args.lesion_atlas_path)
    
    # Analyze lesion distribution if requested
    if args.analyze_distribution:
        counts, sizes, percentages, component_sizes_list = analyze_lesion_distribution(dataset, args.output_dir)
        
        # Save results
        distribution_dir = os.path.join(args.output_dir, 'original_distribution')
        os.makedirs(distribution_dir, exist_ok=True)
        
        # Save the histogram
        plt.savefig(os.path.join(distribution_dir, 'lesion_distribution.png'))
        plt.close()
        
        # Save the raw data
        np.savez(os.path.join(distribution_dir, 'lesion_distribution.npz'), 
                 counts=counts, sizes=sizes, percentages=percentages,
                 component_sizes=component_sizes_list)
        
        # If only analysis was requested, exit
        if args.analyze_only:
            return
    
    # Load target distribution if available and generation is requested
    target_percentiles = None
    if args.generate and args.match_distribution:
        distribution_file = os.path.join(args.output_dir, 'original_distribution', 'lesion_distribution.npz')
        if os.path.exists(distribution_file):
            data = np.load(distribution_file)
            percentages = data['percentages']
            
            # Create distribution percentiles from the data
            p = np.percentile(percentages, [20, 60, 90])
            target_percentiles = [
                (0.2, 0, p[0]),
                (0.4, p[0], p[1]),
                (0.3, p[1], p[2]),
                (0.1, p[2], max(percentages))
            ]
            print("Using target distribution from analysis:")
            for prob, min_pct, max_pct in target_percentiles:
                print(f"  {prob*100:.0f}% of samples: {min_pct:.4f}% - {max_pct:.4f}% lesion coverage")
    
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
    parser.add_argument("--generate", action="store_true", help="Generate samples from a trained model")
    parser.add_argument("--model_path", type=str, help="Path to the trained model for generation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing the output")
    parser.add_argument("--min_threshold", type=float, default=0.01, help="Minimum threshold to try if standard threshold gives empty results")
    parser.add_argument("--no_distribution", action="store_true", help="Skip raw distribution analysis")
    parser.add_argument("--analyze_distribution", action="store_true", help="Analyze lesion distribution in the dataset")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze distribution, don't train or generate")
    parser.add_argument("--match_distribution", action="store_true", help="Try to match original lesion distribution")
    
    args = parser.parse_args()
    
    if args.generate:
        if args.model_path is None:
            print("Error: --model_path must be specified when using --generate")
            exit(1)
        generate_samples(args.model_path, args.lesion_atlas_path, args.output_dir, 
                        args.num_samples, args.threshold, not args.no_distribution,
                        args.match_distribution, None, args.min_threshold)
    else:
        main(args)
