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
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, generate_binary_structure, gaussian_filter
from torchvision import transforms
import time  # Import time for performance measurement


class PerlinNoiseGenerator:
    def __init__(self, octaves=4, persistence=0.5, lacunarity=2.0, repeat=1024, seed=None):
        """
        Generate 3D Perlin noise tailored for lesion generation using GPU acceleration
        
        Args:
            octaves (int): Number of octaves for the noise (higher = more detail)
            persistence (float): How much each octave contributes to the overall shape (0-1)
            lacunarity (float): How much detail is added at each octave
            repeat (int): Period after which the noise pattern repeats
            seed (int): Random seed for reproducibility
        """
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.repeat = repeat
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        
        # Set the random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
    
    def generate_3d_perlin(self, shape, scale=0.1, device='cuda'):
        """
        Generate 3D Perlin noise using GPU acceleration
        
        Args:
            shape (tuple): Shape of the noise volume (D, H, W)
            scale (float): Scale of the noise (smaller = smoother)
            device (str): Device to place the tensor on
            
        Returns:
            torch.Tensor: 3D Perlin noise tensor
        """
        # Start timing
        start_time = time.time()
        
        D, H, W = shape
        
        # Print the volume size being generated
        # print(f"Generating noise for volume size: {D}x{H}x{W}")
        
        # Set random seed for reproducibility for this specific call
        torch.manual_seed(self.seed)
        
        # Generate multiple frequency components that will be combined
        result = torch.zeros((D, H, W), dtype=torch.float32, device=device)
        
        # Define frequency components based on scale
        frequencies = [1.0]
        for i in range(1, self.octaves):
            frequencies.append(frequencies[-1] * self.lacunarity)
        
        # Define amplitude for each octave
        amplitudes = [1.0]
        for i in range(1, self.octaves):
            amplitudes.append(amplitudes[-1] * self.persistence)
        
        max_amplitude = sum(amplitudes)
        
        # Create meshgrid for coordinates
        z = torch.linspace(0, scale, D, device=device)
        y = torch.linspace(0, scale, H, device=device)
        x = torch.linspace(0, scale, W, device=device)
        
        z_grid, y_grid, x_grid = torch.meshgrid(z, y, x, indexing='ij')
        
        # Generate a random phase offset tensor for this specific noise generation
        # This is crucial for making different seeds produce different patterns
        random_tensor = torch.rand(3, 3, device=device)
        
        # Generate noise by adding different frequency components
        for octave in range(self.octaves):
            # Calculate frequency and amplitude for this octave
            freq = frequencies[octave]
            amplitude = amplitudes[octave]
            
            # Scale coordinates by frequency
            phase_x = x_grid * freq
            phase_y = y_grid * freq
            phase_z = z_grid * freq
            
            # Use the random tensor to create truly random phase offsets based on the seed
            phase_offset_x = random_tensor[0, 0] * 6.28 + octave * 0.5
            phase_offset_y = random_tensor[0, 1] * 6.28 + octave * 1.2
            phase_offset_z = random_tensor[0, 2] * 6.28 + octave * 0.8
            
            # We'll also randomize the frequency multipliers to get more varied patterns
            freq_multiplier_x = 1.0 + random_tensor[1, 0] * 0.5
            freq_multiplier_y = 1.0 + random_tensor[1, 1] * 0.5
            freq_multiplier_z = 1.0 + random_tensor[1, 2] * 0.5
            
            # Generate 3D noise using sine waves with randomized phases
            noise_x = torch.sin(2 * np.pi * phase_x * freq_multiplier_x + phase_offset_x)
            noise_y = torch.sin(2 * np.pi * phase_y * freq_multiplier_y + phase_offset_y)
            noise_z = torch.sin(2 * np.pi * phase_z * freq_multiplier_z + phase_offset_z)
            
            # Combine the noise patterns (with random weights)
            weight_x = 0.3 + random_tensor[2, 0] * 0.4
            weight_y = 0.3 + random_tensor[2, 1] * 0.4
            weight_z = 0.3 + random_tensor[2, 2] * 0.4
            weight_sum = weight_x + weight_y + weight_z
            
            noise = (noise_x * weight_x + noise_y * weight_y + noise_z * weight_z) / weight_sum
            
            # Add turbulence/distortion for more natural patterns
            if octave > 0:
                turb_freq = 1.7 + random_tensor[1, 0] * 0.6  # Randomize turbulence frequency
                distortion_x = torch.sin(2 * np.pi * phase_x * turb_freq + phase_offset_x)
                distortion_y = torch.sin(2 * np.pi * phase_y * turb_freq + phase_offset_y)
                distortion_z = torch.sin(2 * np.pi * phase_z * turb_freq + phase_offset_z)
                
                # Make distortion smoother with higher octaves to reduce small isolated areas
                distortion_factor = 0.3 * (1.0 / (octave + 1))
                noise = noise + (distortion_x * distortion_y * distortion_z) * distortion_factor
            
            # Add weighted noise to the total
            result += noise * amplitude
        
        # Normalize to [0, 1] range
        result = result / max_amplitude
        result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        
        # Apply gaussian smoothing to reduce small isolated regions
        # This is a key step to reduce the number of small disconnected lesions
        # We'll use 3D gaussian blur
        kernel_size = 5
        sigma = 1.0
        
        # Create a 3D Gaussian kernel
        kernel_size = int(kernel_size)
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
            
        # Create a 1D Gaussian kernel
        x = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * (x / sigma)**2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Expand to 3D
        kernel_x = kernel_1d.view(1, 1, kernel_size, 1, 1).expand(1, 1, kernel_size, 1, 1)
        kernel_y = kernel_1d.view(1, 1, 1, kernel_size, 1).expand(1, 1, 1, kernel_size, 1)
        kernel_z = kernel_1d.view(1, 1, 1, 1, kernel_size).expand(1, 1, 1, 1, kernel_size)
        
        # Apply separable convolution for efficiency
        result_temp = result.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims [1,1,D,H,W]
        result_temp = F.conv3d(result_temp, kernel_x, padding=(kernel_size//2, 0, 0))
        result_temp = F.conv3d(result_temp, kernel_y, padding=(0, kernel_size//2, 0))
        result_temp = F.conv3d(result_temp, kernel_z, padding=(0, 0, kernel_size//2))
        
        # Remove batch and channel dimensions
        result = result_temp.squeeze(0).squeeze(0)
        
        # End timing and print the time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Noise generation completed in {elapsed_time:.4f} seconds")
        
        return result
    
    def generate_batch_noise(self, batch_size, shape, noise_dim=16, scale=0.1, device='cuda'):
        """
        Generate a batch of 3D Perlin noise vectors
        
        Args:
            batch_size (int): Batch size
            shape (tuple): Shape of each noise volume (D, H, W)
            noise_dim (int): Noise dimension for vector representation
            scale (float): Scale of the noise
            device (str): Device to place the tensor on
            
        Returns:
            torch.Tensor: Batch of noise vectors [batch_size, noise_dim]
        """
        # Create batch by sampling different regions or adding small variations
        batch_noise = []
        for i in range(batch_size):
            # Create a variant by adding a small random offset to the seed
            current_seed = self.seed + i if self.seed is not None else np.random.randint(0, 10000)
            
            # Set the seed for this batch item
            torch.manual_seed(current_seed)
            
            # Generate noise directly on GPU for this batch item
            self.seed = current_seed
            noise_sample = self.generate_3d_perlin(shape, scale, device)
            
            # Flatten and reduce to noise_dim dimensions (simple dimensionality reduction)
            # We'll use average pooling to reduce dimensions
            pool_size = int(np.cbrt(np.prod(shape) / noise_dim))
            if pool_size < 1:
                pool_size = 1
            
            # Reshape for 3D average pooling
            noise_for_pooling = noise_sample.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            
            # Apply 3D average pooling
            pooled = F.avg_pool3d(noise_for_pooling, pool_size)
            
            # Extract noise_dim values from the pooled tensor
            flat_pooled = pooled.view(-1)
            if len(flat_pooled) >= noise_dim:
                noise_vector = flat_pooled[:noise_dim]
            else:
                # If we couldn't extract enough values, repeat and truncate
                repeats = int(np.ceil(noise_dim / len(flat_pooled)))
                noise_vector = flat_pooled.repeat(repeats)[:noise_dim]
            
            batch_noise.append(noise_vector)
        
        # Stack into a batch tensor
        batch_noise_tensor = torch.stack(batch_noise, dim=0)
        
        return batch_noise_tensor


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
    def __init__(self, in_channels=1, out_channels=1, feature_size=24, dropout_rate=0.0, use_noise=True, noise_dim=16,
                 perlin_octaves=4, perlin_persistence=0.5, perlin_lacunarity=2.0, perlin_scale=0.1):
        """
        Generator based on SwinUNETR for lesion synthesis
        
        Args:
            in_channels (int): Number of input channels (1 for lesion atlas)
            out_channels (int): Number of output channels (1 for binary lesion mask)
            feature_size (int): Feature size for SwinUNETR
            dropout_rate (float): Dropout rate
            use_noise (bool): Whether to use noise injection for diversity
            noise_dim (int): Dimension of the noise vector to inject
            perlin_octaves (int): Number of octaves for Perlin noise
            perlin_persistence (float): Persistence parameter for Perlin noise
            perlin_lacunarity (float): Lacunarity parameter for Perlin noise
            perlin_scale (float): Scale parameter for Perlin noise
        """
        super(Generator, self).__init__()
        
        self.use_noise = use_noise
        self.noise_dim = noise_dim
        self.feature_size = feature_size  # Uložíme hodnotu i jako atribut
        self.dropout_rate = dropout_rate  # Uložíme hodnotu i jako atribut
        
        # Initialize Perlin noise generator if using noise
        if use_noise:
            self.perlin_generator = PerlinNoiseGenerator(
                octaves=perlin_octaves,
                persistence=perlin_persistence,
                lacunarity=perlin_lacunarity,
                seed=np.random.randint(0, 10000)
            )
            self.perlin_scale = perlin_scale
        
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
            noise: Optional noise tensor. If None and use_noise is True, Perlin noise will be generated
        
        Returns:
            Generated lesion mask
        """
        # Process noise if using it
        if self.use_noise:
            batch_size, _, D, H, W = x.shape
            
            # Generate Perlin noise if not provided
            if noise is None:
                # Generate Perlin noise vectors for the batch
                noise = self.perlin_generator.generate_batch_noise(
                    batch_size=batch_size,
                    shape=(D, H, W),
                    noise_dim=self.noise_dim,
                    scale=self.perlin_scale,
                    device=x.device
                )
                
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
                 fragmentation_kernel_size=5,
                 perlin_octaves=4,
                 perlin_persistence=0.5,
                 perlin_lacunarity=2.0,
                 perlin_scale=0.1):
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
            perlin_octaves (int): Number of octaves for Perlin noise
            perlin_persistence (float): Persistence parameter for Perlin noise
            perlin_lacunarity (float): Lacunarity parameter for Perlin noise
            perlin_scale (float): Scale parameter controlling smoothness of Perlin noise
        """
        super(SwinGAN, self).__init__()
        
        self.generator = Generator(
            in_channels=in_channels, 
            out_channels=out_channels, 
            feature_size=feature_size, 
            dropout_rate=dropout_rate,
            use_noise=use_noise,
            noise_dim=noise_dim,
            perlin_octaves=perlin_octaves,
            perlin_persistence=perlin_persistence,
            perlin_lacunarity=perlin_lacunarity,
            perlin_scale=perlin_scale
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
        
        # Store Perlin noise parameters for checkpoints
        self.perlin_octaves = perlin_octaves
        self.perlin_persistence = perlin_persistence
        self.perlin_lacunarity = perlin_lacunarity
        self.perlin_scale = perlin_scale
    
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
                    'dropout_rate': self.model.generator.dropout_rate,
                    'perlin_octaves': self.model.perlin_octaves,
                    'perlin_persistence': self.model.perlin_persistence,
                    'perlin_lacunarity': self.model.perlin_lacunarity,
                    'perlin_scale': self.model.perlin_scale
                }, generator_path)
                print(f"Saved generator-only checkpoint to {generator_path}")
                print(f"Checkpoint includes configuration for use_noise={self.model.use_noise}, noise_dim={self.model.noise_dim}, perlin params: octaves={self.model.perlin_octaves}, persistence={self.model.perlin_persistence}, scale={self.model.perlin_scale}")
            
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
    print(f"Perlin noise octaves: {args.perlin_octaves}")
    print(f"Perlin noise persistence: {args.perlin_persistence}")
    print(f"Perlin noise lacunarity: {args.perlin_lacunarity}")
    print(f"Perlin noise scale: {args.perlin_scale}")
    
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
        fragmentation_kernel_size=args.fragmentation_kernel_size,
        perlin_octaves=args.perlin_octaves,
        perlin_persistence=args.perlin_persistence,
        perlin_lacunarity=args.perlin_lacunarity,
        perlin_scale=args.perlin_scale
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


def compute_lesion_coverage(binary_lesion, atlas_mask=None):
    """
    Compute the percentage of voxels covered by lesions

    Args:
        binary_lesion (numpy.ndarray): Binary lesion mask
        atlas_mask (numpy.ndarray, optional): Atlas mask to define the region of interest
    
    Returns:
        float: Percentage of coverage (0.0-100.0)
    """
    if atlas_mask is not None:
        # Count only voxels within the atlas mask
        total_voxels = np.count_nonzero(atlas_mask)
        lesion_voxels = np.count_nonzero(binary_lesion * atlas_mask)
    else:
        # Count all voxels in the volume
        total_voxels = binary_lesion.size
        lesion_voxels = np.count_nonzero(binary_lesion)
    
    # Calculate percentage
    if total_voxels > 0:
        coverage_percentage = (lesion_voxels / total_voxels) * 100.0
    else:
        coverage_percentage = 0.0
    
    return coverage_percentage


def compute_target_coverage_from_training(lesion_dir, sample_count=10, return_list=False, random_seed=None):
    """
    Compute the average lesion coverage percentage from training data or return list of individual coverages

    Args:
        lesion_dir (str): Directory containing lesion files
        sample_count (int): Number of samples to consider
        return_list (bool): If True, returns a list of individual coverage values instead of average
        random_seed (int): Random seed for reproducibility

    Returns:
        float or list: Average coverage percentage or list of individual coverage values
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # List all lesion files
    lesion_files = sorted(glob.glob(os.path.join(lesion_dir, "*lesion.nii*")))
    
    # Filter out empty lesion files
    non_empty_files = []
    for lesion_file in lesion_files:
        lesion = nib.load(lesion_file).get_fdata()
        if np.count_nonzero(lesion) > 0:
            non_empty_files.append(lesion_file)
    
    print(f"Found {len(non_empty_files)} non-empty lesion files in training data")
    
    # If no non-empty files found, return a default value
    if not non_empty_files:
        print("Warning: No non-empty lesion files found, using default coverage of 1%")
        return [1.0] if return_list else 1.0
    
    # Select random samples or use all if fewer than requested
    if len(non_empty_files) > sample_count:
        selected_files = np.random.choice(non_empty_files, size=sample_count, replace=False)
    else:
        selected_files = non_empty_files
    
    # Compute coverage for each file
    coverage_values = []
    for lesion_file in selected_files:
        lesion = nib.load(lesion_file).get_fdata()
        coverage = compute_lesion_coverage(lesion > 0)
        coverage_values.append(coverage)
        print(f"File {os.path.basename(lesion_file)}: {coverage:.4f}% coverage")
    
    # Return list of individual values or average
    if return_list:
        return coverage_values
    else:
        # Calculate average coverage
        avg_coverage = np.mean(coverage_values)
        print(f"Average coverage from training data: {avg_coverage:.4f}%")
        return avg_coverage


def get_single_lesion_coverage(lesion_file):
    """
    Compute coverage percentage for a single lesion file

    Args:
        lesion_file (str): Path to the lesion file

    Returns:
        float: Coverage percentage
    """
    lesion = nib.load(lesion_file).get_fdata()
    coverage = compute_lesion_coverage(lesion > 0)
    return coverage


def find_adaptive_threshold(probability_map, target_coverage, atlas_mask=None, 
                            initial_threshold=0.5, min_threshold=0.00001, max_threshold=0.999, 
                            max_iterations=50, tolerance=0.1):
    """
    Find the threshold value that produces lesions with coverage close to the target
    
    Args:
        probability_map (numpy.ndarray): Probability map from the generator
        target_coverage (float): Target coverage percentage to achieve
        atlas_mask (numpy.ndarray, optional): Atlas mask to define the region of interest
        initial_threshold (float): Starting threshold value
        min_threshold (float): Minimum threshold value
        max_threshold (float): Maximum threshold value
        max_iterations (int): Maximum number of iterations
        tolerance (float): Acceptable difference between actual and target coverage
        
    Returns:
        tuple: (found_threshold, actual_coverage, binary_lesion)
    """
    # First, check if we need a very low threshold by testing the extremes
    binary_high = (probability_map > max_threshold).astype(np.float32)
    if atlas_mask is not None:
        binary_high = binary_high * atlas_mask
    coverage_high = compute_lesion_coverage(binary_high, atlas_mask)
    
    binary_low = (probability_map > min_threshold).astype(np.float32)
    if atlas_mask is not None:
        binary_low = binary_low * atlas_mask
    coverage_low = compute_lesion_coverage(binary_low, atlas_mask)
    
    print(f"Coverage range: [{coverage_low:.4f}% at threshold {min_threshold}] - [{coverage_high:.4f}% at threshold {max_threshold}]")
    
    # Verify that the target is within the achievable range
    if target_coverage > coverage_low:
        print(f"Warning: Target coverage {target_coverage:.4f}% is higher than maximum possible coverage {coverage_low:.4f}%")
        print(f"Using minimum threshold {min_threshold} to achieve maximum coverage")
        return min_threshold, coverage_low, binary_low
    
    if target_coverage < coverage_high:
        print(f"Warning: Target coverage {target_coverage:.4f}% is lower than minimum possible coverage {coverage_high:.4f}%")
        print(f"Using maximum threshold {max_threshold} to achieve minimum coverage")
        return max_threshold, coverage_high, binary_high
    
    # Initialize the binary search with a better initial guess
    if coverage_low != coverage_high:
        # Interpolate to get a better initial threshold based on linear mapping
        # This helps to start closer to the target
        ratio = (target_coverage - coverage_high) / (coverage_low - coverage_high)
        threshold = min_threshold + ratio * (max_threshold - min_threshold)
        threshold = max(min_threshold, min(max_threshold, threshold))  # Clamp
    else:
        threshold = initial_threshold
    
    print(f"Starting search with initial threshold {threshold:.6f}")
    
    threshold_min = min_threshold
    threshold_max = max_threshold
    
    best_threshold = threshold
    best_diff = float('inf')
    best_binary = None
    
    for iteration in range(max_iterations):
        # Binarize the probability map with current threshold
        binary_lesion = (probability_map > threshold).astype(np.float32)
        
        # Apply atlas mask if provided
        if atlas_mask is not None:
            binary_lesion = binary_lesion * atlas_mask
        
        # Compute current coverage
        current_coverage = compute_lesion_coverage(binary_lesion, atlas_mask)
        
        # Track the best result so far
        diff = abs(current_coverage - target_coverage)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
            best_binary = binary_lesion.copy()
        
        # Check if we're close enough to the target
        if diff <= tolerance:
            print(f"Found threshold {threshold:.6f} with coverage {current_coverage:.4f}% (target: {target_coverage:.4f}%)")
            return threshold, current_coverage, binary_lesion
        
        # Adjust threshold using binary search
        if current_coverage > target_coverage:
            # Too much coverage, increase threshold
            threshold_min = threshold
            threshold = (threshold + threshold_max) / 2
        else:
            # Too little coverage, decrease threshold
            threshold_max = threshold
            threshold = (threshold + threshold_min) / 2
        
        print(f"Iteration {iteration+1}: threshold={threshold:.6f}, coverage={current_coverage:.4f}%, target={target_coverage:.4f}%")
    
    # If we exit the loop, use the best threshold found
    binary_lesion = (probability_map > best_threshold).astype(np.float32)
    if atlas_mask is not None:
        binary_lesion = binary_lesion * atlas_mask
    current_coverage = compute_lesion_coverage(binary_lesion, atlas_mask)
    
    print(f"Warning: Reached maximum iterations. Using best threshold {best_threshold:.6f} with coverage {current_coverage:.4f}% (target: {target_coverage:.4f}%)")
    return best_threshold, current_coverage, binary_lesion


def generate_lesions(
    model_checkpoint,
    lesion_atlas,
    output_file=None,
    output_dir=None,
    threshold=0.5,
    device='cuda',
    num_samples=1,
    perlin_octaves=4,
    perlin_persistence=0.5,
    perlin_lacunarity=2.0,
    perlin_scale=0.1,
    min_lesion_size=10,     # Minimální velikost léze (v počtu voxelů)
    smooth_sigma=0.5,       # Hodnota sigma pro vyhlazení lézí
    morph_close_size=2,     # Velikost strukturního elementu pro morfologickou operaci uzavření
    use_adaptive_threshold=False,  # Použít adaptivní threshold
    training_lesion_dir=None,      # Adresář s trénovacími lézemi
    target_coverage=None,          # Cílové pokrytí lézemi v procentech
    min_adaptive_threshold=0.00001,  # Minimální hodnota pro adaptivní threshold
    max_adaptive_threshold=0.999,    # Maximální hodnota pro adaptivní threshold
    adaptive_threshold_iterations=50,  # Maximální počet iterací pro hledání adaptivního thresholdu
    use_different_target_for_each_sample=False  # Použít jiný cílový coverage pro každý vzorek
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
        perlin_octaves (int): Number of octaves for Perlin noise
        perlin_persistence (float): Persistence parameter for Perlin noise
        perlin_lacunarity (float): Lacunarity parameter for Perlin noise
        perlin_scale (float): Scale parameter for Perlin noise
        min_lesion_size (int): Minimální velikost léze v počtu voxelů (menší léze budou odstraněny)
        smooth_sigma (float): Hodnota sigma pro vyhlazení lézí pomocí Gaussovského filtru
        morph_close_size (int): Velikost strukturního elementu pro morfologickou operaci uzavření
        use_adaptive_threshold (bool): Použít adaptivní threshold pro dosažení podobného pokrytí jako v trénovacích datech
        training_lesion_dir (str): Adresář s trénovacími lézemi, používá se pro výpočet cílového pokrytí
        target_coverage (float): Cílové pokrytí lézemi v procentech, pokud není zadáno a use_adaptive_threshold=True, 
                                 použije se průměrné pokrytí z training_lesion_dir
        min_adaptive_threshold (float): Minimální hodnota thresholdu pro adaptivní prahování
        max_adaptive_threshold (float): Maximální hodnota thresholdu pro adaptivní prahování
        adaptive_threshold_iterations (int): Maximální počet iterací pro hledání adaptivního thresholdu
        use_different_target_for_each_sample (bool): Použít jiný cílový coverage pro každý vzorek
    """
    # Validate input parameters
    if num_samples > 1 and output_dir is None:
        raise ValueError("output_dir must be specified when generating multiple samples")
    if num_samples == 1 and output_file is None and output_dir is None:
        raise ValueError("Either output_file or output_dir must be specified")
    if use_adaptive_threshold and training_lesion_dir is None and target_coverage is None:
        raise ValueError("For adaptive threshold, either training_lesion_dir or target_coverage must be specified")
    if use_different_target_for_each_sample and training_lesion_dir is None:
        raise ValueError("For using different target for each sample, training_lesion_dir must be specified")
    
    print(f"Generating lesions with the following parameters:")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Lesion atlas: {lesion_atlas}")
    if output_file:
        print(f"Output file: {output_file}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print(f"Threshold: {threshold if not use_adaptive_threshold else 'adaptive'}")
    print(f"Device: {device}")
    print(f"Number of samples: {num_samples}")
    print(f"Perlin noise octaves: {perlin_octaves}")
    print(f"Perlin noise persistence: {perlin_persistence}")
    print(f"Perlin noise lacunarity: {perlin_lacunarity}")
    print(f"Perlin noise scale: {perlin_scale}")
    print(f"Minimum lesion size: {min_lesion_size}")
    print(f"Smoothing sigma: {smooth_sigma}")
    print(f"Morphological closing size: {morph_close_size}")
    if use_adaptive_threshold:
        print(f"Using adaptive threshold to match training data coverage")
        if training_lesion_dir:
            print(f"Training lesion directory: {training_lesion_dir}")
        if target_coverage is not None:
            print(f"Target coverage: {target_coverage}%")
        print(f"Adaptive threshold range: [{min_adaptive_threshold}, {max_adaptive_threshold}]")
        print(f"Adaptive threshold max iterations: {adaptive_threshold_iterations}")
        if use_different_target_for_each_sample:
            print(f"Using different target coverage for each sample")
    
    # Load the lesion atlas
    atlas_img = nib.load(lesion_atlas)
    atlas_data = atlas_img.get_fdata()
    
    # Store original shape for later
    orig_shape = atlas_data.shape
    
    # Normalize the atlas
    atlas_data = (atlas_data - atlas_data.min()) / (atlas_data.max() - atlas_data.min() + 1e-8)
    
    # Create atlas mask for region of interest
    atlas_mask = atlas_data > 0
    
    # If using adaptive threshold with different targets for each sample, prepare list of target coverages
    target_coverages = None
    if use_adaptive_threshold and use_different_target_for_each_sample and training_lesion_dir is not None:
        # List all lesion files
        lesion_files = sorted(glob.glob(os.path.join(training_lesion_dir, "*lesion.nii*")))
        
        # Filter out empty lesion files
        non_empty_files = []
        for lesion_file in lesion_files:
            lesion = nib.load(lesion_file).get_fdata()
            if np.count_nonzero(lesion) > 0:
                non_empty_files.append(lesion_file)
        
        if not non_empty_files:
            print("Warning: No non-empty lesion files found in training directory")
            if target_coverage is not None:
                # Use specified target_coverage for all samples
                target_coverages = [target_coverage] * num_samples
            else:
                # Use default coverage of 1%
                target_coverages = [1.0] * num_samples
        else:
            # If there are fewer files than samples, allow reusing files
            if len(non_empty_files) < num_samples:
                selected_files = np.random.choice(non_empty_files, size=num_samples, replace=True)
            else:
                selected_files = np.random.choice(non_empty_files, size=num_samples, replace=False)
            
            # Compute coverage for each selected file
            target_coverages = []
            for i, lesion_file in enumerate(selected_files):
                coverage = get_single_lesion_coverage(lesion_file)
                target_coverages.append(coverage)
                print(f"Sample {i+1} will use target coverage from {os.path.basename(lesion_file)}: {coverage:.4f}%")
    # If not using different targets for each sample but using adaptive threshold
    elif use_adaptive_threshold and target_coverage is None and training_lesion_dir is not None:
        # Compute single target coverage from all training data
        target_coverage = compute_target_coverage_from_training(training_lesion_dir)
    
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
        
        # Load Perlin noise parameters from checkpoint if available
        loaded_perlin_octaves = checkpoint.get('perlin_octaves', perlin_octaves)
        loaded_perlin_persistence = checkpoint.get('perlin_persistence', perlin_persistence)
        loaded_perlin_lacunarity = checkpoint.get('perlin_lacunarity', perlin_lacunarity)
        loaded_perlin_scale = checkpoint.get('perlin_scale', perlin_scale)
        
        print(f"Checkpoint configuration: use_noise={use_noise}, noise_dim={noise_dim}, feature_size={feature_size}")
        print(f"Perlin noise parameters: octaves={loaded_perlin_octaves}, persistence={loaded_perlin_persistence}, "
              f"lacunarity={loaded_perlin_lacunarity}, scale={loaded_perlin_scale}")
        
        # Vytvoření nového modelu s načtenými parametry
        model = SwinGAN(
            in_channels=1, 
            out_channels=1,
            feature_size=feature_size,
            dropout_rate=dropout_rate,
            use_noise=use_noise,
            noise_dim=noise_dim,
            perlin_octaves=loaded_perlin_octaves,
            perlin_persistence=loaded_perlin_persistence,
            perlin_lacunarity=loaded_perlin_lacunarity,
            perlin_scale=loaded_perlin_scale
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
    
    # Set master seed for the overall generation process
    master_seed = np.random.randint(0, 1000000)
    print(f"Using master seed: {master_seed} for overall generation process")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Derive a unique seed for this sample using the master seed
            sample_seed = (master_seed + i * 2053) % 1000000  # Use a large prime number (2053) for offset
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            
            print(f"\nGenerating sample {i+1} with seed {sample_seed}")
            
            # Get the target coverage for this specific sample if using different targets
            current_target_coverage = target_coverages[i] if target_coverages else target_coverage
            
            # Create a random noise vector for this sample
            if model.use_noise:
                # Initialize Perlin noise generator using parameters from checkpoint if available
                perlin_octaves_model = getattr(model, 'perlin_octaves', perlin_octaves)
                perlin_persistence_model = getattr(model, 'perlin_persistence', perlin_persistence)
                perlin_lacunarity_model = getattr(model, 'perlin_lacunarity', perlin_lacunarity)
                perlin_scale_model = getattr(model, 'perlin_scale', perlin_scale)
                
                # Add more variation by slightly perturbing the noise parameters for each sample
                # This helps create more diverse samples
                sample_octaves = max(1, perlin_octaves_model + np.random.randint(-1, 2))  # +/- 1
                sample_persistence = max(0.1, min(0.9, perlin_persistence_model + np.random.uniform(-0.1, 0.1)))
                sample_lacunarity = max(1.0, min(3.0, perlin_lacunarity_model + np.random.uniform(-0.2, 0.2)))
                sample_scale = max(0.05, min(0.3, perlin_scale_model + np.random.uniform(-0.03, 0.03)))
                
                print(f"  Sample-specific noise parameters: octaves={sample_octaves}, "
                      f"persistence={sample_persistence:.3f}, lacunarity={sample_lacunarity:.3f}, scale={sample_scale:.3f}")
                
                # Create a new seed for Perlin that's different for each sample
                # Use a more complex seed derivation with better randomization
                perlin_seed = (sample_seed ^ 0x55AA55AA) + (i * 7919)  # XOR with a constant and add prime offset
                
                # Further mix the seed with a time component to ensure uniqueness even with same master seed
                time_component = int(time.time() * 1000) % 10000
                perlin_seed = (perlin_seed + time_component) % 1000000
                
                # Vytvoření zcela nového generátoru Perlin šumu pro každý vzorek
                perlin_gen = PerlinNoiseGenerator(
                    octaves=sample_octaves,
                    persistence=sample_persistence,
                    lacunarity=sample_lacunarity,
                    seed=perlin_seed
                )
                
                # Generování základního šumu
                noise = perlin_gen.generate_batch_noise(
                    batch_size=1, 
                    shape=(atlas_data.shape[0], atlas_data.shape[1], atlas_data.shape[2]),
                    noise_dim=model.noise_dim,
                    scale=sample_scale,
                    device=device
                )
                
                # AGRESIVNÍ TRANSFORMACE ŠUMU - aplikujeme vždy
                # Vytvoříme vícenásobné vrstvy šumu a zkombinujeme je pro větší komplexitu
                
                # Generování několika různých šumových komponent
                noise_layers = []
                num_layers = np.random.randint(2, 5)  # 2-4 vrstvy šumu
                
                for layer in range(num_layers):
                    # Různý seed pro každou vrstvu
                    layer_seed = np.random.randint(0, 1000000)
                    layer_octaves = np.random.randint(2, 6)
                    layer_persistence = np.random.uniform(0.3, 0.7)
                    layer_lacunarity = np.random.uniform(1.5, 2.5)
                    layer_scale = np.random.uniform(0.05, 0.15)
                    
                    # Vytvoření nového generátoru pro tuto vrstvu
                    layer_gen = PerlinNoiseGenerator(
                        octaves=layer_octaves,
                        persistence=layer_persistence,
                        lacunarity=layer_lacunarity,
                        seed=layer_seed
                    )
                    
                    # Generování šumu pro tuto vrstvu
                    layer_noise = layer_gen.generate_batch_noise(
                        batch_size=1,
                        shape=(atlas_data.shape[0], atlas_data.shape[1], atlas_data.shape[2]),
                        noise_dim=model.noise_dim,
                        scale=layer_scale,
                        device=device
                    )
                    
                    noise_layers.append(layer_noise)
                
                # Kombinace všech vrstev šumu s různou váhou
                combined_noise = noise.clone()  # Začínáme základním šumem
                for layer_idx, layer_noise in enumerate(noise_layers):
                    weight = np.random.uniform(0.2, 0.8)
                    # Různé způsoby kombinace šumu pro různé typy interakcí
                    if layer_idx % 3 == 0:
                        # Aditivní kombinace
                        combined_noise = combined_noise * (1 - weight) + layer_noise * weight
                    elif layer_idx % 3 == 1:
                        # Multiplikativní kombinace
                        combined_noise = combined_noise * (layer_noise * weight + (1 - weight))
                    else:
                        # Max/min kombinace
                        if np.random.random() > 0.5:
                            combined_noise = torch.max(combined_noise * (1 - weight), layer_noise * weight)
                        else:
                            combined_noise = torch.min(combined_noise * (1 + weight), layer_noise * (1 + weight))
                
                noise = combined_noise
                print(f"  Applied {num_layers} noise layers with complex blending for extreme variation")
            else:
                noise = None
            
            # Aplikujeme náhodnou transformaci vstupního atlasu pro každý vzorek
            # To způsobí, že model bude generovat léze v mírně odlišných lokalitách
            if np.random.random() > 0.3:  # Pro 70% vzorků
                # Vyrobíme kopii atlasu pro manipulaci
                transformed_atlas = atlas_tensor.clone()
                
                # Náhodné posunutí atlasu o několik voxelů
                shift_x = np.random.randint(-5, 6)  # Posunutí o -5 až +5 voxelů
                shift_y = np.random.randint(-5, 6)
                shift_z = np.random.randint(-3, 4)  # Méně v ose Z
                
                if shift_x != 0 or shift_y != 0 or shift_z != 0:
                    # Použití torch.roll pro cyklické posunutí atlasu
                    transformed_atlas = torch.roll(transformed_atlas, shifts=(shift_z, shift_x, shift_y), dims=(2, 3, 4))
                    print(f"  Applied spatial shift to atlas: ({shift_x}, {shift_y}, {shift_z})")
                
                # Náhodná lokální úprava intenzity atlasu
                if np.random.random() > 0.5:
                    # Vytvoříme náhodnou masku pro lokální zesílení/zeslabení atlasu
                    intensity_factor = np.random.uniform(0.8, 1.2)
                    mask_size = np.random.randint(10, 30)
                    
                    # Vytvoříme náhodnou lokální masku
                    mask_center_x = np.random.randint(mask_size, transformed_atlas.shape[3] - mask_size)
                    mask_center_y = np.random.randint(mask_size, transformed_atlas.shape[4] - mask_size)
                    mask_center_z = np.random.randint(mask_size//2, transformed_atlas.shape[2] - mask_size//2)
                    
                    # Aplikujeme změnu intenzity v dané oblasti
                    transformed_atlas[0, 0, 
                                      max(0, mask_center_z - mask_size//2):min(transformed_atlas.shape[2], mask_center_z + mask_size//2),
                                      max(0, mask_center_x - mask_size):min(transformed_atlas.shape[3], mask_center_x + mask_size),
                                      max(0, mask_center_y - mask_size):min(transformed_atlas.shape[4], mask_center_y + mask_size)] *= intensity_factor
                    
                    print(f"  Applied local intensity modification with factor {intensity_factor:.2f}")
                
                # Použití transformovaného atlasu pro generování
                atlas_input = transformed_atlas
            else:
                atlas_input = atlas_tensor
            
            # Generate the lesion
            fake_lesion_logits = model(atlas_input, noise)
            
            # Apply sigmoid to get probability map
            fake_lesion = torch.sigmoid(fake_lesion_logits)
            
            # Convert to numpy array
            fake_lesion_np = fake_lesion.squeeze().cpu().numpy()
            
            # AGRESIVNÍ MANIPULACE S PRAVDĚPODOBNOSTNÍ MAPOU
            # Aplikujeme různé transformace na vygenerovanou pravděpodobnostní mapu
            
            # 1. Aplikace náhodné prostorové perturbace - vždy, ne jen pro polovinu vzorků
            # Vytvoříme výraznější perturbace pro větší diverzitu
            perturbation_strength = np.random.uniform(0.02, 0.1)  # Silnější perturbace (0.02-0.1 místo pevné 0.02)
            perturbation = np.random.normal(0, perturbation_strength, fake_lesion_np.shape)
            fake_lesion_np = fake_lesion_np + perturbation
            
            # 2. Aplikace nelineárních transformací pro změnu charakteru mapy
            transformation_type = np.random.randint(0, 5)  # 5 různých transformací
            
            if transformation_type == 0:
                # Exponenciální transformace - zvýrazní vyšší hodnoty
                power = np.random.uniform(0.5, 2.0)
                fake_lesion_np = np.power(fake_lesion_np, power)
                print(f"  Applied exponential transformation with power {power:.2f}")
            elif transformation_type == 1:
                # Sigmoidní transformace - zvýrazní kontrast mezi nízkou a vysokou pravděpodobností
                alpha = np.random.uniform(5, 15)
                beta = np.random.uniform(0.3, 0.7)
                fake_lesion_np = 1 / (1 + np.exp(-alpha * (fake_lesion_np - beta)))
                print(f"  Applied sigmoid transformation with alpha={alpha:.2f}, beta={beta:.2f}")
            elif transformation_type == 2:
                # Logaritmická transformace - zvýrazní nižší hodnoty
                epsilon = 1e-5  # Malá hodnota pro zabránění log(0)
                fake_lesion_np = np.log(fake_lesion_np + epsilon) / np.log(1 + epsilon)
                print(f"  Applied logarithmic transformation")
            elif transformation_type == 3:
                # Inverzní transformace v určitých regionech - lokální inverze hodnot
                if np.random.random() > 0.5:
                    inv_mask = np.random.uniform(0, 1, fake_lesion_np.shape) > 0.8  # 20% voxelů
                    fake_lesion_np[inv_mask] = 1 - fake_lesion_np[inv_mask]
                    print(f"  Applied local intensity inversion")
            elif transformation_type == 4:
                # Prahování s více úrovněmi - vytvoří více různých stupňů
                thresholds = np.sort(np.random.uniform(0.2, 0.8, np.random.randint(2, 5)))
                levels = np.linspace(0.2, 1.0, len(thresholds) + 1)
                
                result = np.zeros_like(fake_lesion_np)
                mask = fake_lesion_np > thresholds[0]
                result[mask] = levels[0]
                
                for i in range(1, len(thresholds)):
                    mask = fake_lesion_np > thresholds[i]
                    result[mask] = levels[i]
                
                mask = fake_lesion_np > thresholds[-1]
                result[mask] = levels[-1]
                
                fake_lesion_np = result
                print(f"  Applied multi-level thresholding with {len(thresholds)} levels")
            
            # Zajistíme, že hodnoty zůstanou v platném rozsahu [0, 1]
            fake_lesion_np = np.clip(fake_lesion_np, 0.0, 1.0)
            
            # Apply Gaussian smoothing to reduce noise and small fragments
            if smooth_sigma > 0:
                # Náhodný výběr typu vyhlazení
                smoothing_type = np.random.randint(0, 3)
                
                if smoothing_type == 0:
                    # Standardní Gaussovské vyhlazení s náhodnou sigmou
                    sample_smooth_sigma = np.random.uniform(0.2, 1.0)  # Širší rozsah (0.2-1.0)
                    fake_lesion_np = gaussian_filter(fake_lesion_np, sigma=sample_smooth_sigma)
                    print(f"  Using Gaussian smoothing with sigma: {sample_smooth_sigma:.3f}")
                elif smoothing_type == 1:
                    # Anisotropické vyhlazení - různé sigmy pro různé směry
                    sigma_x = np.random.uniform(0.3, 1.2)
                    sigma_y = np.random.uniform(0.3, 1.2)
                    sigma_z = np.random.uniform(0.3, 1.2)
                    fake_lesion_np = gaussian_filter(fake_lesion_np, sigma=(sigma_z, sigma_x, sigma_y))
                    print(f"  Using anisotropic smoothing with sigma: ({sigma_x:.2f}, {sigma_y:.2f}, {sigma_z:.2f})")
                elif smoothing_type == 2:
                    # Selektivní vyhlazení - vyhlazení pouze určitých oblastí
                    baseline_sigma = np.random.uniform(0.3, 0.8)
                    # Vytvoříme masku pro selektivní vyhlazení
                    smooth_threshold = np.random.uniform(0.3, 0.7)
                    areas_to_smooth = fake_lesion_np > smooth_threshold
                    
                    # Aplikujeme vyhlazení pouze na vybrané oblasti
                    temp_result = fake_lesion_np.copy()
                    temp_smooth = gaussian_filter(fake_lesion_np, sigma=baseline_sigma)
                    temp_result[areas_to_smooth] = temp_smooth[areas_to_smooth]
                    fake_lesion_np = temp_result
                    print(f"  Using selective smoothing with threshold {smooth_threshold:.2f} and sigma {baseline_sigma:.2f}")
            
            # Apply adaptive threshold if requested
            if use_adaptive_threshold and current_target_coverage is not None:
                print(f"Finding adaptive threshold for sample {i+1} to match coverage of {current_target_coverage:.4f}%")
                found_threshold, actual_coverage, binary_lesion = find_adaptive_threshold(
                    fake_lesion_np, 
                    current_target_coverage, 
                    atlas_mask,
                    initial_threshold=threshold,
                    min_threshold=min_adaptive_threshold,
                    max_threshold=max_adaptive_threshold,
                    max_iterations=adaptive_threshold_iterations
                )
                print(f"Sample {i+1}: Using threshold {found_threshold:.6f} with coverage {actual_coverage:.4f}%")
            else:
                # Use fixed threshold
                binary_lesion = (fake_lesion_np > threshold).astype(np.float32)
                # Ensure lesions only appear in regions with non-zero atlas values
                binary_lesion = binary_lesion * atlas_mask
            
            # Apply morphological operations to remove small isolated regions and fill holes
            if morph_close_size > 0:
                # Vary morphological operation parameters for each sample
                sample_morph_size = morph_close_size + np.random.randint(-1, 2)  # -1, 0, or +1
                sample_morph_size = max(1, sample_morph_size)  # Ensure at least 1
                
                struct = generate_binary_structure(3, 1)  # 6-connectivity
                for _ in range(sample_morph_size - 1):
                    struct = binary_dilation(struct)
                
                print(f"  Using sample-specific morphological closing size: {sample_morph_size}")
                
                # Close holes in the lesions
                binary_lesion = binary_closing(binary_lesion, structure=struct)
            
            # Remove small isolated lesions
            if min_lesion_size > 0:
                # Vary minimum lesion size slightly for each sample
                sample_min_size = int(min_lesion_size * (0.9 + np.random.uniform(0, 0.2)))  # 0.9-1.1x the original
                sample_min_size = max(1, sample_min_size)  # Ensure at least 1
                
                print(f"  Using sample-specific minimum lesion size: {sample_min_size}")
                
                labeled_array, num_features = measure.label(binary_lesion)
                component_sizes = np.bincount(labeled_array.ravel())
                # Set background (index 0) size to 0
                if len(component_sizes) > 0:
                    component_sizes[0] = 0
                # Filter by size
                too_small = component_sizes < sample_min_size
                too_small_mask = too_small[labeled_array]
                binary_lesion[too_small_mask] = 0
            
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
            # Ensure data is of supported type (not bool)
            binary_lesion_for_saving = binary_lesion.astype(np.int16)  # Convert to int16, which is supported by NIfTI
            lesion_img = nib.Nifti1Image(binary_lesion_for_saving, atlas_img.affine)
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
                print(f"Sample {i+1}: Total lesion volume: {lesion_volume_percentage:.4f}% of brain volume ({lesion_volume_ml:.2f} ml)")
                
                # Varování pokud počet lézí je mimo očekávaný rozsah podle trénovací množiny
                if num_lesions > 75:
                    print(f"WARNING: Sample {i+1} has {num_lesions} lesions, which is much higher than expected (1-75 based on training set)")
            else:
                print(f"Generated {num_lesions} distinct lesions")
                print(f"Total lesion volume: {lesion_volume_percentage:.4f}% of brain volume ({lesion_volume_ml:.2f} ml)")
                
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
        
        # Ensure mean_lesion is of a supported data type
        mean_lesion = mean_lesion.astype(np.float32)  # Use float32 for probability map
        mean_img = nib.Nifti1Image(mean_lesion, atlas_img.affine)
        nib.save(mean_img, mean_output_file)
        print(f"Saved mean probability map to {mean_output_file}")


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='SwinGAN - Synthetic Lesion Generator')
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train a model')
    
    # Training arguments
    train_parser.add_argument('--lesion_dir', type=str, required=True, 
                             help='Directory containing lesion NIfTI files')
    train_parser.add_argument('--lesion_atlas', type=str, required=True, 
                             help='Path to lesion atlas (frequency map)')
    train_parser.add_argument('--output_dir', type=str, required=True, 
                             help='Directory to save model checkpoints and logs')
    train_parser.add_argument('--batch_size', type=int, default=4, 
                             help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=100, 
                             help='Number of epochs to train')
    train_parser.add_argument('--lr_g', type=float, default=1e-4, 
                             help='Learning rate for generator')
    train_parser.add_argument('--lr_d', type=float, default=4e-4, 
                             help='Learning rate for discriminator')
    train_parser.add_argument('--feature_size', type=int, default=24, 
                             help='Feature size for SwinUNETR')
    train_parser.add_argument('--dropout_rate', type=float, default=0.0, 
                             help='Dropout rate for SwinUNETR')
    train_parser.add_argument('--lambda_focal', type=float, default=1.0, 
                             help='Weight for focal loss')
    train_parser.add_argument('--lambda_l1', type=float, default=10.0, 
                             help='Weight for L1 loss')
    train_parser.add_argument('--lambda_fragmentation', type=float, default=1.0, 
                             help='Weight for fragmentation loss')
    train_parser.add_argument('--focal_alpha', type=float, default=0.75, 
                             help='Alpha parameter for focal loss')
    train_parser.add_argument('--focal_gamma', type=float, default=2.0, 
                             help='Gamma parameter for focal loss')
    train_parser.add_argument('--save_interval', type=int, default=10, 
                             help='Interval for saving model checkpoints (epochs)')
    train_parser.add_argument('--generator_save_interval', type=int, default=5, 
                             help='Interval for saving generator checkpoints (epochs)')
    train_parser.add_argument('--val_split', type=float, default=0.1, 
                             help='Fraction of data to use for validation')
    train_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                             help='Device to use (cuda or cpu)')
    train_parser.add_argument('--disable_amp', action='store_true', 
                             help='Disable automatic mixed precision')
    train_parser.add_argument('--min_non_zero', type=float, default=0.00001,
                             help='Minimum percentage of non-zero voxels to include a sample')
    train_parser.add_argument('--fragmentation_kernel_size', type=int, default=3, 
                             help='Kernel size for fragmentation loss')
    train_parser.add_argument('--perlin_octaves', type=int, default=4, 
                             help='Number of octaves for Perlin noise (higher = more detail)')
    train_parser.add_argument('--perlin_persistence', type=float, default=0.5, 
                             help='Persistence parameter for Perlin noise (0-1)')
    train_parser.add_argument('--perlin_lacunarity', type=float, default=2.0, 
                             help='Lacunarity parameter for Perlin noise (how quickly detail increases)')
    train_parser.add_argument('--perlin_scale', type=float, default=0.1, 
                             help='Scale parameter for Perlin noise (smaller = smoother)')
    train_parser.add_argument('--val_lesion_dir', type=str, default=None,
                             help='Directory containing validation lesion NIfTI files')
    train_parser.add_argument('--num_workers', type=int, default=4,
                             help='Number of workers for data loading')
    
    # Generate mode
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic lesions')
    
    # Generation arguments
    gen_parser.add_argument('--model_checkpoint', type=str, required=True, 
                           help='Path to model checkpoint')
    gen_parser.add_argument('--lesion_atlas', type=str, required=True, 
                           help='Path to lesion atlas (frequency map)')
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
    gen_parser.add_argument('--perlin_octaves', type=int, default=4, 
                           help='Number of octaves for Perlin noise (higher = more detail)')
    gen_parser.add_argument('--perlin_persistence', type=float, default=0.5, 
                           help='Persistence parameter for Perlin noise (0-1)')
    gen_parser.add_argument('--perlin_lacunarity', type=float, default=2.0, 
                           help='Lacunarity parameter for Perlin noise (how quickly detail increases)')
    gen_parser.add_argument('--perlin_scale', type=float, default=0.1, 
                           help='Scale parameter for Perlin noise (smaller = smoother)')
    gen_parser.add_argument('--min_lesion_size', type=int, default=10,
                           help='Minimum size of lesions in voxels (smaller will be removed)')
    gen_parser.add_argument('--smooth_sigma', type=float, default=0.5,
                           help='Sigma value for Gaussian smoothing of lesion probability map')
    gen_parser.add_argument('--morph_close_size', type=int, default=2,
                           help='Size of structuring element for morphological closing operation')
    gen_parser.add_argument('--use_adaptive_threshold', action='store_true',
                           help='Use adaptive thresholding to match training data coverage')
    gen_parser.add_argument('--training_lesion_dir', type=str, default=None,
                           help='Directory with training lesions to compute target coverage')
    gen_parser.add_argument('--target_coverage', type=float, default=None,
                           help='Target lesion coverage percentage (if not specified, computed from training data)')
    gen_parser.add_argument('--min_adaptive_threshold', type=float, default=0.00001,
                           help='Minimum threshold value for adaptive thresholding')
    gen_parser.add_argument('--max_adaptive_threshold', type=float, default=0.999,
                           help='Maximum threshold value for adaptive thresholding')
    gen_parser.add_argument('--adaptive_threshold_iterations', type=int, default=50,
                           help='Maximum iterations for adaptive threshold search')
    gen_parser.add_argument('--use_different_target_for_each_sample', action='store_true',
                           help='Use a different target coverage for each sample (requires training_lesion_dir)')
    
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
            num_samples=args.num_samples,
            perlin_octaves=args.perlin_octaves,
            perlin_persistence=args.perlin_persistence,
            perlin_lacunarity=args.perlin_lacunarity,
            perlin_scale=args.perlin_scale,
            min_lesion_size=args.min_lesion_size,
            smooth_sigma=args.smooth_sigma,
            morph_close_size=args.morph_close_size,
            use_adaptive_threshold=args.use_adaptive_threshold,
            training_lesion_dir=args.training_lesion_dir,
            target_coverage=args.target_coverage,
            min_adaptive_threshold=args.min_adaptive_threshold,
            max_adaptive_threshold=args.max_adaptive_threshold,
            adaptive_threshold_iterations=args.adaptive_threshold_iterations,
            use_different_target_for_each_sample=args.use_different_target_for_each_sample
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
