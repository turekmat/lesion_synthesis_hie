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
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, generate_binary_structure, gaussian_filter, binary_erosion
from torchvision import transforms
import time  # Import time for performance measurement
import matplotlib.pyplot as plt


class PerlinNoiseGenerator:
    def __init__(self, octaves=4, persistence=0.5, lacunarity=2.0, repeat=1024, seed=None, learnable=False):
        """
        Generate 3D Perlin noise tailored for lesion generation using GPU acceleration
        
        Args:
            octaves (int): Number of octaves for the noise (higher = more detail)
            persistence (float): How much each octave contributes to the overall shape (0-1)
            lacunarity (float): How much detail is added at each octave
            repeat (int): Period after which the noise pattern repeats
            seed (int): Random seed for reproducibility
            learnable (bool): Whether to use learnable parameters for the noise
        """
        # MODIFIED: Lower default octaves for less detail
        self.octaves = octaves 
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.repeat = repeat
        self.learnable = learnable
        
        # Zajistíme, že seed je v platném rozsahu
        if seed is not None:
            # Omezíme seed na rozsah 0 až 2^32-1
            self.seed = seed % (2**32 - 1)
        else:
            self.seed = np.random.randint(0, 10000)
        
        # Set the random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Inicializace učitelných parametrů pokud learnable=True
        if learnable:
            # Parametry pro každou oktávu - učí se samostatně
            self.learnable_octaves = min(octaves, 8)  # Omezíme počet učitelných oktáv
            
            # Pro každou oktávu vytvoříme učitelné parametry
            # Každá oktáva může mít vlastní persistence a lacunarity
            self.learn_persistence = nn.Parameter(torch.ones(self.learnable_octaves) * persistence)
            self.learn_lacunarity = nn.Parameter(torch.ones(self.learnable_octaves) * lacunarity)
            
            # Learnable weights pro každou oktávu - jak moc každá oktáva přispívá
            self.octave_weights = nn.Parameter(torch.ones(self.learnable_octaves))
            
            # Learnable parametry pro modulaci fázových posunů
            self.phase_shifts = nn.Parameter(torch.rand(3, self.learnable_octaves) * 2 * np.pi)
            
            # Learnable parametry pro frekvence
            self.frequency_mods = nn.Parameter(torch.ones(3, self.learnable_octaves))
            
            # Parametry pro míchání různých typů šumu
            self.noise_type_weights = nn.Parameter(torch.ones(4))  # 4 typy šumu
        
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
        
        # Použití learnable parametrů pokud jsou dostupné
        if self.learnable:
            # Normalizace octave weights pro stabilitu a použití softplus pro zajištění pozitivity
            normalized_weights = F.softplus(self.octave_weights) / F.softplus(self.octave_weights).sum()
            
            # Ensure persistence is between 0 and 1 using sigmoid
            learn_persistence = torch.sigmoid(self.learn_persistence)
            
            # Ensure lacunarity is positive using softplus
            learn_lacunarity = F.softplus(self.learn_lacunarity) + 1.0
            
            # Normalizace vah pro typy šumu
            noise_weights = F.softmax(self.noise_type_weights, dim=0)
            
            # Používáme learnable_octaves místo pevného počtu oktáv
            octaves_to_use = self.learnable_octaves
        else:
            # Define frequency components based on scale
            frequencies = [1.0]
            for i in range(1, self.octaves):
                frequencies.append(frequencies[-1] * self.lacunarity)
            
            # Define amplitude for each octave
            amplitudes = [1.0]
            for i in range(1, self.octaves):
                amplitudes.append(amplitudes[-1] * self.persistence)
            
            # MODIFIED: Reduced amplification factor for less aggressive noise
            amplification_factor = 1.5 + torch.rand(1).item() * 1.0  # Reduced from 2.5-4.0 to 1.5-2.5
            amplitudes = [amp * amplification_factor for amp in amplitudes]
            
            max_amplitude = sum(amplitudes)
            octaves_to_use = self.octaves
            
            # MODIFIED: Simpler noise weights for less variability
            noise_weights = torch.ones(4, device=device) * 0.25  # Equal weights instead of random weights
            
        
        # MODIFIED: Use smaller scale for smoother, less detailed noise
        scale = scale * (0.8 + torch.rand(1).item() * 0.5)  # Reduced from 1.5-2.5x to 0.8-1.3x
        
        # Create meshgrid for coordinates
        z = torch.linspace(0, scale, D, device=device)
        y = torch.linspace(0, scale, H, device=device)
        x = torch.linspace(0, scale, W, device=device)
        
        z_grid, y_grid, x_grid = torch.meshgrid(z, y, x, indexing='ij')
        
        # Generate a random phase offset tensor for this specific noise generation
        # This is crucial for making different seeds produce different patterns
        # Zvýšíme rozměr random_tensor pro více nezávislých fázových posunů
        random_tensor = torch.rand(4, 4, device=device)
        
        # Přidáme různé typy komplexního šumu
        noise_types = ["standard", "turbulent", "ridged", "billow"]
        
        # Zkonstruujeme frekvence a amplitudy podle toho, zda používáme učitelné parametry
        if self.learnable:
            frequencies = [1.0]
            for i in range(1, octaves_to_use):
                frequencies.append(frequencies[-1] * learn_lacunarity[i-1].item())
                
            amplitudes = [1.0]
            for i in range(1, octaves_to_use):
                amplitudes.append(amplitudes[-1] * learn_persistence[i-1].item())
                
            # Aplikace learnable weights na amplitudy
            amplitudes = [amplitudes[i] * normalized_weights[i].item() for i in range(octaves_to_use)]
            max_amplitude = sum(amplitudes)
        
        # Generate noise by adding different frequency components
        for octave in range(octaves_to_use):
            # Calculate frequency and amplitude for this octave
            freq = frequencies[octave]
            amplitude = amplitudes[octave]
            
            # Scale coordinates by frequency
            phase_x = x_grid * freq
            phase_y = y_grid * freq
            phase_z = z_grid * freq
            
            if self.learnable:
                # Použijeme učitelné fázové posuny místo náhodných
                phase_offset_x = self.phase_shifts[0, octave % self.learnable_octaves]
                phase_offset_y = self.phase_shifts[1, octave % self.learnable_octaves]
                phase_offset_z = self.phase_shifts[2, octave % self.learnable_octaves]
                
                # Použijeme učitelné frekvenční modulátory
                freq_multiplier_x = F.softplus(self.frequency_mods[0, octave % self.learnable_octaves]) + 0.5
                freq_multiplier_y = F.softplus(self.frequency_mods[1, octave % self.learnable_octaves]) + 0.5
                freq_multiplier_z = F.softplus(self.frequency_mods[2, octave % self.learnable_octaves]) + 0.5
            else:
                # Use the random tensor to create truly random phase offsets based on the seed
                phase_offset_x = random_tensor[0, 0] * 6.28 + octave * 0.5
                phase_offset_y = random_tensor[0, 1] * 6.28 + octave * 1.2
                phase_offset_z = random_tensor[0, 2] * 6.28 + octave * 0.8
                
                # ZMĚNA: Zvýšíme rozsah náhodnosti frequency multipliers pro více variability
                freq_multiplier_x = 1.0 + random_tensor[1, 0] * 1.2  # Zvýšený rozsah až na 1.2
                freq_multiplier_y = 1.0 + random_tensor[1, 1] * 1.2
                freq_multiplier_z = 1.0 + random_tensor[1, 2] * 1.2
            
            # Generování různých typů šumu pro různé oktávy
            noise_type_idx = octave % len(noise_types)
            noise_type = noise_types[noise_type_idx]
            
            # Standard noise using sine waves with randomized phases
            noise_x = torch.sin(2 * np.pi * phase_x * freq_multiplier_x + phase_offset_x)
            noise_y = torch.sin(2 * np.pi * phase_y * freq_multiplier_y + phase_offset_y)
            noise_z = torch.sin(2 * np.pi * phase_z * freq_multiplier_z + phase_offset_z)
            
            # Alternativní typy šumu pro více variability
            noise_turbulent_x = torch.abs(torch.sin(2 * np.pi * phase_x * freq_multiplier_x + phase_offset_x * 2.0))
            noise_turbulent_y = torch.abs(torch.sin(2 * np.pi * phase_y * freq_multiplier_y + phase_offset_y * 2.0))
            noise_turbulent_z = torch.abs(torch.sin(2 * np.pi * phase_z * freq_multiplier_z + phase_offset_z * 2.0))
            
            noise_ridged_x = 1.0 - torch.abs(torch.sin(2 * np.pi * phase_x * freq_multiplier_x + phase_offset_x))
            noise_ridged_y = 1.0 - torch.abs(torch.sin(2 * np.pi * phase_y * freq_multiplier_y + phase_offset_y))
            noise_ridged_z = 1.0 - torch.abs(torch.sin(2 * np.pi * phase_z * freq_multiplier_z + phase_offset_z))
            
            noise_billow_x = 2.0 * torch.abs(torch.sin(2 * np.pi * phase_x * freq_multiplier_x + phase_offset_x)) - 1.0
            noise_billow_y = 2.0 * torch.abs(torch.sin(2 * np.pi * phase_y * freq_multiplier_y + phase_offset_y)) - 1.0
            noise_billow_z = 2.0 * torch.abs(torch.sin(2 * np.pi * phase_z * freq_multiplier_z + phase_offset_z)) - 1.0
            
            # Combine the noise patterns (with random weights)
            weight_x = 0.3 + random_tensor[2, 0] * 0.4
            weight_y = 0.3 + random_tensor[2, 1] * 0.4
            weight_z = 0.3 + random_tensor[2, 2] * 0.4
            weight_sum = weight_x + weight_y + weight_z
            
            # Kombinace různých typů šumu
            standard_noise = (noise_x * weight_x + noise_y * weight_y + noise_z * weight_z) / weight_sum
            turbulent_noise = (noise_turbulent_x * weight_x + noise_turbulent_y * weight_y + noise_turbulent_z * weight_z) / weight_sum
            ridged_noise = (noise_ridged_x * weight_x + noise_ridged_y * weight_y + noise_ridged_z * weight_z) / weight_sum
            billow_noise = (noise_billow_x * weight_x + noise_billow_y * weight_y + noise_billow_z * weight_z) / weight_sum
            
            # Váhovaná kombinace všech typů šumu (buď learnable nebo fixní váhy)
            combined_noise = (
                standard_noise * noise_weights[0] + 
                turbulent_noise * noise_weights[1] + 
                ridged_noise * noise_weights[2] + 
                billow_noise * noise_weights[3]
            )
            
            # Přidáme náhodné nelinearity pro větší komplexitu
            if octave > 0 and torch.rand(1).item() < 0.5:
                combined_noise = torch.tanh(combined_noise * (1.0 + random_tensor[3, 0] * 0.5))
            
            # Add this octave to the result
            result += combined_noise * amplitude
        
        # Normalize to range [-1, 1] considering the max possible amplitude
        result = result / max_amplitude
        
        # Apply Gaussian smoothing if requested (adaptive based on seed)
        # Upravíme smoothing podle náhodného faktoru pro každý generovaný šum
        smooth_factor = 0.3 + 0.4 * torch.rand(1).item()  # Hodnota mezi 0.3 a 0.7
        kernel_size = 3
        sigma = smooth_factor
        
        # Create 1D Gaussian kernels for separable convolution
        kernel_x = torch.exp(-torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, dtype=torch.float32, device=device) ** 2 / (2 * sigma ** 2))
        kernel_x = kernel_x / kernel_x.sum()
        kernel_x = kernel_x.view(1, 1, kernel_size, 1, 1)
        
        kernel_y = torch.exp(-torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, dtype=torch.float32, device=device) ** 2 / (2 * sigma ** 2))
        kernel_y = kernel_y / kernel_y.sum()
        kernel_y = kernel_y.view(1, 1, 1, kernel_size, 1)
        
        kernel_z = torch.exp(-torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, dtype=torch.float32, device=device) ** 2 / (2 * sigma ** 2))
        kernel_z = kernel_z / kernel_z.sum()
        kernel_z = kernel_z.view(1, 1, 1, 1, kernel_size)
        
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
    def __init__(self, in_channels=1, out_channels=1, feature_size=24, dropout_rate=0.0, use_noise=True, noise_dim=32,
                 perlin_octaves=6, perlin_persistence=0.6, perlin_lacunarity=2.5, perlin_scale=0.2, use_learnable_noise=False):
        """
        Generator using SwinUNETR architecture
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            feature_size (int): Feature size
            dropout_rate (float): Dropout rate
            use_noise (bool): Whether to use noise injection (for diversity)
            noise_dim (int): Dimension of the noise vector to inject
            perlin_octaves (int): Number of octaves for Perlin noise
            perlin_persistence (float): Persistence parameter for Perlin noise
            perlin_lacunarity (float): Lacunarity parameter for Perlin noise
            perlin_scale (float): Scale parameter for Perlin noise
            use_learnable_noise (bool): Whether to use learnable Perlin noise parameters
        """
        super(Generator, self).__init__()
        
        self.use_noise = use_noise
        self.noise_dim = noise_dim
        self.feature_size = feature_size  # Uložíme hodnotu i jako atribut
        self.dropout_rate = dropout_rate  # Uložíme hodnotu i jako atribut
        self.use_learnable_noise = use_learnable_noise
        
        # Initialize Perlin noise generator if using noise
        if use_noise:
            self.perlin_generator = PerlinNoiseGenerator(
                octaves=perlin_octaves,
                persistence=perlin_persistence,
                lacunarity=perlin_lacunarity,
                seed=np.random.randint(0, 10000),
                learnable=use_learnable_noise
            )
            self.perlin_scale = perlin_scale
        
        # Always use 2 input channels for SwinUNETR
        actual_in_channels = 2  # Fixed to 2 channels (atlas + noise or zero channel)
            
        if use_noise:
            # Nahradím původní jednoduchý noise_processor složitějším modelem
            # Původní noise_processor měl jen dvě konvoluce s LeakyReLU
            self.noise_processor = nn.Sequential(
                # První vrstva - expanze kanálů
                nn.Conv3d(1, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm3d(16),
                
                # Druhá vrstva - udržení kanálů, zvýšení receptivního pole
                nn.Conv3d(16, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm3d(16),
                
                # Residuální blok 1
                nn.Conv3d(16, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm3d(16),
                nn.Conv3d(16, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm3d(16),
                
                # Residuální blok 2 s dilatovanou konvolucí pro větší receptivní pole
                nn.Conv3d(16, 16, kernel_size=3, padding=2, dilation=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm3d(16),
                nn.Conv3d(16, 16, kernel_size=3, padding=2, dilation=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm3d(16),
                
                # Závěrečné vrstvy pro redukci na jeden kanál
                nn.Conv3d(16, 8, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm3d(8),
                
                nn.Conv3d(8, 1, kernel_size=3, padding=1),
                nn.Tanh()  # Tanh pro normalizaci do rozsahu [-1, 1]
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
        Forward pass of the generator
        
        Args:
            x: Input tensor (lesion atlas)
            noise: Optional noise tensor
            
        Returns:
            Generated lesion logits
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
                
                # ZMĚNA: Použijeme více kanálů pro větší variabilitu
                num_noise_channels = torch.randint(3, min(self.noise_dim, 8) + 1, (1,)).item()  # Náhodně 3-8 kanálů (nebo max self.noise_dim)
                noise_indices = torch.randperm(self.noise_dim)[:num_noise_channels]
                
                # ZMĚNA: Přidáme Gaussovský šum pro zvýšení variability
                combined_noise = torch.randn((batch_size, 1, D, H, W), device=x.device) * 0.3
                
                # Kombinace několika kanálů šumu s různými váhami pro větší komplexitu
                for i, idx in enumerate(noise_indices):
                    # ZMĚNA: Použijeme náhodné váhy pro více variability
                    channel_weight = torch.rand(1, device=x.device).item() * 0.5 + 0.5  # Váhy mezi 0.5-1.0
                    combined_noise += expanded_noise[:, idx:idx+1, :, :, :] * channel_weight
                
                # ZMĚNA: Přidáme další zdroj šumu - čistě náhodný šum
                if torch.rand(1).item() < 0.8:  # 80% šance
                    random_noise = torch.randn((batch_size, 1, D, H, W), device=x.device) * 0.5
                    combined_noise = combined_noise + random_noise
                
                noise_3d = combined_noise / num_noise_channels  # Normalizace
                
                # ZMĚNA: Zvýšíme amplifikaci šumu pro agresivnější efekt
                noise_amplification = 2.0 + torch.rand(1).item() * 1.5  # 2.0-3.5x zesílení
                noise_3d = noise_3d * noise_amplification
            else:
                # Předpokládáme, že noise už má správný tvar
                # ZMĚNA: Zvýšená amplifikace i pro předem dodaný šum
                noise_amplification = 2.0 + torch.rand(1).item() * 1.5  # 2.0-3.5x zesílení
                noise_3d = noise * noise_amplification
            
            # Process noise through a more complex network to make it harder to ignore
            processed_noise = self.noise_processor(noise_3d)
            
            # ZMĚNA: Vždy přidáme další náhodnost do zpracovaného šumu s různou frekvencí
            # Vytvoříme dodatečný nízkofrekvenční šum
            low_freq_noise = torch.randn_like(processed_noise) * 0.4  # Zvýšená amplituda
            # Použijeme blur pro vytvoření nízkofrekvenčního šumu
            kernel_size = 5
            padding = kernel_size // 2
            low_freq_noise = F.avg_pool3d(low_freq_noise, kernel_size=kernel_size, stride=1, padding=padding)
            # Kombinujeme s původním šumem
            processed_noise = processed_noise + low_freq_noise
            
            # ZMĚNA: Přidáme zcela nový vysokofrekvenční šum pro více detailů
            high_freq_noise = torch.randn_like(processed_noise) * 0.2
            processed_noise = processed_noise + high_freq_noise
            
            # Přímé přidání částí nezpracovaného šumu do výstupu pro zvýšení variability
            direct_noise_weight = 0.4 + 0.3 * torch.rand(1).item()  # ZMĚNA: Zvýšená váha 0.4-0.7
            processed_noise = processed_noise + (noise_3d * direct_noise_weight)
            
            # ZMĚNA: Občas přidáme nelinearitu pro více komplexní struktury
            if torch.rand(1).item() < 0.5:
                processed_noise = torch.tanh(processed_noise * (1.5 + torch.rand(1).item()))
            
            # Normalizace šumu pro zachování rozumného rozsahu
            processed_noise = F.instance_norm(processed_noise)
            
            # ZMĚNA: Přidáme ještě poslední vrstvu náhodnosti po normalizaci
            if torch.rand(1).item() < 0.3:  # 30% šance
                processed_noise = processed_noise + (torch.randn_like(processed_noise) * 0.2)
            
            # Always concatenate noise with input for correct channel count (2 channels)
            x = torch.cat([x, processed_noise], dim=1)
        else:
            # If not using noise, we still need 2 input channels as expected by SwinUNETR
            batch_size, _, D, H, W = x.shape
            
            # ZMĚNA: I když nepoužíváme Perlin šum, přidáme alespoň nějaký náhodný šum
            zero_channel = torch.randn_like(x) * 0.3
            x = torch.cat([x, zero_channel], dim=1)
        
        # SwinUNETR features
        features = self.swin_unetr(x)
        
        # Final layers
        x = self.final_conv(features)
        
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=2, feature_size=24, depth=4):
        """
        Wasserstein Critic pro WGAN
        
        Args:
            in_channels (int): Number of input channels (atlas + lesion = 2)
            feature_size (int): Initial feature size
            depth (int): Number of downsampling layers
        """
        super(Discriminator, self).__init__()
        
        # Initial convolution with spectral normalization
        layers = [
            nn.utils.spectral_norm(nn.Conv3d(in_channels, feature_size, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Downsampling layers with spectral normalization
        current_feature_size = feature_size
        for _ in range(depth - 1):
            layers += [
                nn.utils.spectral_norm(nn.Conv3d(current_feature_size, current_feature_size * 2, kernel_size=4, stride=2, padding=1, bias=False)),
                nn.BatchNorm3d(current_feature_size * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            current_feature_size *= 2
        
        # Final layers - no sigmoid for WGAN critic
        layers += [
            nn.utils.spectral_norm(nn.Conv3d(current_feature_size, 1, kernel_size=4, stride=1, padding=1, bias=False))
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
                 noise_dim=32,
                 fragmentation_kernel_size=5,
                 perlin_octaves=6,
                 perlin_persistence=0.6,
                 perlin_lacunarity=2.5,
                 perlin_scale=0.2,
                 use_learnable_noise=False,
                 gradient_penalty_weight=30.0):
        """
        SwinGAN for lesion synthesis using Wasserstein GAN (WGAN) with gradient penalty
        
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
            use_learnable_noise (bool): Whether to use learnable Perlin noise parameters
            gradient_penalty_weight (float): Weight for gradient penalty in WGAN-GP
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
            perlin_scale=perlin_scale,
            use_learnable_noise=use_learnable_noise
        )
        self.discriminator = Discriminator(in_channels + out_channels, feature_size)
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.l1_loss = nn.L1Loss()
        
        # Loss weights
        self.lambda_focal = lambda_focal
        self.lambda_l1 = lambda_l1
        self.lambda_fragmentation = lambda_fragmentation
        
        # For generating diverse samples
        self.use_noise = use_noise
        self.noise_dim = noise_dim
        self.use_learnable_noise = use_learnable_noise
        
        # For fragmentation loss
        self.fragmentation_kernel_size = fragmentation_kernel_size
        
        # Store Perlin noise parameters for checkpoints
        self.perlin_octaves = perlin_octaves
        self.perlin_persistence = perlin_persistence
        self.perlin_lacunarity = perlin_lacunarity
        self.perlin_scale = perlin_scale
        
        # For WGAN-GP
        self.gradient_penalty_weight = gradient_penalty_weight
    
    def compute_gradient_penalty(self, atlas, real_lesion, fake_lesion, device):
        """
        Compute gradient penalty for WGAN-GP
        
        Args:
            atlas: Lesion atlas input
            real_lesion: Real lesion data
            fake_lesion: Generated lesion data
            device: Device to use
            
        Returns:
            torch.Tensor: Gradient penalty term
        """
        # Random interpolation coefficient for each sample
        batch_size = real_lesion.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device)
        
        # Create interpolated lesions
        interpolated = alpha * real_lesion + (1 - alpha) * fake_lesion
        interpolated.requires_grad_(True)  # Enable gradient computation
        
        # Get discriminator prediction on interpolated data
        interp_pred = self.discriminator(atlas, interpolated)
        
        # Compute gradients
        grad_outputs = torch.ones_like(interp_pred, device=device)
        gradients = torch.autograd.grad(
            outputs=interp_pred,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Calculate gradient penalty: ((|grad| - 1)^2)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def generator_loss(self, fake_lesion, real_lesion, atlas, fake_pred):
        """
        Generator loss computation for WGAN
        
        Args:
            fake_lesion: Generated lesion
            real_lesion: Ground truth lesion
            atlas: Lesion atlas (frequency map)
            fake_pred: Discriminator prediction on fake lesion
        """
        # WGAN adversarial loss (maximize fake prediction scores)
        adv_loss = -torch.mean(fake_pred)
        
        # Apply sigmoid to fake_lesion for other losses since we're using raw logits
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
                    # Výpočet 3D Gaussovy funkce
                    exponent = -((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) / (2 * sigma ** 2)
                    kernel[0, 0, x, y, z] = np.exp(exponent)
        
        # Normalizace kernelu
        kernel = kernel / kernel.sum()
        
        # Aplikace gaussovského kernelu na predikce pro vyhlazení
        smoothed_fake = F.conv3d(fake_lesion_sigmoid, kernel, padding=center)
        
        # Výpočet ztráty fragmentace - rozdíl mezi původním a vyhlazeným výstupem
        # Léze s více fragmentací budou mít větší rozdíl po vyhlazení
        fragmentation_loss = self.l1_loss(fake_lesion_sigmoid, smoothed_fake)
        
        # Kombinace všech složek ztráty
        total_g_loss = (adv_loss + 
                       self.lambda_focal * focal_loss + 
                       self.lambda_l1 * l1_loss + 
                       self.lambda_fragmentation * fragmentation_loss + 
                       constraint_loss)
        
        # Dictionary s jednotlivými složkami ztráty pro monitorování
        loss_values = {
            'adv_g_loss': adv_loss.item(),
            'focal_loss': focal_loss.item(),
            'l1_loss': l1_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'fragmentation_loss': fragmentation_loss.item(),
            'total_g_loss': total_g_loss.item()
        }
        
        return total_g_loss, loss_values
    
    def discriminator_loss(self, real_pred, fake_pred, atlas, real_lesion, fake_lesion, device):
        """
        Discriminator (critic) loss computation for WGAN-GP
        
        Args:
            real_pred: Discriminator prediction on real lesion
            fake_pred: Discriminator prediction on fake lesion
            atlas: Lesion atlas input
            real_lesion: Real lesion data
            fake_lesion: Fake lesion data
            device: Computation device
        """
        # WGAN adversarial loss
        real_loss = -torch.mean(real_pred)
        fake_loss = torch.mean(fake_pred)
        
        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(atlas, real_lesion, fake_lesion.detach(), device)
        
        # Total discriminator loss
        total_loss = real_loss + fake_loss + self.gradient_penalty_weight * gradient_penalty
        
        return total_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'gradient_penalty': gradient_penalty.item(),
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
                 generator_save_interval=4,
                 n_critic=5):
        """
        Trainer for SwinGAN with WGAN-GP
        
        Args:
            model: SwinGAN model
            optimizer_g: Optimizer for generator
            optimizer_d: Optimizer for discriminator/critic
            device: Device to use
            output_dir: Output directory for saving models and samples
            use_amp: Whether to use automatic mixed precision
            generator_save_interval: Interval for saving generator-only checkpoints
            n_critic: Number of critic updates per generator update (usually 5 for WGAN)
        """
        self.model = model
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.generator_save_interval = generator_save_interval
        self.n_critic = n_critic  # Number of critic iterations per generator iteration
        
        # Parametry pro postprocessing
        self.smooth_sigma = 0.5  # Výchozí hodnota
        self.morph_close_size = 2  # Výchozí hodnota
        
        # For mixed precision training
        if self.use_amp:
            self.scaler_g = GradScaler('cuda' if device == 'cuda' else 'cpu')
            self.scaler_d = GradScaler('cuda' if device == 'cuda' else 'cpu')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create generator checkpoints directory
        self.generator_dir = os.path.join(self.output_dir, 'generator_checkpoints')
        os.makedirs(self.generator_dir, exist_ok=True)
        
        # Create visualization directory
        self.visualization_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
    
    def save_visualization(self, epoch, dataloader, num_samples=4):
        """
        Generuje a ukládá vizualizace lézí pro detekci mode collapse
        
        Args:
            epoch: Aktuální epocha
            dataloader: Dataloader pro získání atlasů
            num_samples: Počet různých vzorků k vygenerování pro každý atlas
        """
        # Kontrola, zda je matplotlib nainstalován
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("VAROVÁNÍ: Modul matplotlib není nainstalován. Vizualizace nebude vytvořena.")
            print("Pro instalaci: pip install matplotlib")
            return
        
        print(f"Generování vizualizací po epoše {epoch+1}...")
        print(f"Použité parametry - smooth_sigma: {self.smooth_sigma}, morph_close_size: {self.morph_close_size}")
        
        # Přepneme model do eval módu
        self.model.eval()
        
        # Vytvoříme adresář pro aktuální epochu
        epoch_dir = os.path.join(self.visualization_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Získáme několik atlasů z dataloaderů
        try:
            atlas_samples = []
            real_lesion_samples = []
            
            # Získáme až 4 různé atlasy z dataloaderu
            max_batches = min(4, len(dataloader))
            dataloader_iter = iter(dataloader)
            
            for i in range(max_batches):
                batch = next(dataloader_iter)
                atlas = batch['atlas'].to(self.device)
                real_lesion = batch['lesion'].to(self.device)
                
                # Uložíme pouze první atlas a lézi z každého batche
                atlas_samples.append(atlas[0:1])
                real_lesion_samples.append(real_lesion[0:1])
            
            # Pro každý atlas vygenerujeme několik různých lézí
            with torch.no_grad():
                for i, (atlas, real_lesion) in enumerate(zip(atlas_samples, real_lesion_samples)):
                    # Připravíme mřížku obrázků - atlas, reálná léze a několik generovaných
                    fig, axes = plt.subplots(2, num_samples + 1, figsize=(4 * (num_samples + 1), 8))
                    
                    # Vytvoříme řez středem objemu
                    atlas_np = atlas[0, 0].cpu().numpy()
                    real_lesion_np = real_lesion[0, 0].cpu().numpy()
                    
                    # Najdeme střední řez, kde je nejvíce nenulových hodnot v atlasu
                    non_zero_counts = [np.count_nonzero(atlas_np[:, :, z]) for z in range(atlas_np.shape[2])]
                    mid_z = np.argmax(non_zero_counts)
                    
                    # Zobrazíme atlas
                    axes[0, 0].imshow(atlas_np[:, :, mid_z], cmap='gray')
                    axes[0, 0].set_title('Atlas')
                    axes[0, 0].axis('off')
                    
                    # Zobrazíme reálnou lézi
                    axes[1, 0].imshow(real_lesion_np[:, :, mid_z], cmap='hot')
                    axes[1, 0].set_title('Reálná léze')
                    axes[1, 0].axis('off')
                    
                    # Vygenerujeme více vzorků s různými šumy
                    for j in range(num_samples):
                        # Generujeme nový šum pro každý vzorek
                        noise = self.model.generator.perlin_generator.generate_batch_noise(
                            batch_size=1,
                            shape=atlas_np.shape,
                            noise_dim=self.model.noise_dim,
                            scale=self.model.perlin_scale,
                            device=self.device
                        )
                        
                        # Vygenerujeme lézi s tímto šumem
                        fake_lesion = self.model.generator(atlas, noise)
                        
                        # Použijeme sigmoid pro konverzi na pravděpodobnostní mapu
                        probability_map = torch.sigmoid(fake_lesion)
                        
                        # Konvertujeme na numpy
                        probability_map_np = probability_map[0, 0].cpu().numpy()
                        
                        # Prahujeme pro získání binární masky
                        binary_mask = (probability_map_np > 0.5).astype(np.float32)
                        
                        # Aplikujeme postprocessing - Gaussovské vyhlazení a morfologické uzavření
                        if self.smooth_sigma > 0:
                            binary_mask = gaussian_filter(binary_mask, sigma=self.smooth_sigma)
                            binary_mask = (binary_mask > 0.5).astype(np.float32)  # Re-threshold
                        
                        if self.morph_close_size > 0:
                            # Vytvoříme strukturní element pro uzavření
                            struct = generate_binary_structure(3, 1)
                            struct = binary_dilation(struct, iterations=self.morph_close_size-1)
                            binary_mask = binary_closing(binary_mask, structure=struct)
                        
                        # Zobrazíme pravděpodobnostní mapu
                        im1 = axes[0, j+1].imshow(probability_map_np[:, :, mid_z], cmap='hot', vmin=0, vmax=1)
                        axes[0, j+1].set_title(f'Pravděpodobnost {j+1}')
                        axes[0, j+1].axis('off')
                        
                        # Zobrazíme binární lézi (po aplikaci postprocessingu)
                        im2 = axes[1, j+1].imshow(binary_mask[:, :, mid_z], cmap='hot', vmin=0, vmax=1)
                        axes[1, j+1].set_title(f'Binární léze {j+1}')
                        axes[1, j+1].axis('off')
                    
                    # Přidáme colorbar
                    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.01, 0.3])
                    fig.colorbar(im1, cax=cbar_ax1)
                    
                    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.01, 0.3])
                    fig.colorbar(im2, cax=cbar_ax2)
                    
                    plt.tight_layout(rect=[0, 0, 0.9, 1])
                    plt.savefig(os.path.join(epoch_dir, f'sample_{i+1}.png'), dpi=150)
                    plt.close()
                
                # Vytvoříme také souhrnný obrázek zobrazující více řezů jednoho atlasu a léze
                if len(atlas_samples) > 0:
                    atlas = atlas_samples[0]
                    atlas_np = atlas[0, 0].cpu().numpy()
                    
                    # Najdeme střed objemu
                    depth = atlas_np.shape[2]
                    
                    # Vygenerujeme jednu lézi
                    noise = self.model.generator.perlin_generator.generate_batch_noise(
                        batch_size=1,
                        shape=atlas_np.shape,
                        noise_dim=self.model.noise_dim,
                        scale=self.model.perlin_scale,
                        device=self.device
                    )
                    
                    fake_lesion = self.model.generator(atlas, noise)
                    probability_map = torch.sigmoid(fake_lesion)
                    probability_map_np = probability_map[0, 0].cpu().numpy()
                    
                    # Prahujeme a aplikujeme postprocessing
                    binary_mask = (probability_map_np > 0.5).astype(np.float32)
                    
                    # Aplikujeme postprocessing - vyhlazení a morfologické uzavření
                    if self.smooth_sigma > 0:
                        binary_mask = gaussian_filter(binary_mask, sigma=self.smooth_sigma)
                        binary_mask = (binary_mask > 0.5).astype(np.float32)  # Re-threshold
                    
                    if self.morph_close_size > 0:
                        struct = generate_binary_structure(3, 1)
                        struct = binary_dilation(struct, iterations=self.morph_close_size-1)
                        binary_mask = binary_closing(binary_mask, structure=struct)
                    
                    # Vytvoříme vizualizaci více řezů
                    num_slices = 5
                    slice_indices = np.linspace(depth // 4, 3 * depth // 4, num_slices).astype(int)
                    
                    fig, axes = plt.subplots(3, num_slices, figsize=(4 * num_slices, 12))
                    
                    for j, z in enumerate(slice_indices):
                        # Atlas
                        axes[0, j].imshow(atlas_np[:, :, z], cmap='gray')
                        axes[0, j].set_title(f'Atlas - Řez {z+1}')
                        axes[0, j].axis('off')
                        
                        # Pravděpodobnostní mapa
                        axes[1, j].imshow(probability_map_np[:, :, z], cmap='hot', vmin=0, vmax=1)
                        axes[1, j].set_title(f'Pravděpodobnost - Řez {z+1}')
                        axes[1, j].axis('off')
                        
                        # Binární léze
                        axes[2, j].imshow(binary_mask[:, :, z], cmap='hot', vmin=0, vmax=1)
                        axes[2, j].set_title(f'Binární léze - Řez {z+1}')
                        axes[2, j].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(epoch_dir, 'multi_slice_view.png'), dpi=150)
                    plt.close()
                
        except Exception as e:
            print(f"Chyba při generování vizualizací: {e}")
            import traceback
            traceback.print_exc()
        
        # Přepneme model zpět do train módu
        self.model.train()
        print(f"Vizualizace uloženy do {epoch_dir}")
    
    def train(self, dataloader, epochs, val_dataloader=None, save_interval=5):
        """Train the model for a specified number of epochs using WGAN-GP."""
        start_time = time.time()
        best_val_loss = float('inf')
        self.model.train()
        
        # For WGAN training
        gen_iterations = 0
        
        for epoch in range(epochs):
            epoch_losses = defaultdict(float)
            
            # Training loop
            for batch_idx, batch in enumerate(dataloader):
                atlas = batch['atlas'].to(self.device)
                real_lesion = batch['lesion'].to(self.device)
                
                ############################
                # (1) Train critic/discriminator
                ############################
                
                # Train the critic for n_critic iterations
                critic_iterations = self.n_critic if gen_iterations > 0 else 100
                
                for _ in range(critic_iterations):
                    self.optimizer_d.zero_grad()
                    
                    with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_amp):
                        # Generate fake lesions
                        fake_lesion = self.model.generator(atlas, noise=None)
                        
                        # Compute discriminator predictions
                        real_pred = self.model.discriminator(atlas, real_lesion)
                        fake_pred = self.model.discriminator(atlas, fake_lesion.detach())
                        
                        # Compute discriminator loss with gradient penalty
                        d_loss, d_losses_dict = self.model.discriminator_loss(
                            real_pred, fake_pred, atlas, real_lesion, fake_lesion.detach(), self.device
                        )
                    
                    # Update discriminator
                    if self.use_amp:
                        self.scaler_d.scale(d_loss).backward()
                        self.scaler_d.step(self.optimizer_d)
                        self.scaler_d.update()
                    else:
                        d_loss.backward()
                        self.optimizer_d.step()
                
                ############################
                # (2) Train generator
                ############################
                
                self.optimizer_g.zero_grad()
                
                with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_amp):
                    # Generate fake lesions
                    fake_lesion = self.model.generator(atlas, noise=None)
                    
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
                
                # Increment generator iterations counter
                gen_iterations += 1
                
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
                  f"D Loss: {epoch_losses['total_d_loss']:.4f} | "
                  f"GP: {epoch_losses.get('gradient_penalty', 0):.4f}")
            
            # Generování vizualizací po každé epoše
            self.save_visualization(epoch, dataloader)
            
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
                    'perlin_scale': self.model.perlin_scale,
                    'use_learnable_noise': self.model.use_learnable_noise
                }, generator_path)
                print(f"Saved generator-only checkpoint to {generator_path}")
                print(f"Checkpoint includes configuration for use_noise={self.model.use_noise}, noise_dim={self.model.noise_dim}, perlin params: octaves={self.model.perlin_octaves}, persistence={self.model.perlin_persistence}, scale={self.model.perlin_scale}, learnable_noise={self.model.use_learnable_noise}")
            
            # Validation
            if val_dataloader is not None and (epoch + 1) % 1 == 0:
                self.validate(epoch, val_dataloader)
    
    def validate(self, epoch, val_dataloader):
        """
        Validate the model using WGAN evaluation
        
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
                    fake_lesion = self.model.generator(atlas, noise=None)
                    
                    # Compute discriminator predictions
                    real_pred = self.model.discriminator(atlas, real_lesion)
                    fake_pred = self.model.discriminator(atlas, fake_lesion)
                    
                    # Compute losses
                    g_loss, g_losses_dict = self.model.generator_loss(
                        fake_lesion, real_lesion, atlas, fake_pred
                    )
                    d_loss, d_losses_dict = self.model.discriminator_loss(
                        real_pred, fake_pred, atlas, real_lesion, fake_lesion, self.device
                    )
                
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
              f"D Loss: {val_losses['total_d_loss']:.4f} | "
              f"GP: {val_losses.get('gradient_penalty', 0):.4f}")
        
        # Return generator loss as the main validation metric
        return val_losses['total_g_loss']


def train_model(args):
    """
    Train the SwinGAN model
    
    Args:
        args: Command line arguments
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    train_dataset = HieLesionDataset(
        args.train_lesion_dir,
        args.lesion_atlas_path,
        filter_empty=args.filter_empty,
        min_non_zero_percentage=args.min_non_zero_percentage
    )
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create validation dataloader if path is provided
    val_dataloader = None
    if args.val_lesion_dir:
        val_dataset = HieLesionDataset(
            args.val_lesion_dir,
            args.lesion_atlas_path,
            filter_empty=args.filter_empty,
            min_non_zero_percentage=args.min_non_zero_percentage
        )
        print(f"Validation dataset size: {len(val_dataset)}")
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
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
        use_noise=args.use_noise,
        noise_dim=args.noise_dim,
        fragmentation_kernel_size=args.fragmentation_kernel_size,
        perlin_octaves=args.perlin_octaves,
        perlin_persistence=args.perlin_persistence,
        perlin_lacunarity=args.perlin_lacunarity,
        perlin_scale=args.perlin_scale,
        use_learnable_noise=args.use_learnable_noise,
        gradient_penalty_weight=args.gradient_penalty_weight
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizers - use lower learning rates for WGAN training stability
    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=0.0002)  # Ponechat vyšší než kritik
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=0.0001)  # Snížení z vyšší hodnoty
    
    # Load optimizer states if provided
    if args.checkpoint:
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    # Trainer
    trainer = SwinGANTrainer(
        model=model,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=device,
        output_dir=args.output_dir,
        use_amp=args.use_amp,
        generator_save_interval=args.generator_save_interval,
        n_critic=args.n_critic
    )
    
    # Nastavení parametrů pro postprocessing
    trainer.smooth_sigma = args.smooth_sigma
    trainer.morph_close_size = args.morph_close_size
    
    # Train model
    trainer.train(
        dataloader=train_dataloader,
        epochs=args.epochs,
        val_dataloader=val_dataloader,
        save_interval=args.save_interval
    )


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
    # High threshold = low coverage
    binary_at_high_threshold = (probability_map > max_threshold).astype(np.float32)
    if atlas_mask is not None:
        binary_at_high_threshold = binary_at_high_threshold * atlas_mask
    low_coverage = compute_lesion_coverage(binary_at_high_threshold, atlas_mask)
    
    # Low threshold = high coverage
    binary_at_low_threshold = (probability_map > min_threshold).astype(np.float32)
    if atlas_mask is not None:
        binary_at_low_threshold = binary_at_low_threshold * atlas_mask
    high_coverage = compute_lesion_coverage(binary_at_low_threshold, atlas_mask)
    
    print(f"Coverage range: [{low_coverage:.4f}% at threshold {max_threshold}] - [{high_coverage:.4f}% at threshold {min_threshold}]")
    
    # Verify that the target is within the achievable range
    if target_coverage > high_coverage:
        print(f"Warning: Target coverage {target_coverage:.4f}% is higher than maximum possible coverage {high_coverage:.4f}%")
        print(f"Using minimum threshold {min_threshold} to achieve maximum coverage")
        return min_threshold, high_coverage, binary_at_low_threshold
    
    if target_coverage < low_coverage:
        print(f"Warning: Target coverage {target_coverage:.4f}% is lower than minimum possible coverage {low_coverage:.4f}%")
        print(f"Using maximum threshold {max_threshold} to achieve minimum coverage")
        return max_threshold, low_coverage, binary_at_high_threshold
    
    # Initialize the binary search with a better initial guess
    if high_coverage != low_coverage:
        # Interpolate to get a better initial threshold based on linear mapping
        # This helps to start closer to the target
        ratio = (target_coverage - low_coverage) / (high_coverage - low_coverage)
        threshold = max_threshold - ratio * (max_threshold - min_threshold)
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
            # Too much coverage, increase threshold (higher threshold = less coverage)
            threshold_min = threshold
            threshold = (threshold + threshold_max) / 2
        else:
            # Too little coverage, decrease threshold (lower threshold = more coverage)
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
    # MODIFIED: Reduce octaves and modify perlin parameters for simpler shapes
    perlin_octaves=3,  # Reduced from 6 to 3
    perlin_persistence=0.4,  # Reduced from 0.6 to 0.4
    perlin_lacunarity=2.0,  # Reduced from 2.5 to 2.0
    perlin_scale=0.1,  # Reduced from 0.2 to 0.1
    min_lesion_size=10,     # Minimální velikost léze (v počtu voxelů)
    # MODIFIED: Reduce smoothing for blockier appearance
    smooth_sigma=0.2,       # Reduced from 0.5 to 0.2
    morph_close_size=1,     # Reduced from 2 to 1
    use_adaptive_threshold=False,  # Použít adaptivní threshold
    training_lesion_dir=None,      # Adresář s trénovacími lézemi
    target_coverage=None,          # Cílové pokrytí lézemi v procentech
    min_adaptive_threshold=0.00001,  # Minimální hodnota pro adaptivní threshold
    max_adaptive_threshold=0.999,    # Maximální hodnota pro adaptivní threshold
    adaptive_threshold_iterations=50,  # Maximální počet iterací pro hledání adaptivního thresholdu
    use_different_target_for_each_sample=False,  # Použít jiný cílový coverage pro každý vzorek
    use_learnable_noise=False,     # Použít naučené parametry Perlin šumu
    gradient_penalty_weight=30.0,  # Weight for gradient penalty in WGAN-GP
    # MODIFIED: Make smoothing and morphological operations less likely
    apply_smoothing=False,         # Changed from True to False to preserve blocky appearance
    apply_morph_close=False        # Changed from True to False to preserve blocky appearance
):
    """
    Generuje léze pomocí natrénovaného modelu SwinGAN.
    
    Args:
        model_checkpoint: Cesta k checkpointu modelu
        lesion_atlas: Cesta k atlasu lézí (frequency map)
        output_file: Cesta pro uložení jedné léze (pouze pro num_samples=1)
        output_dir: Adresář pro uložení více lézí (pro num_samples>1)
        threshold: Prahová hodnota pro binarizaci lézí
        device: Zařízení pro výpočet ('cuda' nebo 'cpu')
        num_samples: Počet lézí k vygenerování
        perlin_octaves: Počet oktáv pro Perlinův šum
        perlin_persistence: Persistence pro Perlinův šum
        perlin_lacunarity: Lacunarity pro Perlinův šum
        perlin_scale: Měřítko pro Perlinův šum
        min_lesion_size: Minimální velikost léze v počtu voxelů
        smooth_sigma: Parametr sigma pro Gaussovské vyhlazení
        morph_close_size: Velikost strukturního elementu pro morfologické uzavření
        use_adaptive_threshold: Použít adaptivní prahování
        training_lesion_dir: Adresář s trénovacími lézemi pro adaptivní prahování
        target_coverage: Cílové pokrytí lézemi v procentech
        min_adaptive_threshold: Minimální hodnota pro adaptivní prahování
        max_adaptive_threshold: Maximální hodnota pro adaptivní prahování
        adaptive_threshold_iterations: Maximální počet iterací pro hledání adaptivního prahu
        use_different_target_for_each_sample: Použít jiný cílový coverage pro každý vzorek
        use_learnable_noise: Použít naučené parametry pro Perlinův šum
        gradient_penalty_weight: Váha penalizace gradientu pro WGAN-GP
        apply_smoothing: Zda aplikovat Gaussovské vyhlazení na generované léze
        apply_morph_close: Zda aplikovat morfologické uzavření na generované léze
    
    Returns:
        Seznam vygenerovaných lézí jako numpy array
    """
    # Validate inputs
    if output_file is None and output_dir is None:
        raise ValueError("Either output_file or output_dir must be provided")
    
    if output_file is not None and num_samples > 1:
        raise ValueError("output_file can only be used when generating a single sample. Use output_dir for multiple samples.")
    
    if use_adaptive_threshold and target_coverage is None and training_lesion_dir is None:
        raise ValueError("For adaptive thresholding, either target_coverage or training_lesion_dir must be provided")
    
    # Ensure output directory exists
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensuring output directory exists: {output_dir}")
    elif output_file is not None:
        # If only output_file is specified, ensure its parent directory exists
        parent_dir = os.path.dirname(output_file)
        if parent_dir:  # Only create if parent_dir is not empty
            os.makedirs(parent_dir, exist_ok=True)
            print(f"Ensuring parent directory exists: {parent_dir}")
    
    # Load lesion atlas
    atlas_nii = nib.load(lesion_atlas)
    atlas_data = atlas_nii.get_fdata()
    header = atlas_nii.header
    affine = atlas_nii.affine
    
    # Nastavení pro NIfTI export
    atlas_header = header
    atlas_affine = affine
    
    # Convert to tensor with batch dimension and channel dimension
    atlas_tensor = torch.from_numpy(atlas_data).float().unsqueeze(0).unsqueeze(0)
    
    # Move to device
    atlas_tensor = atlas_tensor.to(device)
    
    # Create model
    model = SwinGAN(
        in_channels=1, 
        out_channels=1, 
        feature_size=24,
        use_noise=True,
        noise_dim=32,
        perlin_octaves=perlin_octaves,
        perlin_persistence=perlin_persistence,
        perlin_lacunarity=perlin_lacunarity,
        perlin_scale=perlin_scale,
        use_learnable_noise=use_learnable_noise,
        gradient_penalty_weight=gradient_penalty_weight  # Add WGAN parameter
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'generator_state_dict' in checkpoint:
        # Starší formát checkpointu, který ukládá pouze generátor
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        print("Loaded generator weights from checkpoint")
    elif isinstance(checkpoint, dict) and all(k in checkpoint for k in ['epoch', 'generator_state_dict']):
        # Training loop checkpoint
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded generator weights from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        try:
            # Zkusit načíst jako přímý state_dict
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Trying to load partial weights...")
            
            # Pokud selže, pokusíme se načíst pouze dostupné parametry
            model_dict = model.state_dict()
            
            # Filtrujeme state_dict na klíče, které existují v našem modelu
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            
            if not pretrained_dict:
                # Pokud stále nemáme žádné klíče, zkontrolujeme, zda checkpoint obsahuje pouze váhy generátoru
                if any(k.startswith('swin_unetr') for k in checkpoint.keys()):
                    print("Found generator weights with different prefix structure")
                    # Mapování starých klíčů na nové
                    for k, v in checkpoint.items():
                        if k.startswith('swin_unetr'):
                            new_k = f"generator.{k}"
                            if new_k in model_dict:
                                pretrained_dict[new_k] = v
            
            if pretrained_dict:
                print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                raise ValueError("Could not load any parameters from checkpoint")
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Generate samples
    print(f"Generating {num_samples} lesion samples with threshold={threshold}, min_lesion_size={min_lesion_size}, smooth_sigma={smooth_sigma}, morph_close_size={morph_close_size}")
    
    # Initialize array for lesion coverage statistics if using adaptive threshold
    if use_adaptive_threshold:
        coverage_stats = []
    
    # Strukturní element pro morfologické operace
    struct_element = generate_binary_structure(3, 1)  # 3D connectivity
    
    # Generate each sample
    generated_lesions = []
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}")
        
        # Pro každý vzorek vybereme náhodný target coverage z trénovacích dat
        if use_adaptive_threshold and training_lesion_dir is not None:
            # Ověříme, zda training_lesion_dir je řetězec (cesta k adresáři) nebo seznam souborů
            if isinstance(training_lesion_dir, str):
                # Je to cesta k adresáři, najdeme všechny soubory s příponou .nii nebo .nii.gz
                lesion_files = []
                for ext in ['.nii', '.nii.gz']:
                    lesion_files.extend(glob.glob(os.path.join(training_lesion_dir, f'*{ext}')))
                
                if not lesion_files:
                    raise ValueError(f"No .nii or .nii.gz files found in directory: {training_lesion_dir}")
                
                print(f"Found {len(lesion_files)} lesion files in directory")
            else:
                # Předpokládáme, že training_lesion_dir je již seznam souborů
                lesion_files = training_lesion_dir
                
                if not lesion_files:
                    raise ValueError("Empty list of training lesion files provided")
            
            # Nyní vybereme náhodný soubor ze seznamu
            random_training_file = np.random.choice(lesion_files)
            # Načteme jeho lézi a spočítáme coverage
            lesion = nib.load(random_training_file).get_fdata()
            current_target = compute_lesion_coverage(lesion > 0)
            print(f"Náhodně vybraný trénovací soubor: {os.path.basename(random_training_file)}, coverage: {current_target:.6f}%")
        else:
            current_target = target_coverage
        
        # Generate noise if using noise
        noise = None
        if True:
            # Generate a new seed for each sample
            current_seed = np.random.randint(0, 10000)
            torch.manual_seed(current_seed)
            
            # Create noise with matching shape to atlas
            D, H, W = atlas_data.shape
            
            # Initialize PerlinNoiseGenerator with the specified parameters
            # and use it to generate noise
            noise_generator = PerlinNoiseGenerator(
                octaves=perlin_octaves,
                persistence=perlin_persistence,
                lacunarity=perlin_lacunarity,
                seed=current_seed,
                learnable=use_learnable_noise
            )
            
            # Generate the noise vector
            noise = noise_generator.generate_batch_noise(
                batch_size=1,
                shape=(D, H, W),
                noise_dim=32,
                scale=perlin_scale,
                device=device
            )
        
        
        # Generate the lesion
        with torch.no_grad():
            # Pass the atlas through the generator
            fake_lesion = model.generator(atlas_tensor, noise)
            
            # Convert to probability map using sigmoid
            probability_map = torch.sigmoid(fake_lesion)
            
            # Convert to numpy array
            probability_map_np = probability_map.squeeze().cpu().numpy()
            
            # Apply threshold to get binary mask
            if use_adaptive_threshold:
                # Find the threshold that gives the target coverage
                threshold_result = find_adaptive_threshold(
                    probability_map_np, 
                    current_target,  # Použijeme current_target místo target_coverage
                    atlas_mask=(atlas_data > 0),
                    initial_threshold=threshold,
                    min_threshold=min_adaptive_threshold,
                    max_threshold=max_adaptive_threshold,
                    max_iterations=adaptive_threshold_iterations
                )
                # Funkce vrací (threshold, coverage, binary_lesion)
                threshold_value = threshold_result[0]
                found_coverage = threshold_result[1]
                binary_mask = threshold_result[2]
                
                print(f"Sample {i+1}: Using adaptive threshold {threshold_value:.4f} for target coverage {current_target:.6f}%, found coverage: {found_coverage:.6f}%")
            else:
                threshold_value = threshold
                # Apply threshold
                binary_mask = (probability_map_np > threshold_value).astype(np.float32)
            
            # Apply Gaussian smoothing if requested
            if smooth_sigma > 0 and apply_smoothing:
                binary_mask = gaussian_filter(binary_mask, sigma=smooth_sigma)
                binary_mask = (binary_mask > 0.5).astype(np.float32)  # Re-threshold after smoothing
                # Ensure lesions respect atlas mask after smoothing
                atlas_mask = atlas_data > 0
                binary_mask = binary_mask * atlas_mask
            
            # Apply morphological closing if requested
            if morph_close_size > 0 and apply_morph_close:
                # Create a structural element for closing
                close_struct = generate_binary_structure(3, 1)
                close_struct = binary_dilation(close_struct, iterations=morph_close_size-1)
                
                # Apply closing: first dilation, then erosion
                binary_mask = binary_closing(binary_mask, structure=close_struct)
                
                # Convert back to float32
                binary_mask = binary_mask.astype(np.float32)
                
                # Ensure lesions respect atlas mask (don't add voxels where atlas is 0)
                atlas_mask = atlas_data > 0
                binary_mask = binary_mask * atlas_mask
            
            # Remove components smaller than min_lesion_size
            if min_lesion_size > 0:
                # Label connected components
                # Oprava: measure.label vrací tuple (labeled_array, num_features) ve vyšších verzích scipy
                labeled_result = measure.label(binary_mask, structure=struct_element)
                if isinstance(labeled_result, tuple):
                    labeled_array = labeled_result[0]  # Získáme pouze labeled_array z tuple
                else:
                    labeled_array = labeled_result  # Pro starší verze, které vracejí přímo pole
                
                # Get component sizes
                component_sizes = np.bincount(labeled_array.ravel())
                
                # Zero out small components
                too_small = component_sizes < min_lesion_size
                too_small[0] = False  # Don't remove background
                remove_pixels = too_small[labeled_array]
                binary_mask[remove_pixels] = 0
                
                # Count final number of components - oprava
                labeled_result = measure.label(binary_mask, structure=struct_element)
                if isinstance(labeled_result, tuple):
                    labeled_array = labeled_result[0]  # Získáme pouze labeled_array z tuple
                    final_components = labeled_result[1]  # Můžeme získat počet komponent přímo
                else:
                    labeled_array = labeled_result  # Pro starší verze, které vracejí přímo pole
                    # Zjistíme počet komponent (počet různých hodnot kromě pozadí)
                    # V starších verzích scipy musíme počet komponent zjistit pomocí np.max
                    final_components = np.max(labeled_array)
                print(f"Final lesion has {final_components} connected components")
                
            # Inicializujeme binary_lesion jako kopii binary_mask pro další zpracování a výpočty
            binary_lesion = binary_mask.copy()
            
            # Calculate lesion coverage for this sample
            if use_adaptive_threshold:
                # Použijeme již vypočítanou hodnotu coverage z find_adaptive_threshold
                coverage_stats.append(found_coverage)
                print(f"Lesion coverage confirmed: {found_coverage:.6f}% (target: {current_target:.6f}%)")
            
            # Store the generated lesion
            generated_lesions.append(binary_lesion)
            
            # Save the lesion
            if output_file is not None:
                # Save as NIfTI file
                print(f"Saving lesion to {output_file}")
                lesion_nii = nib.Nifti1Image(binary_lesion, atlas_affine, atlas_header)
                nib.save(lesion_nii, output_file)
            elif output_dir is not None:
                # Save as NIfTI file
                # Set background (index 0) size to 0
                if len(component_sizes) > 0:
                    component_sizes[0] = 0
                
                # Upravíme filtrování malých komponent - zvýšíme minimální velikost
                if len(component_sizes) > 1:
                    too_small = np.ones_like(component_sizes, dtype=bool)
                    too_small[0] = False  # background není "too small"
                    
                    # Pro každou komponentu určíme, zda ji zachovat
                    for j in range(1, len(component_sizes)):
                        size = component_sizes[j]
                        if size >= min_lesion_size:  # Používáme min_lesion_size místo nedefinované sample_min_size
                            too_small[j] = False  # Není příliš malá
                        else:
                            # Snížíme šanci na zachování malých bloků
                            prob_keep = min(0.3, (size / min_lesion_size) * 0.5)  # Max 30% šance zachování (namísto 95%)
                            if np.random.random() < prob_keep and size >= 10:  # Dodatečná podmínka: musí mít alespoň 10 voxelů
                                too_small[j] = False  # Zachováme tuto malou komponentu
                    
                    # Aplikujeme masku pro odstranění vybraných malých komponent
                    too_small_mask = too_small[labeled_array]
                    binary_mask[too_small_mask] = 0
            
            # Přiřadíme binary_mask do binary_lesion pro další zpracování
            binary_lesion = binary_mask.copy()
            
            # Pro blokovitější vzhled snížíme Gaussovské vyhlazení na minimum nebo ho přeskočíme
            if smooth_sigma > 0 and apply_smoothing and np.random.random() < 0.2:  # Reduced chance from 0.8 to 0.2
                # Použijeme vyšší sigma pro méně ostré hrany
                sample_sigma = smooth_sigma * 0.5  # Further reduced from 0.7 to 0.5
                
                print(f"  Using smoothing with sigma: {sample_sigma:.2f}")
                smoothed = gaussian_filter(binary_lesion.astype(float), sigma=sample_sigma)
                
                # Adaptivní threshold s vysokou hodnotou pro zachování ostrých přechodů
                orig_nonzero = np.count_nonzero(binary_lesion)
                if orig_nonzero > 0:
                    sorted_values = np.sort(smoothed.ravel())
                    threshold_idx = max(0, len(sorted_values) - orig_nonzero)
                    adaptive_threshold = sorted_values[threshold_idx]
                    # Mírně zvýšíme threshold pro ostřejší hrany
                    adaptive_threshold = max(0.05, adaptive_threshold * 1.1)  # Increased from 1.05 to 1.1
                    binary_lesion = smoothed > adaptive_threshold
                    # Ensure binary_lesion respects atlas mask after adaptive thresholding
                    binary_lesion = binary_lesion.astype(np.float32) * (atlas_data > 0)
                
                # MODIFIED: Reduce chance of adding block structures
                if np.random.random() < 0.3:  # Reduced from 0.7 to 0.3
                    # Najdeme hranice léze
                    binary_bool = binary_lesion.astype(bool)
                    dilated_bool = binary_dilation(binary_bool, iterations=1)
                    # Ensure dilation respects atlas mask
                    atlas_mask = atlas_data > 0
                    dilated_bool = dilated_bool & atlas_mask
                    border = dilated_bool & ~binary_bool
                    
                    # Vytvoříme náhodné bloky na hranách - pixelový šum
                    block_noise = np.random.random(border.shape) < 0.15  # Reduced from 0.3 to 0.15
                    block_noise = block_noise & border
                    
                    # Přidáme pixelový šum k lézi - zaručíme že pracujeme s bool typem
                    binary_bool = binary_lesion.astype(bool)
                    binary_bool = binary_bool | block_noise
                    binary_lesion = binary_bool.astype(binary_lesion.dtype)
                    
                    # MODIFIED: Drastically reduce isolated blocks
                    if np.random.random() < 0.1:  # Reduced from 0.4 to 0.1
                        # Vytvoříme broader border
                        binary_bool = binary_lesion.astype(bool)
                        dilated_1 = binary_dilation(binary_bool, iterations=1)
                        # Ensure dilation respects atlas mask
                        dilated_1 = dilated_1 & atlas_mask
                        dilated_3 = binary_dilation(binary_bool, iterations=2)  # Further reduced from 3 to 2
                        # Ensure dilation respects atlas mask
                        dilated_3 = dilated_3 & atlas_mask
                        outer_border = dilated_3 & ~dilated_1
                        
                        # Vytvoříme méně častější izolované bloky
                        isolated_blocks = np.random.random(outer_border.shape) < 0.01  # Reduced from 0.05 to 0.01
                        isolated_blocks = isolated_blocks & outer_border
                        
                        # Přidáme izolované bloky - zaručíme že pracujeme s bool typem
                        binary_bool = binary_lesion.astype(bool)
                        binary_bool = binary_bool | isolated_blocks
                        binary_lesion = binary_bool.astype(binary_lesion.dtype)
            else:
                # MODIFIED: Make base blocks cleaner with less noise at edges
                print("  Skipping smoothing to preserve blocky appearance")
                
                # Přidáme pixelový šum na hranách
                # Nejprve konvertujeme binary_lesion na bool pro logické operace
                binary_bool = binary_lesion.astype(bool)
                dilated_bool = binary_dilation(binary_bool, iterations=1)
                # Ensure dilation respects atlas mask
                atlas_mask = atlas_data > 0
                dilated_bool = dilated_bool & atlas_mask
                border = dilated_bool & ~binary_bool
                pixel_noise = np.random.random(border.shape) < 0.05  # Further reduced from 0.15 to 0.05
                # Použijeme logický OR na bool a pak konvertujeme zpět na původní typ
                binary_bool = binary_bool | (pixel_noise & border)
                binary_lesion = binary_bool.astype(binary_lesion.dtype)
                
                # MODIFIED: Further reduce isolated pixels for cleaner shapes
                if np.random.random() < 0.2:  # Reduced from 0.5 to 0.2
                    # Opět používáme booleovský typ pro správné logické operace
                    binary_bool = binary_lesion.astype(bool)
                    dilated_1 = binary_dilation(binary_bool, iterations=1)
                    # Ensure dilation respects atlas mask
                    dilated_1 = dilated_1 & atlas_mask
                    dilated_4 = binary_dilation(binary_bool, iterations=2)  # Further reduced from 3 to 2
                    # Ensure dilation respects atlas mask
                    dilated_4 = dilated_4 & atlas_mask
                    outer_region = dilated_4 & ~dilated_1
                    isolated_pixels = np.random.random(outer_region.shape) < 0.001  # Reduced from 0.01 to 0.001
                    # Použijeme logický OR na bool a pak konvertujeme zpět na původní typ
                    binary_bool = binary_bool | (isolated_pixels & outer_region)
                    binary_lesion = binary_bool.astype(binary_lesion.dtype)

            generated_lesions.append(binary_lesion)
            
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
            
            # Výpočet objemu v procentech celkového objemu mozku před uložením
            total_brain_volume = np.count_nonzero(atlas_data > 0)  # Počet voxelů, kde je atlas nenulový
            lesion_volume_voxels = binary_lesion.sum()
            lesion_volume_percentage = (lesion_volume_voxels / total_brain_volume) * 100.0
            
            # Skip saving files with 0.000% coverage (no lesions)
            if lesion_volume_percentage <= 0.0:
                if num_samples > 1:
                    print(f"Sample {i+1}: No lesions detected (0.000% coverage) - skipping save")
                else:
                    print("No lesions detected (0.000% coverage) - skipping save")
                continue
            
            # Create a new NIfTI image and save it
            # Ensure data is of supported type (not bool)
            binary_lesion_for_saving = binary_lesion.astype(np.int16)  # Convert to int16, which is supported by NIfTI
            lesion_img = nib.Nifti1Image(binary_lesion_for_saving, atlas_affine, atlas_header)
            
            # Ensure the parent directory of sample_output_file exists
            sample_output_dir = os.path.dirname(sample_output_file)
            if sample_output_dir and not os.path.exists(sample_output_dir):
                print(f"Creating directory for sample: {sample_output_dir}")
                os.makedirs(sample_output_dir, exist_ok=True)
            
            nib.save(lesion_img, sample_output_file)
            
            # Print some statistics
            # measure.label vrací tuple (labeled_array, num_features) ve vyšších verzích scipy
            # V naší verzi musíme počet lézí určit jinak
            labeled_result = measure.label(binary_lesion)
            if isinstance(labeled_result, tuple):
                labeled_array = labeled_result[0]
                num_lesions = labeled_result[1]  # Počet komponent přímo z výsledku
            else:
                labeled_array = labeled_result
                num_lesions = np.max(labeled_array)  # Zjistíme počet pomocí maxima hodnot
            
            # Také vypočítáme objem v ml pro úplnost
            lesion_volume_ml = lesion_volume_voxels * np.prod(np.abs(np.diag(atlas_affine)[:3])) / 1000.0  # in ml
            
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


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='SwinGAN for HIE lesion synthesis')
    subparsers = parser.add_subparsers(dest='mode', help='Mode')
    
    # Create train subparser
    train_parser = subparsers.add_parser('train', help='Train the model')
    
    # Data paths
    train_parser.add_argument('--train_lesion_dir', type=str, required=True, help='Path to directory with training lesion files')
    train_parser.add_argument('--val_lesion_dir', type=str, help='Path to directory with validation lesion files')
    train_parser.add_argument('--lesion_atlas_path', type=str, required=True, help='Path to lesion atlas file')
    train_parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    # Training parameters
    train_parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--lr_generator', type=float, default=1e-4, help='Generator learning rate')
    train_parser.add_argument('--lr_discriminator', type=float, default=1e-5, help='Discriminator learning rate')
    train_parser.add_argument('--save_interval', type=int, default=5, help='Interval for saving checkpoints')
    train_parser.add_argument('--generator_save_interval', type=int, default=4, help='Interval for saving generator-only checkpoints')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    train_parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    train_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for resuming training')
    train_parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    train_parser.add_argument('--n_critic', type=int, default=5, help='Number of critic updates per generator update (WGAN)')
    
    # Model parameters
    train_parser.add_argument('--feature_size', type=int, default=24, help='Feature size for SwinUNETR')
    train_parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
    train_parser.add_argument('--lambda_focal', type=float, default=10.0, help='Weight for focal loss')
    train_parser.add_argument('--lambda_l1', type=float, default=5.0, help='Weight for L1 loss')
    train_parser.add_argument('--lambda_fragmentation', type=float, default=50.0, help='Weight for fragmentation loss')
    train_parser.add_argument('--focal_alpha', type=float, default=0.75, help='Alpha parameter for focal loss')
    train_parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    train_parser.add_argument('--use_noise', action='store_true', default=True, help='Use noise injection for diversity')
    train_parser.add_argument('--noise_dim', type=int, default=32, help='Dimension of the noise vector')
    train_parser.add_argument('--fragmentation_kernel_size', type=int, default=5, help='Size of kernel for fragmentation loss')
    train_parser.add_argument('--perlin_octaves', type=int, default=6, help='Number of octaves for Perlin noise')
    train_parser.add_argument('--perlin_persistence', type=float, default=0.6, help='Persistence parameter for Perlin noise')
    train_parser.add_argument('--perlin_lacunarity', type=float, default=2.5, help='Lacunarity parameter for Perlin noise')
    train_parser.add_argument('--perlin_scale', type=float, default=0.2, help='Scale parameter for Perlin noise')
    train_parser.add_argument('--use_learnable_noise', action='store_true', help='Use learnable Perlin noise parameters')
    train_parser.add_argument('--gradient_penalty_weight', type=float, default=30.0, help='Weight for gradient penalty in WGAN-GP')
    train_parser.add_argument('--smooth_sigma', type=float, default=0.5, help='Sigma for Gaussian smoothing during postprocessing')
    train_parser.add_argument('--morph_close_size', type=int, default=2, help='Size of structural element for morphological closing')
    
    # Dataset parameters
    train_parser.add_argument('--filter_empty', action='store_true', help='Filter out empty lesion files')
    train_parser.add_argument('--min_non_zero_percentage', type=float, default=0.00001, help='Minimum percentage of non-zero voxels to include sample')
    
    # Create generate subparser
    generate_parser = subparsers.add_parser('generate', help='Generate lesions using a trained model')
    
    # Generation parameters
    generate_parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to generator checkpoint')
    generate_parser.add_argument('--lesion_atlas', type=str, required=True, help='Path to lesion atlas file')
    generate_parser.add_argument('--output_file', type=str, help='Output file path (for single sample)')
    generate_parser.add_argument('--output_dir', type=str, help='Output directory (for multiple samples)')
    generate_parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    generate_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    generate_parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    generate_parser.add_argument('--perlin_octaves', type=int, default=4, help='Number of octaves for Perlin noise')
    generate_parser.add_argument('--perlin_persistence', type=float, default=0.5, help='Persistence parameter for Perlin noise')
    generate_parser.add_argument('--perlin_lacunarity', type=float, default=2.0, help='Lacunarity parameter for Perlin noise')
    generate_parser.add_argument('--perlin_scale', type=float, default=0.1, help='Scale parameter for Perlin noise')
    generate_parser.add_argument('--min_lesion_size', type=int, default=10, help='Minimum lesion size in voxels')
    generate_parser.add_argument('--smooth_sigma', type=float, default=0.5, help='Sigma for Gaussian smoothing')
    generate_parser.add_argument('--morph_close_size', type=int, default=2, help='Size of structural element for morphological closing')
    generate_parser.add_argument('--use_adaptive_threshold', action='store_true', help='Use adaptive thresholding')
    generate_parser.add_argument('--training_lesion_dir', type=str, help='Directory with training lesions for adaptive thresholding')
    generate_parser.add_argument('--target_coverage', type=float, help='Target lesion coverage in percent')
    generate_parser.add_argument('--min_adaptive_threshold', type=float, default=0.00001, help='Minimum value for adaptive threshold')
    generate_parser.add_argument('--max_adaptive_threshold', type=float, default=0.999, help='Maximum value for adaptive threshold')
    generate_parser.add_argument('--adaptive_threshold_iterations', type=int, default=50, help='Maximum number of iterations for adaptive threshold search')
    generate_parser.add_argument('--use_different_target_for_each_sample', action='store_true', help='Use different target coverage for each sample')
    generate_parser.add_argument('--use_learnable_noise', action='store_true', help='Use learnable parameters for Perlin noise')
    generate_parser.add_argument('--apply_smoothing', action='store_true', default=True, help='Apply Gaussian smoothing to lesions')
    generate_parser.add_argument('--no_smoothing', action='store_false', dest='apply_smoothing', help='Disable Gaussian smoothing')
    generate_parser.add_argument('--apply_morph_close', action='store_true', default=True, help='Apply morphological closing to lesions')
    generate_parser.add_argument('--no_morph_close', action='store_false', dest='apply_morph_close', help='Disable morphological closing')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute appropriate function based on mode
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
            use_different_target_for_each_sample=args.use_different_target_for_each_sample,
            use_learnable_noise=args.use_learnable_noise,
            apply_smoothing=args.apply_smoothing,
            apply_morph_close=args.apply_morph_close
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
