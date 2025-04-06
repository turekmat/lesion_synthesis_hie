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
            
            # Zvýšíme celkovou amplitudu šumu pro větší vliv na výstup
            amplification_factor = 1.5 + torch.rand(1).item() * 0.5  # Náhodné zesílení 1.5-2.0x
            amplitudes = [amp * amplification_factor for amp in amplitudes]
            
            max_amplitude = sum(amplitudes)
            octaves_to_use = self.octaves
            noise_weights = torch.rand(4, device=device)
            noise_weights = noise_weights / noise_weights.sum()  # Normalizace vah
        
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
                
                # We'll also randomize the frequency multipliers to get more varied patterns
                freq_multiplier_x = 1.0 + random_tensor[1, 0] * 0.7  # Zvýšený rozsah z 0.5 na 0.7
                freq_multiplier_y = 1.0 + random_tensor[1, 1] * 0.7
                freq_multiplier_z = 1.0 + random_tensor[1, 2] * 0.7
            
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
    def __init__(self, in_channels=1, out_channels=1, feature_size=24, dropout_rate=0.0, use_noise=True, noise_dim=16,
                 perlin_octaves=4, perlin_persistence=0.5, perlin_lacunarity=2.0, perlin_scale=0.1, use_learnable_noise=False):
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
                # Vezmeme více dimenzí (kanálů) pro noise_processor pro větší variabilitu
                # Původně se brala jen jedna dimenze, nyní vezmeme náhodně 2-4 dimenze a zkombinujeme je
                num_noise_channels = torch.randint(2, 5, (1,)).item()  # Náhodně 2-4 kanály
                noise_indices = torch.randperm(self.noise_dim)[:num_noise_channels]
                combined_noise = torch.zeros((batch_size, 1, D, H, W), device=x.device)
                
                # Kombinace několika kanálů šumu s různými váhami pro větší komplexitu
                for i, idx in enumerate(noise_indices):
                    channel_weight = 1.0 / (i + 1)  # Klesající váhy pro další kanály
                    combined_noise += expanded_noise[:, idx:idx+1, :, :, :] * channel_weight
                
                noise_3d = combined_noise / num_noise_channels  # Normalizace
                
                # Přidáme náhodné zesílení šumu pro větší nepředvídatelnost
                noise_amplification = 1.0 + torch.rand(1).item()  # Random mezi 1.0 a 2.0
                noise_3d = noise_3d * noise_amplification
            else:
                # Předpokládáme, že noise už má správný tvar
                noise_3d = noise * (1.0 + 0.5 * torch.rand(1).item())  # Náhodné zesílení
            
            # Process noise through a more complex network to make it harder to ignore
            # Původní noise_processor byl příliš jednoduchý, rozšíříme ho
            processed_noise = self.noise_processor(noise_3d)
            
            # Přidáme další náhodnost do zpracovaného šumu s různou frekvencí
            if torch.rand(1).item() < 0.7:  # 70% šance na přidání další náhodnosti
                # Vytvoříme dodatečný nízkofrekvenční šum
                low_freq_noise = torch.randn_like(processed_noise) * 0.2
                # Použijeme blur pro vytvoření nízkofrekvenčního šumu
                kernel_size = 5
                padding = kernel_size // 2
                low_freq_noise = F.avg_pool3d(low_freq_noise, kernel_size=kernel_size, stride=1, padding=padding)
                # Kombinujeme s původním šumem
                processed_noise = processed_noise + low_freq_noise
            
            # Přímé přidání částí nezpracovaného šumu do výstupu pro zvýšení variability
            direct_noise_weight = 0.3 * torch.rand(1).item()  # Náhodná váha 0-0.3
            processed_noise = processed_noise + (noise_3d * direct_noise_weight)
            
            # Normalizace šumu pro zachování rozumného rozsahu
            processed_noise = F.instance_norm(processed_noise)
            
            # Always concatenate noise with input for correct channel count (2 channels)
            x = torch.cat([x, processed_noise], dim=1)
        else:
            # If not using noise, we still need 2 input channels as expected by SwinUNETR
            batch_size, _, D, H, W = x.shape
            zero_channel = torch.zeros_like(x)
            x = torch.cat([x, zero_channel], dim=1)
        
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
                 perlin_scale=0.1,
                 use_learnable_noise=False):
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
            use_learnable_noise (bool): Whether to use learnable Perlin noise parameters
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
        self.bce_loss = nn.BCEWithLogitsLoss()
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
                    fake_lesion = self.model.generator(atlas, noise=None)
                    
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
                    # Generate fake lesions - explicitly pass None for noise parameter
                    fake_lesion = self.model.generator(atlas, noise=None)
                    
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
    Train the SwinGAN model with the given arguments
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup dataset
    train_dataset = HieLesionDataset(
        lesion_dir=args.train_lesion_dir,
        lesion_atlas_path=args.lesion_atlas_path,
        filter_empty=args.filter_empty,
        min_non_zero_percentage=args.min_non_zero_percentage
    )
    
    val_dataset = None
    if args.val_lesion_dir:
        val_dataset = HieLesionDataset(
            lesion_dir=args.val_lesion_dir,
            lesion_atlas_path=args.lesion_atlas_path,
            filter_empty=args.filter_empty,
            min_non_zero_percentage=args.min_non_zero_percentage
        )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = None
    if val_dataset:
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
        use_learnable_noise=args.use_learnable_noise
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizers
    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=args.lr_generator, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr_discriminator, betas=(0.5, 0.999))
    
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
        generator_save_interval=args.generator_save_interval
    )
    
    # Train
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
    use_different_target_for_each_sample=False,  # Použít jiný cílový coverage pro každý vzorek
    use_learnable_noise=False     # Použít naučené parametry Perlin šumu
):
    # Validate inputs
    if output_file is None and output_dir is None:
        raise ValueError("Either output_file or output_dir must be provided")
    
    if output_file is not None and num_samples > 1:
        raise ValueError("output_file can only be used with num_samples=1")
    
    if use_adaptive_threshold and target_coverage is None and training_lesion_dir is None:
        raise ValueError("For adaptive thresholding, either target_coverage or training_lesion_dir must be provided")
    
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load the atlas
    atlas_data = nib.load(lesion_atlas).get_fdata()
    atlas_affine = nib.load(lesion_atlas).affine
    atlas_header = nib.load(lesion_atlas).header
    
    # Normalize the atlas to [0, 1]
    atlas_data = atlas_data / np.max(atlas_data)
    
    # Convert to torch tensor and add batch and channel dimensions
    atlas_tensor = torch.from_numpy(atlas_data).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Compute target coverage if using adaptive threshold based on training data
    if use_adaptive_threshold and target_coverage is None and training_lesion_dir is not None:
        print("Computing target coverage from training data...")
        target_coverage = compute_target_coverage_from_training(
            training_lesion_dir, 
            sample_count=10
        )
        print(f"Using target coverage of {target_coverage:.6f}%")
    
    # Get individual target coverages if using different targets for each sample
    target_coverages = []
    if use_adaptive_threshold and use_different_target_for_each_sample and training_lesion_dir is not None:
        print("Computing individual target coverages from training data...")
        target_coverages = compute_target_coverage_from_training(
            training_lesion_dir, 
            sample_count=min(30, num_samples),  # Limit the number of samples to avoid long computation
            return_list=True,
            random_seed=42  # For reproducibility
        )
        # If we need more samples than we have coverages, repeat the list
        while len(target_coverages) < num_samples:
            target_coverages.extend(target_coverages[:num_samples - len(target_coverages)])
        print(f"Using {len(target_coverages)} different target coverages ranging from {min(target_coverages):.6f}% to {max(target_coverages):.6f}%")
    
    # Load the generator
    checkpoint = torch.load(model_checkpoint, map_location=device)
    
    # Extract configuration from checkpoint
    config = {}
    for key in ['feature_size', 'dropout_rate', 'use_noise', 'noise_dim',
                'perlin_octaves', 'perlin_persistence', 'perlin_lacunarity', 'perlin_scale']:
        config[key] = checkpoint.get(key, None)
    
    # Handle missing config values and override with arguments
    feature_size = config.get('feature_size', 24)
    dropout_rate = config.get('dropout_rate', 0.0)
    use_noise = config.get('use_noise', True)
    noise_dim = config.get('noise_dim', 16)
    
    # Override Perlin noise parameters if provided
    if 'perlin_octaves' in checkpoint and perlin_octaves is None:
        perlin_octaves = checkpoint['perlin_octaves']
    
    if 'perlin_persistence' in checkpoint and perlin_persistence is None:
        perlin_persistence = checkpoint['perlin_persistence']
    
    if 'perlin_lacunarity' in checkpoint and perlin_lacunarity is None:
        perlin_lacunarity = checkpoint['perlin_lacunarity']
    
    if 'perlin_scale' in checkpoint and perlin_scale is None:
        perlin_scale = checkpoint['perlin_scale']
    
    # Check for learnable_noise in checkpoint
    if 'use_learnable_noise' in checkpoint:
        use_learnable_noise = checkpoint['use_learnable_noise']
    
    # Create the generator model
    generator = Generator(
        in_channels=1,
        out_channels=1,
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
    
    # Load state dict
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    # Move to device and set to evaluation mode
    generator.to(device)
    generator.eval()
    
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
        
        # Generate noise if using noise
        noise = None
        if use_noise:
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
                noise_dim=noise_dim,
                scale=perlin_scale,
                device=device
            )
        
        # Generate the lesion
        with torch.no_grad():
            # Forward pass with noise
            fake_lesion = generator(atlas_tensor, noise)
            
            # Apply sigmoid to get probability map
            fake_lesion_prob = torch.sigmoid(fake_lesion)
            
            # Convert to numpy for post-processing
            lesion_prob_np = fake_lesion_prob.squeeze().cpu().numpy()
            
            # Binarize the lesion
            if use_adaptive_threshold:
                # Get target coverage for this sample
                if use_different_target_for_each_sample and target_coverages:
                    current_target = target_coverages[i % len(target_coverages)]
                else:
                    current_target = target_coverage
                
                print(f"Finding adaptive threshold for target coverage of {current_target:.6f}%")
                
                # Compute optimal threshold for target coverage
                adaptive_threshold, actual_coverage, _ = find_adaptive_threshold(
                    lesion_prob_np,
                    current_target,
                    atlas_mask=(atlas_data > 0),
                    initial_threshold=threshold,
                    min_threshold=min_adaptive_threshold,
                    max_threshold=max_adaptive_threshold,
                    max_iterations=adaptive_threshold_iterations
                )
                
                print(f"Using adaptive threshold: {adaptive_threshold:.6f}")
                print(f"Actual coverage at this threshold: {actual_coverage:.6f}%")
                # Use the adaptive threshold
                lesion_binary = (lesion_prob_np >= adaptive_threshold).astype(np.float32)
            else:
                # Use fixed threshold
                lesion_binary = (lesion_prob_np >= threshold).astype(np.float32)
            
            # Apply Gaussian smoothing if requested
            if smooth_sigma > 0:
                lesion_binary = gaussian_filter(lesion_binary, sigma=smooth_sigma)
                lesion_binary = (lesion_binary > 0.5).astype(np.float32)  # Re-threshold after smoothing
            
            # Apply morphological closing if requested
            if morph_close_size > 0:
                # Create a structural element for closing
                close_struct = generate_binary_structure(3, 1)
                close_struct = binary_dilation(close_struct, iterations=morph_close_size-1)
                
                # Apply closing: first dilation, then erosion
                lesion_binary = binary_closing(lesion_binary, structure=close_struct)
                
                # Convert back to float32
                lesion_binary = lesion_binary.astype(np.float32)
            
            # Remove components smaller than min_lesion_size
            if min_lesion_size > 0:
                # Label connected components
                # Oprava: measure.label nemá parametr return_num v této verzi scipy
                labeled_array = measure.label(lesion_binary, structure=struct_element)
                
                # Get component sizes
                component_sizes = np.bincount(labeled_array.ravel())
                
                # Zero out small components
                too_small = component_sizes < min_lesion_size
                too_small[0] = False  # Don't remove background
                remove_pixels = too_small[labeled_array]
                lesion_binary[remove_pixels] = 0
                
                # Count final number of components - oprava
                labeled_array = measure.label(lesion_binary, structure=struct_element)
                # Zjistíme počet komponent (počet různých hodnot kromě pozadí)
                # V starších verzích scipy musíme počet komponent zjistit pomocí np.max
                final_components = np.max(labeled_array)
                print(f"Final lesion has {final_components} connected components")
            
            # Calculate lesion coverage for this sample
            if use_adaptive_threshold:
                current_coverage = compute_lesion_coverage(lesion_binary, atlas_mask=(atlas_data > 0))
                coverage_stats.append(current_coverage)
                print(f"Lesion coverage: {current_coverage:.6f}% (target: {current_target:.6f}%)")
            
            # Store the generated lesion
            generated_lesions.append(lesion_binary)
            
            # Save the lesion
            if output_file is not None:
                # Save as NIfTI file
                print(f"Saving lesion to {output_file}")
                lesion_nii = nib.Nifti1Image(lesion_binary, atlas_affine, atlas_header)
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
                    binary_lesion[too_small_mask] = 0
            
            # Pro blokovitější vzhled snížíme Gaussovské vyhlazení na minimum nebo ho přeskočíme
            if smooth_sigma > 0 and np.random.random() < 0.8:  # Zvýšíme šanci na vyhlazení z 50% na 80%
                # Použijeme vyšší sigma pro méně ostré hrany
                sample_sigma = smooth_sigma * 0.7  # Méně snížená hodnota (než původní 0.3)
                
                print(f"  Using smoothing with sigma: {sample_sigma:.2f}")
                smoothed = gaussian_filter(binary_lesion.astype(float), sigma=sample_sigma)
                
                # Adaptivní threshold s vysokou hodnotou pro zachování ostrých přechodů
                orig_nonzero = np.count_nonzero(binary_lesion)
                if orig_nonzero > 0:
                    sorted_values = np.sort(smoothed.ravel())
                    threshold_idx = max(0, len(sorted_values) - orig_nonzero)
                    adaptive_threshold = sorted_values[threshold_idx]
                    # Mírně zvýšíme threshold pro ostřejší hrany
                    adaptive_threshold = max(0.05, adaptive_threshold * 1.05)
                    binary_lesion = smoothed > adaptive_threshold
                
                # Místo přidávání plynulého šumu přidáme blokové struktury
                if np.random.random() < 0.7:
                    # Najdeme hranice léze
                    binary_bool = binary_lesion.astype(bool)
                    dilated_bool = binary_dilation(binary_bool, iterations=1)
                    border = dilated_bool & ~binary_bool
                    
                    # Vytvoříme náhodné bloky na hranách - pixelový šum
                    block_noise = np.random.random(border.shape) < 0.3
                    block_noise = block_noise & border
                    
                    # Přidáme pixelový šum k lézi - zaručíme že pracujeme s bool typem
                    binary_bool = binary_lesion.astype(bool)
                    binary_bool = binary_bool | block_noise
                    binary_lesion = binary_bool.astype(binary_lesion.dtype)
                    
                    # Pro některé léze přidáme další izolované bloky v blízkosti
                    if np.random.random() < 0.4:
                        # Vytvoříme broader border
                        binary_bool = binary_lesion.astype(bool)
                        dilated_1 = binary_dilation(binary_bool, iterations=1)
                        dilated_3 = binary_dilation(binary_bool, iterations=3)  # Sníženo z iterations=4 na 3
                        outer_border = dilated_3 & ~dilated_1
                        
                        # Vytvoříme méně častější izolované bloky
                        isolated_blocks = np.random.random(outer_border.shape) < 0.05
                        isolated_blocks = isolated_blocks & outer_border
                        
                        # Přidáme izolované bloky - zaručíme že pracujeme s bool typem
                        binary_bool = binary_lesion.astype(bool)
                        binary_bool = binary_bool | isolated_blocks
                        binary_lesion = binary_bool.astype(binary_lesion.dtype)
            else:
                # Pro plně blokovitý vzhled přidáme pixelový šum přímo bez vyhlazení
                print("  Skipping smoothing to preserve blocky appearance")
                
                # Přidáme pixelový šum na hranách
                # Nejprve konvertujeme binary_lesion na bool pro logické operace
                binary_bool = binary_lesion.astype(bool)
                dilated_bool = binary_dilation(binary_bool, iterations=1)
                border = dilated_bool & ~binary_bool
                pixel_noise = np.random.random(border.shape) < 0.15  # Snížení z 0.25 na 0.15 pro méně šumu
                # Použijeme logický OR na bool a pak konvertujeme zpět na původní typ
                binary_bool = binary_bool | (pixel_noise & border)
                binary_lesion = binary_bool.astype(binary_lesion.dtype)
                
                # Pro ještě blokovitější vzhled můžeme přidat i nepřipojené pixely
                if np.random.random() < 0.5:
                    # Opět používáme booleovský typ pro správné logické operace
                    binary_bool = binary_lesion.astype(bool)
                    dilated_1 = binary_dilation(binary_bool, iterations=1)
                    dilated_4 = binary_dilation(binary_bool, iterations=3)  # Sníženo z iterations=4 na 3
                    outer_region = dilated_4 & ~dilated_1
                    isolated_pixels = np.random.random(outer_region.shape) < 0.01  # Sníženo z 0.03 na 0.01
                    # Použijeme logický OR na bool a pak konvertujeme zpět na původní typ
                    binary_bool = binary_bool | (isolated_pixels & outer_region)
                    binary_lesion = binary_bool.astype(binary_lesion.dtype)

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
            lesion_img = nib.Nifti1Image(binary_lesion_for_saving, atlas_img.affine)
            nib.save(lesion_img, sample_output_file)
            
            # Print some statistics
            # measure.label vrací tuple (labeled_array, num_features) ve vyšších verzích scipy
            # V naší verzi musíme počet lézí určit jinak
            labeled_array = measure.label(binary_lesion)
            num_lesions = np.max(labeled_array)
            
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
    train_parser.add_argument('--lr_discriminator', type=float, default=4e-4, help='Discriminator learning rate')
    train_parser.add_argument('--save_interval', type=int, default=5, help='Interval for saving checkpoints')
    train_parser.add_argument('--generator_save_interval', type=int, default=4, help='Interval for saving generator-only checkpoints')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    train_parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    train_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for resuming training')
    train_parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    
    # Model parameters
    train_parser.add_argument('--feature_size', type=int, default=24, help='Feature size for SwinUNETR')
    train_parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
    train_parser.add_argument('--lambda_focal', type=float, default=10.0, help='Weight for focal loss')
    train_parser.add_argument('--lambda_l1', type=float, default=5.0, help='Weight for L1 loss')
    train_parser.add_argument('--lambda_fragmentation', type=float, default=50.0, help='Weight for fragmentation loss')
    train_parser.add_argument('--focal_alpha', type=float, default=0.75, help='Alpha parameter for focal loss')
    train_parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    train_parser.add_argument('--use_noise', action='store_true', help='Use noise injection for diversity')
    train_parser.add_argument('--noise_dim', type=int, default=16, help='Dimension of the noise vector')
    train_parser.add_argument('--fragmentation_kernel_size', type=int, default=5, help='Size of kernel for fragmentation loss')
    train_parser.add_argument('--perlin_octaves', type=int, default=4, help='Number of octaves for Perlin noise')
    train_parser.add_argument('--perlin_persistence', type=float, default=0.5, help='Persistence parameter for Perlin noise')
    train_parser.add_argument('--perlin_lacunarity', type=float, default=2.0, help='Lacunarity parameter for Perlin noise')
    train_parser.add_argument('--perlin_scale', type=float, default=0.1, help='Scale parameter for Perlin noise')
    train_parser.add_argument('--use_learnable_noise', action='store_true', help='Use learnable Perlin noise parameters')
    
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
    generate_parser.add_argument('--training_lesion_dir', type=str, help='Directory with training lesions for coverage calculation')
    generate_parser.add_argument('--target_coverage', type=float, help='Target lesion coverage percentage')
    generate_parser.add_argument('--min_adaptive_threshold', type=float, default=0.00001, help='Minimum threshold for adaptive thresholding')
    generate_parser.add_argument('--max_adaptive_threshold', type=float, default=0.999, help='Maximum threshold for adaptive thresholding')
    generate_parser.add_argument('--adaptive_threshold_iterations', type=int, default=50, help='Maximum iterations for adaptive threshold search')
    generate_parser.add_argument('--use_different_target_for_each_sample', action='store_true', help='Use different target coverage for each sample')
    generate_parser.add_argument('--use_learnable_noise', action='store_true', help='Use learnable Perlin noise parameters')
    
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
            use_different_target_for_each_sample=args.use_different_target_for_each_sample
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
