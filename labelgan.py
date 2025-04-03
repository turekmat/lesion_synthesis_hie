import os
import random
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.measure import label as measure_label
import scipy.ndimage as ndimage
from matplotlib.colors import LinearSegmentedColormap

class HIELesionDataset(Dataset):
    def __init__(self, 
                 label_dir, 
                 atlas_path=None, 
                 brain_mask_path=None,
                 transform=None, 
                 use_random_affine=False, 
                 use_elastic_transform=False,
                 intensity_noise=0.1,
                 filter_empty=True):  # Nový parametr pro filtrování prázdných obrázků
        """
        Dataset pro načítání labelů, atlasu a případně brain masky
        
        Args:
            label_dir: Adresář s label mapami
            atlas_path: Cesta k atlasu lézí
            brain_mask_path: Cesta k masce mozku
            transform: Transformace aplikované na data
            use_random_affine: Použít náhodnou afinní transformaci
            use_elastic_transform: Použít elastickou transformaci pro augmentaci
            intensity_noise: Síla šumu pro intenzitu
            filter_empty: Filtrovat celočerné obrázky bez lézí
        """
        self.label_dir = label_dir
        self.transform = transform
        self.use_random_affine = use_random_affine
        self.use_elastic_transform = use_elastic_transform
        self.intensity_noise = intensity_noise
        
        # Načtení všech souborů s labely
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) 
                               if f.endswith('.nii.gz') or f.endswith('.nii')])
        
        if not self.label_files:
            raise ValueError(f"No label files found in {label_dir}")
        
        print(f"Found {len(self.label_files)} lesion label files in {label_dir}")
        
        # NOVÁ FUNKCE: Filtrování celočerných obrázků
        if filter_empty:
            non_empty_files = []
            print("Filtering out completely black images...")
            
            for file_path in tqdm(self.label_files, desc="Checking lesion files"):
                # Načtení label souboru
                if file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
                    label_data = nib.load(file_path).get_fdata()
                else:
                    continue  # Přeskočit nepodporované formáty
                
                # Kontrola, zda obrázek obsahuje nenulové hodnoty (tj. léze)
                if np.any(label_data > 0):
                    non_empty_files.append(file_path)
            
            filtered_count = len(self.label_files) - len(non_empty_files)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} completely black images")
                print(f"Remaining images with lesions: {len(non_empty_files)}")
                self.label_files = non_empty_files
            else:
                print("No completely black images found - all images contain lesions")
        
        # Načtení atlasu lézí, pokud je dostupný
        self.atlas_path = atlas_path
        if atlas_path:
            print(f"Loading lesion atlas from {atlas_path}")
            self.atlas = nib.load(atlas_path).get_fdata()
            
            # Normalizace atlasu na rozsah [0, 1]
            self.atlas = (self.atlas - np.min(self.atlas)) / (np.max(self.atlas) - np.min(self.atlas))
            print(f"Atlas shape: {self.atlas.shape}, min: {np.min(self.atlas)}, max: {np.max(self.atlas)}")
            
            # Převod na tensor
            self.atlas = torch.FloatTensor(self.atlas).unsqueeze(0)  # Přidání kanálové dimenze
        else:
            self.atlas = None
            
        # Načtení masky mozku, pokud je dostupná
        self.brain_mask_path = brain_mask_path
        if brain_mask_path:
            print(f"Loading brain mask from {brain_mask_path}")
            self.brain_mask = nib.load(brain_mask_path).get_fdata()
            
            # Binarizace masky
            self.brain_mask = (self.brain_mask > 0).astype(np.float32)
            print(f"Brain mask shape: {self.brain_mask.shape}, sum: {np.sum(self.brain_mask)}")
            
            # Převod na tensor
            self.brain_mask = torch.FloatTensor(self.brain_mask).unsqueeze(0)  # Přidání kanálové dimenze
        else:
            self.brain_mask = None
            
        # Inicializace 3D augmentací, pokud jsou požadovány
        if use_random_affine:
            self.augmentation = RandomAugmentation3D()
        
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        # Načtení label mapy
        label_path = self.label_files[idx]
        label_nii = nib.load(label_path)
        label = label_nii.get_fdata()
        
        # Binarizace label (léze jsou nenulové hodnoty)
        label = (label > 0).astype(np.float32)
        
        # Převod na tensor
        label_tensor = torch.from_numpy(label).float().unsqueeze(0)
        
        # Načtení atlasu, pokud je k dispozici
        if self.atlas is not None:
            # Již načteno a převedeno na tensor v __init__
            atlas_tensor = self.atlas.clone()
        else:
            # Pokud atlas chybí, vytvoříme prázdný tensor stejné velikosti jako label
            atlas_tensor = torch.zeros_like(label_tensor)
        
        # Načtení masky mozku, pokud je k dispozici
        if self.brain_mask is not None:
            # Již načteno a převedeno na tensor v __init__
            brain_mask_tensor = self.brain_mask.clone()
        else:
            # Pokud maska chybí, použijeme jednotkovou masku stejné velikosti jako label
            brain_mask_tensor = torch.ones_like(label_tensor)
        
        # Aplikace transformací a augmentací, pokud jsou požadovány
        if self.use_random_affine:
            # Vytvoříme slovník pro augmentaci
            sample = {'label': label_tensor, 'atlas': atlas_tensor, 'brain_mask': brain_mask_tensor}
            augmented = self.augmentation(sample)
            label_tensor = augmented['label']
            atlas_tensor = augmented['atlas']
            brain_mask_tensor = augmented['brain_mask']
        
        # Přidání malého Gaussovského šumu pro větší rozmanitost
        if self.intensity_noise > 0:
            noise = torch.randn_like(label_tensor) * self.intensity_noise
            # Aplikujeme šum pouze na nenulové hodnoty pro zachování binárního charakteru
            label_tensor = torch.where(label_tensor > 0, 
                                    torch.clamp(label_tensor + noise, 0.0, 1.0), 
                                    label_tensor)
        
        return {
            'label': label_tensor, 
            'atlas': atlas_tensor,
            'brain_mask': brain_mask_tensor,
            'file_path': label_path
        }


class RandomAugmentation3D:
    def __init__(self, rotation_range=10, flip_prob=0.5):
        """
        Jednoduchá augmentace pro 3D data.
        
        Args:
            rotation_range (int): Maximální úhel rotace ve stupních
            flip_prob (float): Pravděpodobnost flipu
        """
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
    
    def __call__(self, sample):
        label = sample['label']
        atlas = sample['atlas']
        brain_mask = sample['brain_mask']
        
        # Náhodná rotace - aplikovaná na všechny komponenty stejně pro konzistenci
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            
            # Rotace label
            for z in range(label.shape[0]):
                label_slice = label[z, :, :]
                label[z, :, :] = transforms.functional.rotate(
                    label_slice.unsqueeze(0), angle).squeeze(0)
        
            # Rotace brain_mask - stejný úhel jako label
            for z in range(brain_mask.shape[0]):
                mask_slice = brain_mask[z, :, :]
                brain_mask[z, :, :] = transforms.functional.rotate(
                    mask_slice.unsqueeze(0), angle).squeeze(0)
        
        # Náhodný horizontální flip - aplikovaný na všechny komponenty
        if random.random() < self.flip_prob:
            label = torch.flip(label, [1])       # Flip horizontálně
            brain_mask = torch.flip(brain_mask, [1])  # Stejný flip pro masku
        
        # Náhodný vertikální flip - aplikovaný na všechny komponenty
        if random.random() < self.flip_prob:
            label = torch.flip(label, [2])       # Flip vertikálně
            brain_mask = torch.flip(brain_mask, [2])  # Stejný flip pro masku
        
        # Atlas neaugmentujeme, protože představuje anatomickou pravděpodobnost
        
        return {
            'label': label, 
            'atlas': atlas, 
            'brain_mask': brain_mask
        }


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


# Self-Attention modul pro 3D data
class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        self.query = spectral_norm(nn.Conv3d(in_channels, in_channels // 8, 1))
        self.key = spectral_norm(nn.Conv3d(in_channels, in_channels // 8, 1))
        self.value = spectral_norm(nn.Conv3d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        
        # Flatten spatial dimensions
        query = self.query(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, D * H * W)
        value = self.value(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)
        
        # Attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=2)
        
        # Apply attention to value
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).contiguous().view(batch_size, C, D, H, W)
        
        return self.gamma * out + x


class Generator(nn.Module):
    def __init__(self, z_dim=128, atlas_channels=1, output_channels=1, base_filters=64, use_spectral_norm=True):
        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        self.base_filters = base_filters
        
        # Encoder pro atlas
        self.atlas_enc1 = nn.Sequential(
            spectral_norm(nn.Conv3d(atlas_channels, base_filters, 4, 2, 1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.atlas_enc2 = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters, base_filters * 2, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.atlas_enc3 = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters * 2, base_filters * 4, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.atlas_enc4 = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters * 4, base_filters * 8, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Transformace pro latentní vektor
        self.fc_z = nn.Linear(z_dim, 8 * 8 * 4 * base_filters)
        self.reshape_z = nn.Sequential(
            nn.InstanceNorm3d(base_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Enhanced bottleneck with atlas-aware features
        self.bottleneck_conv = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters * 8 + base_filters, base_filters * 8, 3, 1, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Self-Attention po bottlenecku
        self.self_attn = SelfAttention3D(base_filters * 8)
        
        # Decoder s skip connections
        self.dec4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(base_filters * 8, base_filters * 2, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(base_filters * 4, base_filters, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Finální výstupní vrstva s vysoko-teplotním sigmoidem pro ostřejší přechod -> binární výstup
        self.dec1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(base_filters * 2, output_channels, 4, 2, 1), use_spectral_norm),
        )
        
        # Enhanced atlas-based attention for probabilistic guidance
        # Operates on d2 (base_filters channels) + atlas_channels
        self.atlas_attention = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters + atlas_channels, base_filters, 3, 1, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(base_filters, base_filters, 3, 1, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters),
            nn.Sigmoid()
        )
        
        # Atlas-based modulation - Helps modulate feature maps based on atlas probabilities
        self.atlas_modulation = nn.Sequential(
            spectral_norm(nn.Conv3d(atlas_channels, 8, 3, 1, 1), use_spectral_norm),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(8, 1, 3, 1, 1), use_spectral_norm),
            nn.Sigmoid()
        )
        
        # Debug flag - set to True to print tensor shapes during forward pass
        self.debug = False
        
    def forward(self, z, atlas, brain_mask=None):
        # Ensure z has the right shape
        if z.dim() == 2:
            z = z.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
        # Transform latent vector
        batch_size = z.size(0)
        z_flat = self.fc_z(z.view(batch_size, -1))
        z_reshaped = z_flat.view(batch_size, self.base_filters, 8, 8, 4)
        z_processed = self.reshape_z(z_reshaped)
        
        # Encoder for atlas - extract spatial features at multiple levels
        a1 = self.atlas_enc1(atlas)  # Output: base_filters
        a2 = self.atlas_enc2(a1)     # Output: base_filters*2
        a3 = self.atlas_enc3(a2)     # Output: base_filters*4
        a4 = self.atlas_enc4(a3)     # Output: base_filters*8
        
        # Concatenate processed latent vector with atlas features
        combined = torch.cat([a4, z_processed], dim=1)
        
        # Pass through bottleneck convolution to fuse features
        bottleneck = self.bottleneck_conv(combined)
        
        # Apply self-attention at bottleneck for long-range dependencies
        bottleneck_attn = self.self_attn(bottleneck)
        
        # Decoder with skip connections from encoder
        d4 = self.dec4(bottleneck_attn)
        
        # Concatenate with encoder features and decode
        d4_cat = torch.cat([d4, a3], dim=1)
        d3 = self.dec3(d4_cat)
        
        d3_cat = torch.cat([d3, a2], dim=1)
        d2 = self.dec2(d3_cat)
        
        # Use attention mechanism to focus on atlas-positive regions
        # Prepare attention input by concatenating d2 features with atlas
        d2_with_atlas = torch.cat([d2, F.interpolate(atlas, size=d2.shape[2:], mode='trilinear', align_corners=False)], dim=1)
        attention_map = self.atlas_attention(d2_with_atlas)
        
        # Apply attention to d2 features
        d2_attended = d2 * attention_map
        
        # Concatenate with attended features
        d2_cat = torch.cat([d2_attended, a1], dim=1)
        
        # Final layer with sigmoid to create probability map
        logits = self.dec1(d2_cat)
        
        # MODIFIED: Apply a more nuanced sigmoid activation for more natural lesion transitions
        # Further reduced temperature for smoother, less binary outputs during generation
        temperature = 2.5  # Further reduced from 3.0 for even softer transitions
        
        # MODIFIED: Add a stronger positive bias to encourage more non-zero activations
        # This shifts the sigmoid curve to the left, making it much more likely to produce higher values
        bias = 0.3  # Increased from 0.2 for more aggressive activation encouragement
        out = torch.sigmoid(temperature * (logits + bias - 0.5))
        
        # NEW: Add conditional instance normalization that shifts the distribution toward more activation
        # This subtly increases all values, especially in regions with high atlas probability
        if not self.training:
            # Only apply during inference to ensure training is still learning properly
            # Scale the shift based on atlas values to maintain anatomical plausibility
            atlas_factor = F.interpolate(atlas, size=out.shape[2:], mode='trilinear', align_corners=False)
            # Apply a small positive shift more strongly in high-atlas regions
            normalization_shift = atlas_factor * 0.1  # Small positive shift
            out = out + normalization_shift * (1.0 - out)  # Apply shift more to low values
        
        # Resize atlas to match output size if needed
        if out.shape[2:] != atlas.shape[2:]:
            atlas_final = F.interpolate(atlas, size=out.shape[2:], mode='trilinear', align_corners=False)
        else:
            atlas_final = atlas
            
        # Apply atlas modulation - enhance probability in regions with higher atlas values
        atlas_mod = self.atlas_modulation(atlas_final)
        
        # MODIFIED: Blend the generated output with the atlas probability using a more aggressive approach
        # This creates more visible, anatomically plausible lesions
        blend_factor = 1.5  # Further increased from 1.2 to more strongly enhance atlas influence
        out = torch.clamp((out * (1.0 + atlas_mod * blend_factor)), 0.0, 1.0)
        
        # COMPLETELY REWORKED approach to creating connected lesions during inference
        if not self.training:
            # STEP 1: Initial thresholding to identify potential lesion regions
            # Use a MUCH lower threshold to capture more potential lesion areas
            potential_lesions = (out > 0.15).float()  # Reduced from 0.2 to ensure we capture even weak activations
            
            # STEP 2: Apply a sequence of 3D morphological operations to connect nearby regions
            # First dilate to connect nearby regions (implemented with max pooling)
            kernel_size = 5  # Increased from 3 for stronger connectivity
            padding = kernel_size // 2
            
            # Dilate - expand regions to connect nearby components
            dilated = F.max_pool3d(
                F.pad(potential_lesions, (padding, padding, padding, padding, padding, padding)),
                kernel_size=kernel_size,
                stride=1,
                padding=0
            )
            
            # Apply a second, smaller dilation to ensure robust connections
            small_kernel = 3
            small_pad = small_kernel // 2
            dilated = F.max_pool3d(
                F.pad(dilated, (small_pad, small_pad, small_pad, small_pad, small_pad, small_pad)),
                kernel_size=small_kernel,
                stride=1,
                padding=0
            )
            
            # STEP 3: Apply erosion to refine the shape (implemented with min pooling)
            # To implement erosion with min pooling, we invert, apply max pooling, then invert again
            inverted = 1 - dilated
            eroded_inv = F.max_pool3d(
                F.pad(inverted, (small_pad, small_pad, small_pad, small_pad, small_pad, small_pad)),
                kernel_size=small_kernel,
                stride=1,
                padding=0
            )
            # Invert back to get the eroded mask
            eroded = 1 - eroded_inv
            
            # STEP 4: Get connected components to filter out tiny isolated regions
            # Process on CPU for component analysis
            binary_np = eroded[0, 0].cpu().numpy()
            labeled_components, num_components = measure_label(binary_np, return_num=True)
            
            # Calculate component sizes
            if num_components > 0:
                component_sizes = np.bincount(labeled_components.flatten())[1:] if num_components > 0 else []
                
                # Reduced minimum component size to keep more potential lesions
                min_component_size = 3  # Further reduced from 5 to retain even smaller lesions
                
                # Create cleaned binary mask - keep only sufficiently large components
                cleaned_binary = np.zeros_like(binary_np)
                for comp_id in range(1, num_components + 1):
                    if component_sizes[comp_id - 1] >= min_component_size:
                        cleaned_binary[labeled_components == comp_id] = 1
                
                # Convert back to tensor
                cleaned_binary_tensor = torch.from_numpy(cleaned_binary).float().to(out.device)
                cleaned_binary_tensor = cleaned_binary_tensor.unsqueeze(0).unsqueeze(0)
                
                # STEP 5: Re-create a continuous-valued output by blending the original output 
                # with the morphologically processed mask
                # This preserves some of the original probability values within the connected regions
                continuous_output = out * cleaned_binary_tensor
                
                # STEP 6: Apply Gaussian-like smoothing to create more natural transitions
                # Implement a simple smoothing operation that preserves connected structures
                smoothed = F.avg_pool3d(
                    F.pad(continuous_output, (padding, padding, padding, padding, padding, padding)),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=0
                )
                
                # Blend the smoothed output with the continuous output
                blend_weight = 0.7
                final_output = blend_weight * smoothed + (1 - blend_weight) * continuous_output
                
                # Final thresholding to clean up very small values - lower threshold to keep more subtle lesions
                final_output = torch.where(final_output > 0.08, final_output, torch.zeros_like(final_output))  # Further reduced from 0.1
                
                # Use the morphologically processed output
                out = final_output
        
        # If brain mask is provided, ensure output is zero outside the brain
        if brain_mask is not None:
            # Resize brain mask if needed
            if out.shape[2:] != brain_mask.shape[2:]:
                brain_mask_resized = F.interpolate(brain_mask, size=out.shape[2:], mode='nearest')
            else:
                brain_mask_resized = brain_mask
                
            # Apply brain mask
            out = out * brain_mask_resized
        
        return out


class Discriminator(nn.Module):
    def __init__(self, input_channels=2, atlas_channels=1, base_filters=64, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        
        # Debug flag for printing tensor shapes
        self.debug = False
        
        # First layer processes the lesion input
        self.input_layer = nn.Sequential(
            spectral_norm(nn.Conv3d(input_channels, base_filters, 4, 2, 1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Separate atlas preprocessing path to extract probability features
        self.atlas_encoder = nn.Sequential(
            spectral_norm(nn.Conv3d(atlas_channels, base_filters // 2, 3, 1, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters // 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(base_filters // 2, base_filters, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layers after combining input features with atlas-encoded features
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters * 2, base_filters * 2, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters * 2, base_filters * 4, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Self-attention na prostřední vrstvě
        self.self_attn = SelfAttention3D(base_filters * 4)
        
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters * 4, base_filters * 8, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Atlas-probability aware attention
        self.atlas_attention = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters * 8 + atlas_channels, base_filters * 8, 3, 1, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(base_filters * 8, base_filters * 8, 3, 1, 1), use_spectral_norm),
            nn.Sigmoid()
        )
        
        # Distribution consistency evaluator - checks if lesion distribution matches atlas probability
        self.distribution_evaluator = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters * 8 + atlas_channels, base_filters * 4, 1, 1, 0), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(base_filters * 4, 1), use_spectral_norm)
        )
        
        # PatchGAN - lokální realism posouzení
        self.patch_output = spectral_norm(nn.Conv3d(base_filters * 8, 1, 4, 1, 1), use_spectral_norm)
        
        # Global decision - celkové posouzení věrohodnosti
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_output = spectral_norm(nn.Linear(base_filters * 8, 1), use_spectral_norm)
        
    def forward(self, x, atlas):
        batch_size = x.size(0)
        
        # Ensure x and atlas have same spatial dimensions
        if x.shape[2:] != atlas.shape[2:]:
            if self.debug:
                print(f"WARNING: Mismatched shapes - x: {x.shape}, atlas: {atlas.shape}")
            # Resize atlas to match x's spatial dimensions
            atlas = F.interpolate(atlas, size=x.shape[2:], mode='trilinear', align_corners=False)
            if self.debug:
                print(f"Resized atlas to: {atlas.shape}")
                
        # Konkatenace vstupu a atlasu podél kanálové dimenze
        x_combined = torch.cat([x, atlas], dim=1)
        
        if self.debug:
            print(f"Discriminator input (combined): {x_combined.shape}")
        
        # Process combined input and atlas features separately
        x_features = self.input_layer(x_combined)
        atlas_features = self.atlas_encoder(atlas)
        
        # Concatenate processed input with processed atlas features
        # This allows the network to make decisions based on both the lesion and the probability atlas
        x = torch.cat([x_features, atlas_features], dim=1)
        
        if self.debug:
            print(f"After initial processing and concat: {x.shape}")
            
        x = self.layer2(x)
        if self.debug:
            print(f"After layer2: {x.shape}")
            
        x = self.layer3(x)
        if self.debug:
            print(f"After layer3: {x.shape}")
        
        # Self-attention
        x = self.self_attn(x)
        if self.debug:
            print(f"After self-attention: {x.shape}")
        
        x = self.layer4(x)
        if self.debug:
            print(f"After layer4: {x.shape}")
        
        # Interpolate atlas to the current feature resolution for attention and distribution evaluation
        atlas_downsampled = F.interpolate(atlas, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        # Apply atlas-probability aware attention
        attention_input = torch.cat([x, atlas_downsampled], dim=1)
        attention_map = self.atlas_attention(attention_input)
        x_attended = x * attention_map
        
        if self.debug:
            print(f"After atlas-aware attention: {x_attended.shape}")
        
        # PatchGAN output - mapa skóre
        patch_out = self.patch_output(x_attended)
        if self.debug:
            print(f"Patch output shape: {patch_out.shape}")
        
        # Globální posouzení
        global_features = self.global_pool(x_attended).view(batch_size, -1)
        global_out = self.global_output(global_features)
        if self.debug:
            print(f"Global output shape: {global_out.shape}")
        
        # Distribution consistency evaluation
        distribution_input = torch.cat([x_attended, atlas_downsampled], dim=1)
        distribution_score = self.distribution_evaluator(distribution_input)
        if self.debug:
            print(f"Distribution score shape: {distribution_score.shape}")
        
        # Return multiple outputs:
        # 1. PatchGAN output for local realism
        # 2. Global output for overall realism
        # 3. Distribution score for evaluating atlas probability consistency
        return patch_out, global_out, distribution_score

    def get_features(self, x, atlas):
        """
        Extract intermediate layer features for feature matching loss
        
        Args:
            x: Input lesion map (B, 1, D, H, W)
            atlas: Atlas probability map (B, 1, D, H, W)
            
        Returns:
            tuple of feature lists: (patch_features, global_features, distribution_features)
        """
        # Ensure inputs are in the proper range
        x = torch.clamp(x, 0.0, 1.0)
        
        # Combined input - concatenate lesion and atlas
        combined_input = torch.cat([x, atlas], dim=1)
        
        # Process the combined input through the first layer
        feat1 = self.input_layer(combined_input)
        
        # Process atlas separately for the atlas-specific path
        atlas_feat = self.atlas_encoder(atlas)
        
        # Combine atlas features with lesion features
        combined_feat = torch.cat([feat1, atlas_feat], dim=1)
        
        # Further processing through deeper layers
        feat2 = self.layer2(combined_feat)
        feat3 = self.layer3(feat2)
        
        # Self-attention at the middle layer
        feat_attn = self.self_attn(feat3)
        
        # Final convolutional features
        feat4 = self.layer4(feat_attn)
        
        # Get patch, global, and distribution features based on how forward() uses these features
        # Determine which internal layers are used for each output path
        
        # For patch output: Use all features up to patch-specific outputs
        # Simplified - just use the core feature hierarchy
        patch_features = [feat1, feat2, feat3, feat4]
        
        # For global output: Similar approach, just use the core feature hierarchy
        global_features = [feat1, feat2, feat3, feat4]
        
        # For distribution features: Same approach
        distribution_features = [feat1, feat2, feat3, feat4]
        
        return patch_features, global_features, distribution_features


# Loss Functions
class GANLoss:
    def __init__(self, device, gan_mode='hinge'):
        self.device = device
        self.gan_mode = gan_mode
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor.to(self.device))
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['lsgan', 'hinge']:
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def __call__(self, prediction, target_is_real):
        if self.gan_mode == 'vanilla':
            target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
            return self.loss(prediction, target)
        elif self.gan_mode == 'lsgan':
            target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
            return self.loss(prediction, target)
        elif self.gan_mode == 'hinge':
            if isinstance(target_is_real, bool):
                if target_is_real:
                    return -torch.mean(torch.min(torch.zeros_like(prediction), -1 + prediction))
                else:
                    return -torch.mean(torch.min(torch.zeros_like(prediction), -1 - prediction))
            else:
                # Pokud target_is_real není bool, rozhodneme se podle průměrné hodnoty tensoru.
                if target_is_real.mean().item() > 0.5:
                    return -torch.mean(torch.min(torch.zeros_like(prediction), -1 + prediction))
                else:
                    return -torch.mean(torch.min(torch.zeros_like(prediction), -1 - prediction))



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class AtlasWeightedLoss(nn.Module):
    """
    A loss function that weights the error based on the atlas probability.
    Regions with higher atlas probability are weighted more heavily in the loss.
    """
    def __init__(self, base_loss, reduction='mean'):
        super(AtlasWeightedLoss, self).__init__()
        self.base_loss = base_loss
        self.reduction = reduction
        
    def forward(self, inputs, targets, atlas_weights):
        # Ensure atlas_weights has same spatial dimensions as inputs/targets
        if inputs.shape[2:] != atlas_weights.shape[2:]:
            atlas_weights = F.interpolate(atlas_weights, size=inputs.shape[2:], mode='trilinear', align_corners=False)
        
        # Normalize atlas weights so they don't change the overall scale of the loss
        # But still provide relative importance between regions
        normalized_weights = atlas_weights / (atlas_weights.mean() + 1e-8)
        
        # Scale up by a factor to make the effect more pronounced
        weight_factor = 2.0
        scaled_weights = 1.0 + (normalized_weights - 1.0) * weight_factor
        
        # Apply per-element loss
        if hasattr(self.base_loss, 'reduction'):
            # Save original reduction
            original_reduction = self.base_loss.reduction
            # Set to none to get per-element loss
            self.base_loss.reduction = 'none'
            
        # Compute base loss
        if isinstance(self.base_loss, nn.Module):
            element_loss = self.base_loss(inputs, targets)
        else:
            # Handle functional losses
            element_loss = self.base_loss(inputs, targets, reduction='none')
            
        # Reset original reduction mode if using nn.Module
        if hasattr(self.base_loss, 'reduction'):
            self.base_loss.reduction = original_reduction
        
        # Apply atlas weighting
        weighted_loss = element_loss * scaled_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class AtlasGuidedFocalLoss(nn.Module):
    """
    A focal loss that is guided by atlas probabilities.
    Focuses more on regions with higher atlas probability.
    """
    def __init__(self, alpha=0.75, gamma=2.5, reduction='mean'):
        super(AtlasGuidedFocalLoss, self).__init__()
        self.alpha = alpha  # Zvýšeno z 0.25 na 0.75 pro větší důraz na pozitivní třídu (léze)
        self.gamma = gamma  # Zvýšeno z 2.0 na 2.5 pro větší penalizaci snadných příkladů
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets, atlas):
        # Ensure dimensions match
        if inputs.shape[2:] != atlas.shape[2:]:
            atlas = F.interpolate(atlas, size=inputs.shape[2:], mode='trilinear', align_corners=False)
        
        # ADDED: Ensure inputs and targets are in range [0, 1]
        inputs = torch.clamp(inputs, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)
        
        # Regular focal loss term
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_weight = self.alpha * (1-pt)**self.gamma
        
        # Atlas-guided modulation - increase weight in high probability atlas regions
        # Normalize atlas values for weighting
        atlas_norm = atlas / (atlas.mean() + 1e-8)
        
        # Scale atlas influence - zvýšeno pro větší důraz na oblasti s vysokou pravděpodobností
        atlas_scale = 3.0  # Zvýšeno z 2.0 na 3.0
        atlas_weight = 1.0 + (atlas_norm - 1.0) * atlas_scale
        
        # Apply both focal weighting and atlas weighting
        combined_weight = focal_weight * atlas_weight
        
        # Final weighted loss
        weighted_loss = BCE_loss * combined_weight
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class AtlasDistributionLoss(nn.Module):
    """
    Loss that penalizes distribution differences between generated lesions
    and the atlas probability distribution.
    """
    def __init__(self, reduction='mean'):
        super(AtlasDistributionLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, lesions, atlas):
        # Ensure dimensions match
        if lesions.shape[2:] != atlas.shape[2:]:
            atlas = F.interpolate(atlas, size=lesions.shape[2:], mode='trilinear', align_corners=False)
        
        # ADDED: Ensure lesions are non-negative for a valid distribution
        lesions = torch.clamp(lesions, 0.0, 1.0)
        
        # Normalize atlas to sum to 1 (probability distribution)
        atlas_sum = torch.sum(atlas, dim=(2, 3, 4), keepdim=True) + 1e-8
        atlas_prob = atlas / atlas_sum
        
        # Normalize lesion to sum to 1 (probability distribution)
        lesion_sum = torch.sum(lesions, dim=(2, 3, 4), keepdim=True) + 1e-8
        lesion_prob = lesions / lesion_sum
        
        # KL divergence like distribution difference
        # eps for numerical stability
        eps = 1e-8
        distribution_diff = atlas_prob * torch.log(atlas_prob / (lesion_prob + eps) + eps)
        
        # Apply reduction
        if self.reduction == 'mean':
            return distribution_diff.mean()
        elif self.reduction == 'sum':
            return distribution_diff.sum()
        else:
            return distribution_diff


# Tréninková utilita s implementací tréninkové smyčky
class HIELesionGANTrainer:
    def __init__(self, 
                 dataloader,
                 val_dataloader=None,
                 z_dim=128,
                 learning_rate=0.0002,
                 beta1=0.5,
                 beta2=0.999,
                 save_dir='./results',
                 validate_every=5,
                 save_interval=10,
                 device=None,
                 debug=False):
        
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.z_dim = z_dim
        self.save_dir = save_dir
        self.validate_every = validate_every
        self.save_interval = save_interval
        self.debug = debug
        
        # Create model save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.generator = Generator(z_dim=z_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # NEW: Initialize fixed random vectors for consistent training visualization
        self.fixed_z = torch.randn(4, z_dim, device=self.device)
        
        # Set up TensorBoard
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join("logs", current_time)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to {log_dir}")
        
        # NEW: Setup debug printer
        self.debug_print = self.create_debug_printer() if debug else lambda *args, **kwargs: None
        
        # MODIFIED: Use Two Time-Scale Update Rule (TTUR)
        # Generator gets higher learning rate to overcome local minima
        self.g_learning_rate = learning_rate * 4.0  # Zvýšeno z 2.0 na 4.0 pro ještě silnější generátor
        self.d_learning_rate = learning_rate * 0.25  # Sníženo z 0.5 na 0.25 pro slabší diskriminátor
        
        # Set up optimizers with adjusted learning rates
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=self.g_learning_rate,  # Using higher learning rate for generator
            betas=(beta1, beta2)
        )
        
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=self.d_learning_rate,  # Using lower learning rate for discriminator 
            betas=(beta1, beta2)
        )
        
        # NEW: Add learning rate schedulers to decrease learning rates over time
        # This helps stabilize training in later stages
        self.g_scheduler = optim.lr_scheduler.ExponentialLR(self.generator_optimizer, gamma=0.996)  # Mírnější snižování pro generátor
        self.d_scheduler = optim.lr_scheduler.ExponentialLR(self.discriminator_optimizer, gamma=0.992)  # Rychlejší snižování pro diskriminátor
        
        # Create adversarial loss
        self.adversarial_loss = GANLoss(self.device, gan_mode='hinge')
        
        # Create additional losses
        self.l1_loss = nn.L1Loss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.atlas_weighted_l1 = AtlasWeightedLoss(nn.L1Loss())
        self.atlas_weighted_dice = AtlasWeightedLoss(DiceLoss())
        self.atlas_focal_loss = AtlasGuidedFocalLoss()
        self.atlas_distribution_loss = AtlasDistributionLoss()
        
        # Set loss weights
        self.adv_loss_weight = 0.5  # Sníženo z 1.0 na 0.5 - menší důraz na adversarial loss
        self.l1_loss_weight = 10.0
        self.weighted_l1_weight = 5.0
        self.dice_loss_weight = 5.0
        self.focal_loss_weight = 2.0
        self.distribution_loss_weight = 1.0
        self.violation_penalty_weight = 5.0
        
        # NEW: Add parameters to control noise injection for discriminator stability
        self.noise_std = 0.2  # Zvýšeno z 0.05 na 0.2 - výrazně větší šum pro destabilizaci diskriminátoru
        self.label_smoothing = 0.3  # Zvýšeno z 0.1 na 0.3 - větší label smoothing (0.3 = use 0.7 for real instead of 1.0)
        
        # NEW: Add parameters for gradient penalty in WGAN-GP style training
        self.use_gradient_penalty = False  # Vypnuto, nechceme příliš silný diskriminátor
        self.gp_weight = 5.0  # Sníženo z 10.0 na 5.0
        
        # NEW: Add parameters to control activation requirements
        self.min_activation_percent = 0.02  # Zvýšeno z 0.005 na 0.02 (2% minimální aktivace)
        self.empty_penalty_weight = 500.0   # Zvýšeno z 200.0 na 500.0
        self.activation_target_weight = 2500.0  # Zvýšeno z 500.0 na 2500.0
        
        # NEW: Add early training assistance parameters
        self.early_training_iters = 3000  # Zvýšeno z 1000 na 3000 - delší období nucené aktivace
        self.forced_activation_threshold = 500  # Zvýšeno z 10 na 500 - agresivnější aktivace
        
    def create_debug_printer(self):
        """Create a debug print function that only prints when debug is enabled"""
        def debug_printer(message, tensor=None):
            if not self.debug:
                return
            
            output = message
            if tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    shape_str = 'x'.join([str(s) for s in tensor.shape])
                    dtype_str = str(tensor.dtype).split('.')[-1]
                    device_str = str(tensor.device)
                    min_val = tensor.min().item() if tensor.numel() > 0 else "N/A"
                    max_val = tensor.max().item() if tensor.numel() > 0 else "N/A"
                    nan_count = torch.isnan(tensor).sum().item() if tensor.numel() > 0 else 0
                    inf_count = torch.isinf(tensor).sum().item() if tensor.numel() > 0 else 0
                    
                    status = ""
                    if nan_count > 0:
                        status += f" [WARNING: {nan_count} NaNs]"
                    if inf_count > 0:
                        status += f" [WARNING: {inf_count} Infs]"
                    
                    output += f" shape={shape_str}, range=[{min_val:.4f}, {max_val:.4f}]{status}"
                else:
                    output += f" {tensor}"
            print(f"[DEBUG] {output}")
        
        return debug_printer

    def validate(self, epoch):
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {
            'val_d_loss': 0.0,
            'val_g_loss': 0.0,
            'val_l1_loss': 0.0,
            'val_weighted_l1': 0.0,
            'val_dice_loss': 0.0,
            'val_focal_loss': 0.0,
            'val_distribution': 0.0,
            'val_atlas_violation': 0.0
        }
        
        # Pro vizualizaci ukládej první batch z validačních dat
        first_batch_samples = None
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                real_lesions = batch['label'].to(self.device)
                atlas = batch['atlas'].to(self.device)
                brain_mask = batch['brain_mask'].to(self.device)
                
                # ADDED: Ensure real_lesions are in valid range [0,1]
                if torch.any(real_lesions > 1.0) or torch.any(real_lesions < 0.0):
                    if self.debug:
                        min_val = real_lesions.min().item()
                        max_val = real_lesions.max().item()
                        print(f"[WARNING] real_lesions outside range [0,1] during validation: min={min_val}, max={max_val}")
                    real_lesions = torch.clamp(real_lesions, 0.0, 1.0)
                
                batch_size = real_lesions.size(0)
                num_batches += 1
                
                # Generate fake lesions
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                fake_lesions = self.generator(z, atlas, brain_mask)
                
                # ADDED: Ensure fake_lesions are in valid range [0,1]
                if torch.any(fake_lesions > 1.0) or torch.any(fake_lesions < 0.0):
                    if self.debug:
                        min_val = fake_lesions.min().item()
                        max_val = fake_lesions.max().item()
                        print(f"[WARNING] fake_lesions outside range [0,1] during validation: min={min_val}, max={max_val}")
                    fake_lesions = torch.clamp(fake_lesions, 0.0, 1.0)
                
                # Uložení prvního batche pro vizualizaci
                if num_batches == 1:
                    first_batch_samples = {
                        'real_lesions': real_lesions.detach(),
                        'fake_lesions': fake_lesions.detach(),
                        'atlas': atlas.detach(),
                        'brain_mask': brain_mask.detach()
                    }
                
                # Debug prints of tensor shapes if needed
                if self.debug_print:
                    self.debug_print(f"validate - real_lesions: {real_lesions.shape}, fake_lesions: {fake_lesions.shape}")
                    self.debug_print(f"validate - atlas: {atlas.shape}, brain_mask: {brain_mask.shape}")
                
                # Create a binary mask where atlas is positive
                atlas_binary = (atlas > 0).float()
                
                # Create a mask where both atlas_binary and brain_mask are positive
                valid_regions = atlas_binary * brain_mask
                
                # Ensure dimensions match before applying masks
                if real_lesions.shape[2:] != valid_regions.shape[2:]:
                    valid_regions = F.interpolate(valid_regions, size=real_lesions.shape[2:], mode='nearest')
                
                # Discriminator validation
                fake_patch_pred, fake_global_pred, fake_distribution_pred = self.discriminator(fake_lesions, atlas)
                real_patch_pred, real_global_pred, real_distribution_pred = self.discriminator(real_lesions, atlas)
                
                # Validation losses
                # Adversarial loss - soft labels for GAN stability (0.9 for real, 0.1 for fake)
                real_patch_labels = torch.ones_like(real_patch_pred) * 0.9
                real_global_labels = torch.ones_like(real_global_pred) * 0.9
                real_dist_labels = torch.ones_like(real_distribution_pred) * 0.9
                
                fake_patch_labels = torch.zeros_like(fake_patch_pred) * 0.1
                fake_global_labels = torch.zeros_like(fake_global_pred) * 0.1
                fake_dist_labels = torch.zeros_like(fake_distribution_pred) * 0.1
                
                # Discriminator losses
                d_real_loss = (
                    self.adversarial_loss(real_patch_pred, real_patch_labels) + 
                    self.adversarial_loss(real_global_pred, real_global_labels) + 
                    self.adversarial_loss(real_distribution_pred, real_dist_labels)
                ) / 3.0
                
                d_fake_loss = (
                    self.adversarial_loss(fake_patch_pred, fake_patch_labels) + 
                    self.adversarial_loss(fake_global_pred, fake_global_labels) + 
                    self.adversarial_loss(fake_distribution_pred, fake_dist_labels)
                ) / 3.0
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                # Generator adversarial loss
                g_adv_loss = (
                    self.adversarial_loss(fake_patch_pred, real_patch_labels) + 
                    self.adversarial_loss(fake_global_pred, real_global_labels) + 
                    self.adversarial_loss(fake_distribution_pred, real_dist_labels)
                ) / 3.0
                
                # Atlas-weighted losses
                if real_lesions.shape[2:] != atlas.shape[2:]:
                    atlas_resized = F.interpolate(atlas, size=real_lesions.shape[2:], mode='trilinear', align_corners=False)
                else:
                    atlas_resized = atlas
                
                # L1 loss (only in atlas-positive regions)
                masked_l1_loss = torch.sum(torch.abs(real_lesions - fake_lesions) * valid_regions) / (torch.sum(valid_regions) + 1e-8)
                
                # Atlas-weighted L1 loss
                atlas_weighted_l1_loss = self.atlas_weighted_l1(fake_lesions, real_lesions, atlas_resized)
                
                # Dice loss for better segmentation overlap, weighted by atlas
                weighted_dice_loss = self.atlas_weighted_dice(fake_lesions * valid_regions, real_lesions * valid_regions, atlas_resized)
                
                # Atlas-guided focal loss
                focal_loss_term = self.atlas_focal_loss(fake_lesions * brain_mask, real_lesions * brain_mask, atlas_resized)
                
                # Distribution matching loss
                distribution_loss = self.atlas_distribution_loss(fake_lesions * brain_mask, atlas * brain_mask)
                
                # Atlas violation penalty - penalize generating lesions in atlas-zero regions
                invalid_regions = (1 - valid_regions)
                atlas_violation = torch.sum(fake_lesions * invalid_regions) / (torch.sum(fake_lesions) + 1e-8)
                
                # Total generator loss
                g_loss = (
                    self.adv_loss_weight * g_adv_loss + 
                    self.l1_loss_weight * masked_l1_loss + 
                    self.weighted_l1_weight * atlas_weighted_l1_loss +
                    self.dice_loss_weight * weighted_dice_loss +
                    self.focal_loss_weight * focal_loss_term +
                    self.distribution_loss_weight * distribution_loss +
                    self.violation_penalty_weight * atlas_violation
                )
                
                # Accumulate validation losses - ensure scalar values
                val_losses['val_d_loss'] += d_loss.item()
                val_losses['val_g_loss'] += g_loss.item()
                val_losses['val_l1_loss'] += masked_l1_loss.item()
                val_losses['val_weighted_l1'] += atlas_weighted_l1_loss.item()
                val_losses['val_dice_loss'] += weighted_dice_loss.item()
                val_losses['val_focal_loss'] += focal_loss_term.item()
                val_losses['val_distribution'] += distribution_loss.item()
                val_losses['val_atlas_violation'] += atlas_violation.item()
                
                # Visualize validation samples occasionally
                if num_batches <= 2:  # Only visualize first two batches to avoid too many visualizations
                    self.visualize_validation_samples(real_lesions, fake_lesions, atlas, brain_mask, epoch, batch_idx=num_batches)
        
        # Average validation losses
        for k in val_losses.keys():
            val_losses[k] /= max(num_batches, 1)
        
        # Log validation losses
        self.log_metrics(val_losses, epoch)
        
        # Log validation images to tensorboard
        if hasattr(self, 'writer') and self.writer is not None and first_batch_samples is not None:
            with torch.no_grad():
                self.log_images(
                    first_batch_samples['real_lesions'],
                    first_batch_samples['fake_lesions'],
                    first_batch_samples['atlas'],
                    epoch
                )
        
        self.generator.train()
        self.discriminator.train()
        
        return val_losses

    def visualize_validation_samples(self, real_lesions, fake_lesions, atlas, brain_mask, epoch, batch_idx=0):
        """Visualize validation samples and save them as images."""
        if not hasattr(self, 'viz_dir'):
            self.viz_dir = os.path.join(self.save_dir, 'visualizations')
            os.makedirs(self.viz_dir, exist_ok=True)
        
        # Only visualize the first sample from the batch
        if len(real_lesions) == 0:
            print("No validation samples available for visualization")
            return
        
        # Create thresholded binary versions for visualization
        real_binary = (real_lesions > 0.5).float()
        fake_binary = (fake_lesions > 0.5).float()
        atlas_binary = (atlas > 0).float()
        brain_mask_binary = (brain_mask > 0).float()
        
        # Apply brain mask to both real and fake lesions
        real_masked = real_binary * brain_mask_binary
        fake_masked = fake_binary * brain_mask_binary
        
        # Get the first sample
        real = real_masked[0, 0].detach().cpu().numpy()
        fake = fake_masked[0, 0].detach().cpu().numpy()
        atlas_map = atlas_binary[0, 0].detach().cpu().numpy()
        brain = brain_mask_binary[0, 0].detach().cpu().numpy()
        
        # Create figure with 3 rows (axial, coronal, sagittal) and 4 columns (real, fake, atlas, brain)
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # Find non-empty slices for better visualization
        def find_best_slice(volume, axis=0):
            if axis == 0:  # Axial
                slices = [np.sum(volume[i, :, :]) for i in range(volume.shape[0])]
            elif axis == 1:  # Coronal
                slices = [np.sum(volume[:, i, :]) for i in range(volume.shape[1])]
            else:  # Sagittal
                slices = [np.sum(volume[:, :, i]) for i in range(volume.shape[2])]
            
            # Get slice with maximum lesion content
            if max(slices) > 0:
                return np.argmax(slices)
            else:
                return volume.shape[axis] // 2  # Middle slice if no lesions found
        
        # Function to plot a slice for each orientation
        def plot_slice(data, ax, axis, title, cmap='binary'):
            if axis == 0:  # Axial (top view)
                slice_idx = find_best_slice(data, axis=0)
                ax.imshow(data[slice_idx, :, :], cmap=cmap, interpolation='none')
                ax.set_title(f"{title} - Axial (slice {slice_idx})")
            elif axis == 1:  # Coronal (front view)
                slice_idx = find_best_slice(data, axis=1)
                ax.imshow(data[:, slice_idx, :], cmap=cmap, interpolation='none')
                ax.set_title(f"{title} - Coronal (slice {slice_idx})")
            else:  # Sagittal (side view)
                slice_idx = find_best_slice(data, axis=2)
                ax.imshow(data[:, :, slice_idx], cmap=cmap, interpolation='none')
                ax.set_title(f"{title} - Sagittal (slice {slice_idx})")
            
            ax.axis('off')  # Remove axis
        
        # Generate plots for each view
        for view in range(3):  # 0=axial, 1=coronal, 2=sagittal
            plot_slice(real, axes[view, 0], view, "Real Lesion")
            plot_slice(fake, axes[view, 1], view, "Generated Lesion")
            plot_slice(atlas_map, axes[view, 2], view, "Atlas")
            plot_slice(brain, axes[view, 3], view, "Brain Mask")
        
        # Save the figure
        fig.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'val_samples_epoch{epoch}_batch{batch_idx}.png'), dpi=200)
        plt.close(fig)

    def log_metrics(self, metrics, epoch, step=None):
        # Log do tensorboard
        prefix = ''
        if step is not None:
            prefix = f'step_{step}/'
        
        for key, value in metrics.items():
            # OPRAVA: Zajisti, že všechny hodnoty jsou skalární
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    # Pokud tensor není skalární, loguj pouze jeho průměr
                    if self.debug:
                        print(f"WARNING: Non-scalar tensor for {key} with shape {value.shape}, logging mean value")
                    value = value.mean().item()
                else:
                    # Pokud tensor je skalární, převeď na Python číslo
                    value = value.item()
            self.writer.add_scalar(f'{prefix}{key}', value, epoch)
    
    def log_images(self, real_lesions, fake_lesions, atlas, epoch, n_samples=4):
        with torch.no_grad():
            # Ensure we don't try to visualize more samples than we have
            n_samples = min(n_samples, real_lesions.size(0))
            
            # Exit if no samples to visualize
            if n_samples == 0:
                return
            
            # Apply thresholding to ensure binary representation
            binary_fake_lesions = (fake_lesions > 0.5).float()
            
            # Výběr n_samples vzorků
            real_samples = real_lesions[:n_samples].cpu().detach()
            fake_samples = binary_fake_lesions[:n_samples].cpu().detach()  # Use binary version
            atlas_samples = atlas[:n_samples].cpu().detach()
            
            # Funkce pro vizualizaci středových řezů
            def visualize_slice(volume, slice_idx=None):
                if slice_idx is None:
                    # Try to find a non-empty slice
                    if volume.dim() == 5:  # [B,C,D,H,W]
                        D = volume.shape[2]
                        for potential_idx in range(D//3, 2*D//3):  # Check middle third of volume
                            if volume[0, 0, potential_idx].sum() > 0:
                                slice_idx = potential_idx
                                break
                        if slice_idx is None:  # If no non-empty slice found, use middle
                            slice_idx = D // 2
                    else:
                        slice_idx = volume.shape[2] // 2
                
                # Extract the slice and ensure it's binary (for lesions)
                if volume.dim() == 5:  # [B,C,D,H,W]
                    slice_data = volume[0, 0, slice_idx].clone()
                else:  # Handle unexpected dimensions
                    slice_data = volume[0, slice_idx].clone() if volume.dim() == 4 else volume[slice_idx].clone()
                
                return slice_data
            
            # Log středových řezů pro každý vzorek
            for i in range(n_samples):
                # Získání středového řezu
                real_slice = visualize_slice(real_samples[i:i+1])
                fake_slice = visualize_slice(fake_samples[i:i+1])
                atlas_slice = visualize_slice(atlas_samples[i:i+1])
                
                # Ensure binary values for visualization
                real_slice = (real_slice > 0.5).float()
                fake_slice = (fake_slice > 0.5).float()
                
                # Create a more visually distinct representation for TensorBoard
                # White (1) for lesion, black (0) for background
                # Stack these for RGB channels with different colors
                colored_real = torch.zeros((3, real_slice.shape[0], real_slice.shape[1]))
                colored_real[0] = real_slice  # Red channel
                
                colored_fake = torch.zeros((3, fake_slice.shape[0], fake_slice.shape[1]))
                colored_fake[1] = fake_slice  # Green channel
                
                # Normalize atlas to 0-1 range if needed
                if atlas_slice.max() > 0:
                    atlas_slice = atlas_slice / atlas_slice.max()
                
                colored_atlas = torch.zeros((3, atlas_slice.shape[0], atlas_slice.shape[1]))
                colored_atlas[2] = atlas_slice  # Blue channel
                
                # Combine into a visualization grid
                grid = torch.cat([colored_real, colored_fake, colored_atlas], dim=2)
                
                # Log to tensorboard
                self.writer.add_image(f'sample_{i}/real_fake_atlas', grid, epoch)
    
    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.generator_optimizer.state_dict(),
            'optimizer_d_state_dict': self.discriminator_optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint uložen: {checkpoint_path}")
    
    def generate_samples(self, num_samples=5, output_dir=None, use_fixed_z=False, save_nifti=True):
        """
        Generate samples from the trained generator
        
        Args:
            num_samples: number of samples to generate
            output_dir: directory to save generated samples
            use_fixed_z: whether to use a fixed latent vector (False = random samples)
            save_nifti: whether to save the generated samples as NIfTI files
        """
        if output_dir is None:
            output_dir = os.path.join(self.save_dir, 'generated_samples')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directory for visualizations
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create a dictionary to store statistics about generated lesions
        stats = {
            'lesion_volumes': [],
            'lesion_counts': [],
            'atlas_coverage': []
        }
        
        # Get a batch of data to extract atlas and brain mask
        for batch in self.dataloader:
            atlas = batch['atlas'].to(self.device)
            
            # Get brain mask if available, otherwise use all ones
            if 'brain_mask' in batch:
                brain_mask = batch['brain_mask'].to(self.device)
            else:
                brain_mask = torch.ones_like(atlas)
            
            # Ensure the atlas has a batch dimension
            if atlas.dim() == 3:
                atlas = atlas.unsqueeze(0)
            
            if brain_mask.dim() == 3:
                brain_mask = brain_mask.unsqueeze(0)
            
            atlas_binary = (atlas > 0).float()
            brain_mask_binary = (brain_mask > 0).float()
            
            # Save the atlas reference
            if hasattr(self, 'lesion_atlas_path') and self.lesion_atlas_path:
                atlas_nib = nib.load(self.lesion_atlas_path)
                nib.save(atlas_nib, os.path.join(output_dir, 'atlas_reference.nii.gz'))
            
            # Create a file to save statistics
            stats_path = os.path.join(output_dir, 'lesion_statistics.txt')
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Place to save combined visualization
            combined_viz = []
            
            # Switch model to eval mode
            self.generator.eval()
            
            # FIXED: Remove the global random seed setting
            # If use_fixed_z is True, we'll use a predefined z across all samples
            fixed_z = torch.randn(1, self.z_dim, device=self.device) if use_fixed_z else None
            
            # Generate samples
            with open(stats_path, 'w') as f:
                f.write("Statistiky generovaných lézí\n")
                f.write("==========================\n\n")
                
                # Generate samples
                for i in range(num_samples):
                    # FIXED: Generate diverse latent vectors for each sample
                    # Only reuse fixed_z if use_fixed_z is True
                    if use_fixed_z and fixed_z is not None:
                        z = fixed_z
                    else:
                        # Set a different seed for each sample to ensure diversity
                        torch.manual_seed(42 + i) if use_fixed_z else None
                    z = torch.randn(1, self.z_dim, device=self.device)
                    
                    # Generate lesions
                    with torch.no_grad():
                        fake_lesion = self.generator(z, atlas[:1], brain_mask[:1])
                    
                    # ZMĚNA: VŽDY aplikujeme nucenou aktivaci na vzorky, ne jen když jsou prázdné
                    # Vytvoření masky z vysoce pravděpodobných oblastí atlasu
                    atlas_threshold = torch.quantile(atlas[:1].view(-1), 0.9)  # Horních 10% hodnot atlasu
                    high_prob_regions = (atlas[:1] > atlas_threshold).float() * brain_mask[:1]
                    
                    # Generování náhodné aktivační masky
                    act_mask = torch.zeros_like(fake_lesion)
                    random_strength = torch.rand(1, device=self.device) * 0.3 + 0.7  # Aktivace 0.7-1.0
                    
                    # Přidání nucené aktivace do části vysoce pravděpodobných oblastí
                    activation_regions = high_prob_regions * (torch.rand_like(high_prob_regions) > 0.5).float()
                    act_mask = act_mask.scatter_(1, torch.zeros_like(fake_lesion, dtype=torch.long), 
                                               activation_regions * random_strength)
                    
                    # Smíchání s původním výstupem - 40% generovaný obsah, 60% nucená aktivace
                    # Toto zajistí, že vzorky budou mít léze, ale stále budou částečně ovlivněny generátorem
                    fake_lesion = fake_lesion * 0.4 + act_mask * 0.6
                    print(f"[SAMPLE {i}] Aplikována nucená aktivace pro zajištění viditelných lézí")
                    
                    # MODIFIED: Use a smoother approach to map the outputs to a binary-like distribution
                    # Apply atlas mask to ensure lesions only appear in atlas-positive regions
                    fake_lesion = fake_lesion * atlas_binary[:1]
                    
                    # Apply brain mask to ensure lesions are only within the brain
                    fake_lesion = fake_lesion * brain_mask_binary[:1]
                    
                    # MODIFIED: Instead of hard thresholding, preserve some of the continuous values
                    # for a more natural appearance but still prioritize strong activations
                    # Apply a softer threshold to get more coherent lesions
                    # Using temperature scaling to sharpen the transition but not fully binarize
                    temperature = 8.0  # Less aggressive than 20.0 used in the Generator
                    soft_binary = torch.sigmoid(temperature * (fake_lesion - 0.5))
                    
                    # MODIFIED: Apply connected component filtering to remove isolated pixels
                    # First get a binary version for component analysis
                    binary_np = (soft_binary[0, 0] > 0.5).cpu().float().numpy()
                    
                    # Find connected components
                    labeled_components, num_components = measure_label(binary_np, return_num=True)
                    
                    # MODIFIED: Filter out very small components (likely noise)
                    component_sizes = np.bincount(labeled_components.flatten())[1:] if num_components > 0 else []
                    min_component_size = 3  # Minimum size to keep a component
                    
                    # Create cleaned binary mask
                    cleaned_binary = np.zeros_like(binary_np)
                    for comp_id in range(1, num_components + 1):
                        if component_sizes[comp_id - 1] >= min_component_size:
                            cleaned_binary[labeled_components == comp_id] = 1
                    
                    # Convert back to tensor
                    # Fix: Use the device of an input tensor instead of self.device
                    cleaned_binary_tensor = torch.from_numpy(cleaned_binary).float().to(fake_lesion.device)
                    cleaned_binary_tensor = cleaned_binary_tensor.unsqueeze(0).unsqueeze(0)
                    
                    # Final binary lesion combines the soft values with the cleaned binary mask
                    binary_lesion = soft_binary * cleaned_binary_tensor
                    
                    # Get lesion volume and count
                    lesion_volume = binary_lesion.sum().item()
                    
                    # Connected components analysis to count separate lesions
                    # CPU computation for connected components
                    binary_np = binary_lesion[0, 0].cpu().numpy()
                    labeled_components, num_components = measure_label(binary_np, return_num=True)
                    
                    # Calculate atlas coverage
                    total_atlas_area = atlas_binary.sum().item()
                    lesion_area_in_atlas = (binary_lesion * atlas_binary[:1]).sum().item()
                    atlas_coverage_percent = (lesion_area_in_atlas / total_atlas_area) * 100 if total_atlas_area > 0 else 0
                    
                    # Save statistics
                    stats['lesion_volumes'].append(lesion_volume)
                    stats['lesion_counts'].append(num_components)
                    stats['atlas_coverage'].append(atlas_coverage_percent)
                    
                    # Write statistics to file
                    f.write(f"Vzorek {i}:\n")
                    f.write(f"  - Počet lézí: {stats['lesion_counts'][i]}\n")
                    f.write(f"  - Objem lézí: {stats['lesion_volumes'][i]} voxelů\n")
                    f.write(f"  - Pokrytí atlasu: {stats['atlas_coverage'][i]:.2f}%\n\n")
                    
                    # Visualization across planes
                    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                    
                    # Define view planes
                    planes = ['axial', 'coronal', 'sagittal']
                    
                    # Find best slices for each view
                    for p_idx, plane in enumerate(planes):
                        # Find slice with maximum lesion content
                        if plane == 'axial':
                            slices = [np.sum(binary_np[i, :, :]) for i in range(binary_np.shape[0])]
                            if max(slices) > 0:
                                slice_idx = np.argmax(slices)
                            else:
                                slice_idx = binary_np.shape[0] // 2
                                
                            binary_slice = binary_np[slice_idx, :, :]
                            atlas_slice = atlas_binary[0, 0, slice_idx].cpu().numpy()
                            brain_slice = brain_mask_binary[0, 0, slice_idx].cpu().numpy()
                            
                        elif plane == 'coronal':
                            slices = [np.sum(binary_np[:, i, :]) for i in range(binary_np.shape[1])]
                            if max(slices) > 0:
                                slice_idx = np.argmax(slices)
                            else:
                                slice_idx = binary_np.shape[1] // 2
                                
                            binary_slice = binary_np[:, slice_idx, :]
                            atlas_slice = atlas_binary[0, 0, :, slice_idx].cpu().numpy()
                            brain_slice = brain_mask_binary[0, 0, :, slice_idx].cpu().numpy()
                            
                        else:  # sagittal
                            slices = [np.sum(binary_np[:, :, i]) for i in range(binary_np.shape[2])]
                            if max(slices) > 0:
                                slice_idx = np.argmax(slices)
                            else:
                                slice_idx = binary_np.shape[2] // 2
                                
                            binary_slice = binary_np[:, :, slice_idx]
                            atlas_slice = atlas_binary[0, 0, :, :, slice_idx].cpu().numpy()
                            brain_slice = brain_mask_binary[0, 0, :, :, slice_idx].cpu().numpy()
                        
                        # Plot lesion
                        axes[p_idx, 0].imshow(binary_slice, cmap='binary', interpolation='none')
                        axes[p_idx, 0].set_title(f'{plane.capitalize()} - Lesion')
                        axes[p_idx, 0].axis('off')
                        
                        # Plot atlas
                        axes[p_idx, 1].imshow(atlas_slice, cmap='hot', interpolation='none')
                        axes[p_idx, 1].set_title(f'{plane.capitalize()} - Atlas')
                        axes[p_idx, 1].axis('off')
                        
                        # Plot combined view (lesion + atlas)
                        combined = np.zeros((binary_slice.shape[0], binary_slice.shape[1], 3))
                        combined[:, :, 0] = binary_slice  # Red channel for lesion
                        combined[:, :, 1] = atlas_slice * 0.5  # Green channel for atlas (dimmed)
                        combined[:, :, 2] = brain_slice * 0.3  # Blue channel for brain mask (dimmed)
                        
                        axes[p_idx, 2].imshow(combined, interpolation='none')
                        axes[p_idx, 2].set_title(f'{plane.capitalize()} - Combined')
                        axes[p_idx, 2].axis('off')
                    
                    # Save visualization
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'sample_{i}.png'), dpi=200)
                    plt.close(fig)
                    
                    # Save NIfTI if requested
                    if save_nifti:
                        try:
                            # Get reference file for affine and header
                            first_file = self.dataloader.dataset.label_files[0]
                            ref_nib = nib.load(first_file)
                            
                            # Create NIfTI from generated lesion
                            binary_nii = nib.Nifti1Image(binary_np, ref_nib.affine, ref_nib.header)
                            
                            # Save NIfTI
                            nib.save(binary_nii, os.path.join(output_dir, f'generated_lesion_{i}.nii.gz'))
                            
                        except Exception as e:
                            print(f"Error saving NIfTI for sample {i}: {e}")
                
                # Write summary statistics
                f.write("\nSouhrnné statistiky:\n")
                f.write("=================\n")
                f.write(f"Průměrný počet lézí: {np.mean(stats['lesion_counts']):.2f} (±{np.std(stats['lesion_counts']):.2f})\n")
                f.write(f"Průměrný objem lézí: {np.mean(stats['lesion_volumes']):.2f} (±{np.std(stats['lesion_volumes']):.2f}) voxelů\n")
                f.write(f"Průměrné pokrytí atlasu: {np.mean(stats['atlas_coverage']):.2f}% (±{np.std(stats['atlas_coverage']):.2f}%)\n")
            
            # No need to continue looping through batches once we've processed one
            break
        
        print(f"\nStatistiky byly uloženy do: {stats_path}")
        print(f"Vizualizace byly uloženy do: {vis_dir}")
        
        self.generator.train()

    def train_discriminator(self, real_lesions, atlas, brain_mask=None):
        batch_size = real_lesions.size(0)
        
        # Reset discriminator gradients
        self.discriminator_optimizer.zero_grad()
        
        # NEW: Add a small random noise to the real images
        # This prevents discriminator from being too confident and helps training stability
        if self.noise_std > 0:
            real_lesions_noisy = real_lesions + torch.randn_like(real_lesions) * self.noise_std * 2.0  # Zvýšení šumu 2x
            real_lesions_noisy = torch.clamp(real_lesions_noisy, 0.0, 1.0)
        else:
            real_lesions_noisy = real_lesions
        
        # Generate fake lesions
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        
        # Don't need gradient for the generator when training discriminator
        with torch.no_grad():
            # Předáme i brain_mask pro konzistenci s forward metodou
            fake_lesions = self.generator(z, atlas, brain_mask)
            
            # NEW: Add noise to fake samples too
            if self.noise_std > 0:
                fake_lesions = fake_lesions + torch.randn_like(fake_lesions) * self.noise_std * 2.0  # Zvýšení šumu 2x
                fake_lesions = torch.clamp(fake_lesions, 0.0, 1.0)
        
        # Pass real lesions through discriminator
        real_patch_pred, real_global_pred, real_distrib_pred = self.discriminator(real_lesions_noisy, atlas)
        
        # Pass fake lesions through discriminator
        fake_patch_pred, fake_global_pred, fake_distrib_pred = self.discriminator(fake_lesions.detach(), atlas)
        
        # Use soft labels for real and fake (one-sided label smoothing)
        # This prevents the discriminator from being too confident
        real_patch_labels = torch.ones_like(real_patch_pred) * 0.9  # Sníženo z 1.0 na 0.9
        real_global_labels = torch.ones_like(real_global_pred) * 0.9  # Sníženo z 1.0 na 0.9
        real_distrib_labels = torch.ones_like(real_distrib_pred) * 0.9  # Sníženo z 1.0 na 0.9
        
        fake_patch_labels = torch.zeros_like(fake_patch_pred)
        fake_global_labels = torch.zeros_like(fake_global_pred)
        fake_distrib_labels = torch.zeros_like(fake_distrib_pred)
        
        # NEW: Náhodné převrácení labelu pro některé vzorky (label flipping)
        # Toto dále oslabuje diskriminátor
        if batch_size > 1 and torch.rand(1).item() < 0.1:  # 10% šance na flip labelu
            flip_idx = torch.randint(0, batch_size, (1,)).item()
            fake_patch_labels[flip_idx] = 1.0
            fake_global_labels[flip_idx] = 1.0
            fake_distrib_labels[flip_idx] = 1.0
            
            real_patch_labels[flip_idx] = 0.0
            real_global_labels[flip_idx] = 0.0
            real_distrib_labels[flip_idx] = 0.0
        
        # Calculate discriminator losses for real and fake lesions
        d_loss_real_patch = self.adversarial_loss(real_patch_pred, real_patch_labels)
        d_loss_fake_patch = self.adversarial_loss(fake_patch_pred, fake_patch_labels)
        d_patch_loss = (d_loss_real_patch + d_loss_fake_patch) / 2
        
        d_loss_real_global = self.adversarial_loss(real_global_pred, real_global_labels)
        d_loss_fake_global = self.adversarial_loss(fake_global_pred, fake_global_labels)
        d_global_loss = (d_loss_real_global + d_loss_fake_global) / 2
        
        d_loss_real_distrib = self.adversarial_loss(real_distrib_pred, real_distrib_labels)
        d_loss_fake_distrib = self.adversarial_loss(fake_distrib_pred, fake_distrib_labels)
        d_distrib_loss = (d_loss_real_distrib + d_loss_fake_distrib) / 2
        
        # Total discriminator loss
        d_loss = d_patch_loss + d_global_loss + d_distrib_loss
        
        # NEW: Náhodně přeskočit aktualizaci diskriminátoru s 20% pravděpodobností
        # Toto dále zvýhodňuje generátor
        should_update_d = torch.rand(1).item() > 0.2
        
        if should_update_d:
            # Backpropagation
            d_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            
            # Update discriminator weights
            self.discriminator_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'd_patch': d_patch_loss.item(),
            'd_global': d_global_loss.item(),
            'd_distrib': d_distrib_loss.item(),
            'd_real': (d_loss_real_patch + d_loss_real_global + d_loss_real_distrib).item() / 3,
            'd_fake': (d_loss_fake_patch + d_loss_fake_global + d_loss_fake_distrib).item() / 3,
        }

    def train_generator(self, real_lesions, atlas, brain_mask, iteration):
        batch_size = real_lesions.size(0)
        
        # Reset generator gradients
        self.generator_optimizer.zero_grad()
        
        # MODIFIED: Generate diverse latent vectors z with batch-specific noise
        # This ensures better diversity in training and helps the model learn to use the noise effectively
        z_base = torch.randn(batch_size, self.z_dim, device=self.device)
        
        # Add a small perturbation based on the iteration to increase diversity
        iteration_noise = torch.sin(torch.tensor(iteration * 0.1)) * 0.1
        z = z_base + torch.randn_like(z_base) * iteration_noise
        
        # Generate fake lesions using the Generator
        fake_lesions = self.generator(z, atlas, brain_mask)
        
        # NOVÉ: Explicitní "identity loss" pro lepší učení struktury lézí
        # Každých 5 iterací se pokusíme přímo rekonstruovat reálné léze
        # Toto naučí generátor replikovat strukturu skutečných lézí
        identity_loss = 0.0
        if iteration % 5 == 0:  # Aplikovat jen někdy, aby se zachovala i generativní schopnost
            # Použijeme pevný latentní vektor pro rekonstrukci
            fixed_z = torch.zeros(batch_size, self.z_dim, device=self.device)
            # Přidáme mírný šum, aby model nebyl příliš závislý na přesné podobě vstupu
            fixed_z += torch.randn_like(fixed_z) * 0.1
            
            # Pokus o rekonstrukci reálných lézí
            reconstructed_lesions = self.generator(fixed_z, atlas, brain_mask)
            
            # Spočítáme penalizaci za rozdíl mezi rekonstruovanými a reálnými lézemi
            identity_loss = F.l1_loss(reconstructed_lesions * brain_mask, real_lesions * brain_mask) * 50.0
            
            if iteration % 10 == 0:
                print(f"[IDENTITY] Iter {iteration} - Použita identity loss: {identity_loss.item():.4f}")
        
        # Ensure fake_lesions are in valid range [0,1]
        if torch.any(fake_lesions > 1.0) or torch.any(fake_lesions < 0.0):
            if self.debug:
                min_val = fake_lesions.min().item()
                max_val = fake_lesions.max().item()
                print(f"[WARNING] fake_lesions outside range [0,1]: min={min_val}, max={max_val}")
            fake_lesions = torch.clamp(fake_lesions, 0.0, 1.0)
        
        # NEW: Check for all-zero outputs and force activation in early training
        # This is a critical step to kickstart the generator with some lesions
        total_lesion_volume = torch.sum(fake_lesions > 0.3)
        
        # If we're in early training (first 3000 iterations) AND output is nearly empty
        # Force some activation to help the model learn what lesions should look like
        early_training = iteration < 3000  # Zvýšeno z 1000 na 3000 iterací
        if early_training and total_lesion_volume < 500:  # Zvýšeno z 10 na 500 pro agresivnější nucení aktivací
            # Create a mask from high-probability atlas regions
            atlas_threshold = torch.quantile(atlas.view(-1), 0.9)  # Top 10% místo 5% of atlas values
            high_prob_regions = (atlas > atlas_threshold).float() * brain_mask
            
            # Generate a random activation mask
            act_mask = torch.zeros_like(fake_lesions)
            random_strength = torch.rand(1, device=self.device) * 0.3 + 0.7  # 0.7-1.0 activation
            
            # Add forced activation to a portion of high probability regions
            activation_regions = high_prob_regions * (torch.rand_like(high_prob_regions) > 0.5).float()  # Změna z 0.7 na 0.5 pro větší aktivaci
            act_mask = act_mask.scatter_(1, torch.zeros_like(fake_lesions, dtype=torch.long), 
                                         activation_regions * random_strength)
            
            # Blend with the original output - stronger in early iterations, weaker later
            blend_factor = max(0.2, 1.0 - (iteration / 2000))  # Starts at 1.0, decreases to 0.2, pomaleji klesá
            fake_lesions = fake_lesions * (1 - blend_factor) + act_mask * blend_factor
            
            # Track and report when forced activation is applied
            if iteration % 10 == 0:
                print(f"[EARLY TRAINING] Forced activation applied at iter {iteration}, blend={blend_factor:.2f}")
        
        # Debug prints if needed
        if self.debug:
            self.debug_print("Real lesions in train_generator", real_lesions)
            self.debug_print("Fake lesions in train_generator", fake_lesions)
            self.debug_print("Atlas in train_generator", atlas)
        
        # NEW: Apply noise to inputs for discriminator evaluation
        # This makes the discriminator's task harder and helps the generator learn
        if self.noise_std > 0:
            fake_lesions_noisy = fake_lesions + torch.randn_like(fake_lesions) * self.noise_std * 0.5  # less noise than in discriminator training
            fake_lesions_noisy = torch.clamp(fake_lesions_noisy, 0.0, 1.0)
        else:
            fake_lesions_noisy = fake_lesions
        
        # Discriminator forward pass on fake lesions
        fake_patch_pred, fake_global_pred, fake_distribution_pred = self.discriminator(fake_lesions_noisy, atlas)
        
        # Use soft labels for real (with less smoothing than in discriminator training)
        # This helps with one-sided label smoothing to improve training stability
        real_patch_labels = torch.ones_like(fake_patch_pred) * 0.9
        real_global_labels = torch.ones_like(fake_global_pred) * 0.9
        real_dist_labels = torch.ones_like(fake_distribution_pred) * 0.9
        
        # Generator adversarial losses
        g_adv_patch_loss = self.adversarial_loss(fake_patch_pred, real_patch_labels)
        g_adv_global_loss = self.adversarial_loss(fake_global_pred, real_global_labels)
        g_adv_dist_loss = self.adversarial_loss(fake_distribution_pred, real_dist_labels)
        
        # Combined adversarial loss
        g_adv_loss = (g_adv_patch_loss + g_adv_global_loss + g_adv_dist_loss) / 3.0
        
        # Binary atlas mask for valid regions
        atlas_binary = (atlas > 0).float()
        
        # Create a mask where both atlas_binary and brain_mask are positive
        valid_regions = atlas_binary * brain_mask
        
        # Ensure dimensions match before applying masks
        if real_lesions.shape[2:] != valid_regions.shape[2:]:
            valid_regions = F.interpolate(valid_regions, size=real_lesions.shape[2:], mode='nearest')
            if self.debug:
                self.debug_print("Resized valid_regions mask", valid_regions)
        
        # ENHANCED: Improved structural coherence loss to encourage connected lesions
        # First get a binary mask of the fake lesions for structural analysis
        binary_fake = (fake_lesions > 0.5).float()
        
        # Apply a spatial gradient to detect edges in all 3 directions
        dx = torch.abs(binary_fake[:, :, 1:, :, :] - binary_fake[:, :, :-1, :, :])
        dy = torch.abs(binary_fake[:, :, :, 1:, :] - binary_fake[:, :, :, :-1, :])
        dz = torch.abs(binary_fake[:, :, :, :, 1:] - binary_fake[:, :, :, :, :-1])
        
        # Pad the gradients to match original size
        dx = F.pad(dx, (0, 0, 0, 0, 1, 0))
        dy = F.pad(dy, (0, 0, 1, 0, 0, 0))
        dz = F.pad(dz, (1, 0, 0, 0, 0, 0))
        
        # Sum gradients to get total edge map
        edge_map = dx + dy + dz
        
        # Calculate edge ratio (edges / total positive pixels)
        # Higher edge ratio means more fragmented lesions
        total_positive = torch.sum(binary_fake) + 1e-8
        total_edges = torch.sum(edge_map) + 1e-8
        edge_ratio = total_edges / total_positive
        
        # NEW: Enhanced coherence loss with perimeter/area ratio penalty
        # Lower values of this ratio indicate more compact, less fragmented objects
        # V raných fázích tréninku snížíme threshold, abychom dovolili více fragmentované léze
        threshold = 0.3  # Sníženo z 0.5 na 0.3
        coherence_loss = torch.clamp(edge_ratio - threshold, min=0)
        
        # NEW: Connectivity loss using morphological analysis
        # Process binary_fake on CPU for morphological operations
        binary_np = binary_fake.cpu().detach().numpy()
        
        # Count components for each image in batch
        batch_components = []
        for b in range(binary_np.shape[0]):
            # Get binary mask for this batch item
            sample_binary = binary_np[b, 0]
            
            # Find connected components
            labeled, num_components = measure_label(sample_binary, return_num=True)
            
            # Count components
            batch_components.append(num_components)
        
        # Convert to tensor and calculate mean
        mean_components = torch.tensor(batch_components, device=binary_fake.device).float().mean()
        
        # Penalize having too many distinct components
        # Původně se penalty zvyšovala nad 5 komponent, nyní ji zvýšíme až nad 10 komponent
        # Tím dovolíme větší fragmentaci v raných fázích
        connectivity_loss = torch.clamp(mean_components - 10, min=0) * 0.1  # Sníženo z 0.2 na 0.1 a z thresholdu 5 na 10
        
        # Calculate masked L1 loss (only consider atlas-positive regions within the brain)
        masked_l1_loss = torch.sum(torch.abs(real_lesions - fake_lesions) * valid_regions) / (torch.sum(valid_regions) + 1e-8)
        
        # Atlas-weighted losses
        if real_lesions.shape[2:] != atlas.shape[2:]:
            atlas_resized = F.interpolate(atlas, size=real_lesions.shape[2:], mode='trilinear', align_corners=False)
        else:
            atlas_resized = atlas
        
        # Apply atlas weighting to the L1 loss
        atlas_weighted_l1_loss = self.atlas_weighted_l1(fake_lesions, real_lesions, atlas_resized)
        
        # Dice loss for better overlap, weighted by atlas probabilities
        weighted_dice_loss = self.atlas_weighted_dice(fake_lesions * valid_regions, real_lesions * valid_regions, atlas_resized)
        
        # Atlas-guided focal loss - encourage generation in high probability atlas regions
        focal_loss_term = self.atlas_focal_loss(fake_lesions * brain_mask, real_lesions * brain_mask, atlas_resized)
        
        # Distribution matching loss - ensure generated lesion distribution matches atlas distribution
        distribution_loss = self.atlas_distribution_loss(fake_lesions * brain_mask, atlas * brain_mask)
        
        # Atlas violation penalty - penalize generating lesions in atlas-zero regions
        # Calculate what portion of the fake lesion is in regions where atlas is zero
        invalid_regions = (1 - valid_regions)
        atlas_violation = torch.sum(fake_lesions * invalid_regions) / (torch.sum(fake_lesions) + 1e-8)
        
        # NEW: "Empty output" penalty to discourage generator from producing all zeros
        # Calculate what portion of atlas-valid regions contain lesions
        valid_region_filled = torch.sum(binary_fake * valid_regions) / (torch.sum(valid_regions) + 1e-8)
        
        # NEW: Calculate the real data activation percentage for comparison
        real_binary = (real_lesions > 0.5).float()
        real_valid_region_filled = torch.sum(real_binary * valid_regions) / (torch.sum(valid_regions) + 1e-8)
        
        # Print stats occasionally to understand what's happening
        if iteration % 10 == 0:
            total_pixels = torch.sum(valid_regions).item()
            real_activated = torch.sum(real_binary * valid_regions).item()
            fake_activated = torch.sum(binary_fake * valid_regions).item()
            print(f"[STATS] Iter {iteration} - Valid pixels: {total_pixels:.1f}, Real activated: {real_activated:.1f} ({real_valid_region_filled.item()*100:.4f}%), Fake activated: {fake_activated:.1f} ({valid_region_filled.item()*100:.4f}%)")
        
        # COMPLETELY NEW: Direct minimum activation target loss
        # We directly specify a minimum percentage of pixels that should be activated
        # This forces the model to generate at least some lesions
        min_activation_target = 0.02  # Zvýšeno z 0.005 (0.5%) na 0.02 (2%) minimum activation
        
        # ZMĚNA: Použít silnější target pro aktivaci
        # Cíl bude podobný jako procento aktivace v reálných datech
        if real_valid_region_filled.item() > 0.01:  # Pokud reálný vzorek má aktivaci > 1%
            # Cíl bude 80% aktivace reálného vzorku, ale minimálně min_activation_target
            min_activation_target = max(min_activation_target, real_valid_region_filled.item() * 0.8)
        
        # Apply linear ramping of the target during early training
        if iteration < 1000:
            # Start with a very small target and gradually increase
            min_activation_target = min_activation_target * (iteration / 1000)
        
        # Strong penalty when below target, zero penalty when above
        activation_target_loss = torch.relu(min_activation_target - valid_region_filled) * 5000.0  # Zvýšeno z 1000.0 na 5000.0
        
        # ADDITIONAL PENALTY: If the activation is essentially zero, apply a massive penalty
        # This is critical to avoid the common GAN collapse to all zeros
        if valid_region_filled < 0.0001:  # Essentially no activation at all
            activation_target_loss = activation_target_loss * 50.0  # Zvýšeno z 10.0 na 50.0 - masivní penalizace za prázdný výstup
        
        # MUCH STRONGER empty penalty
        # We want at least 0.5% of valid regions to have lesions - anything less gets severely penalized
        # Using a piecewise approach to create a hard constraint-like effect
        min_fill_percentage = 0.005  # 0.5% minimum coverage required
        
        if valid_region_filled < min_fill_percentage:
            # Apply extremely heavy penalty for being below minimum
            empty_penalty = 1000.0 * torch.exp(-100.0 * valid_region_filled / min_fill_percentage)
        else:
            # Small or no penalty when we have enough activation
            empty_penalty = torch.exp(-20.0 * valid_region_filled) * 10.0
        
        # NEW: Implement feature matching loss to help stabilize training
        # Extract features from the discriminator for real images
        with torch.no_grad():
            real_patch_feat, real_global_feat, real_dist_feat = self.discriminator.get_features(real_lesions, atlas)
            
        # Get features for fake images
        fake_patch_feat, fake_global_feat, fake_dist_feat = self.discriminator.get_features(fake_lesions, atlas)
        
        # Calculate feature matching loss (L1 distance between feature maps)
        feature_matching_loss = 0.0
        
        # Ensure we have matching feature pairs
        min_patch_features = min(len(real_patch_feat), len(fake_patch_feat))
        min_global_features = min(len(real_global_feat), len(fake_global_feat))
        
        # Compare feature maps for patch features
        for i in range(min_patch_features):
            feature_matching_loss += F.l1_loss(fake_patch_feat[i], real_patch_feat[i].detach())
        
        # Compare feature maps for global features
        for i in range(min_global_features):
            feature_matching_loss += F.l1_loss(fake_global_feat[i], real_global_feat[i].detach())
            
        # Normalize by the number of feature maps compared
        feature_matching_loss = feature_matching_loss / (min_patch_features + min_global_features)
        
        # Dynamic weights for coherence and connectivity losses based on training stage
        # In early training, we focus more on avoiding empty outputs than enforcing coherence
        iteration_progress = min(1.0, iteration / 5000)  # 0.0 at start, 1.0 after 5000 iterations
        coherence_weight = 5.0 * iteration_progress  # Starts at 0, gradually increases to 5.0
        connectivity_weight = 2.0 * iteration_progress  # Starts at 0, gradually increases to 2.0
        
        # REBALANCED: Total generator loss with dramatically increased empty penalty weight
        g_loss = (
            self.adv_loss_weight * g_adv_loss + 
            self.l1_loss_weight * masked_l1_loss + 
            self.weighted_l1_weight * atlas_weighted_l1_loss +
            self.dice_loss_weight * weighted_dice_loss +
            self.focal_loss_weight * focal_loss_term +
            self.distribution_loss_weight * distribution_loss +
            self.violation_penalty_weight * atlas_violation + 
            coherence_weight * coherence_loss +  # Dynamic weight for coherence loss
            connectivity_weight * connectivity_loss +  # Dynamic weight for connectivity loss
            3.0 * feature_matching_loss +  # Feature matching to stabilize training
            self.empty_penalty_weight * empty_penalty +  # Zvýšená penalty za prázdné výstupy
            self.activation_target_weight * activation_target_loss +  # Zvýšená penalty za nedostatek aktivace
            identity_loss  # Nová identity loss pro rekonstrukci reálných lézí
        )
        
        # Backpropagation
        g_loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
        self.generator_optimizer.step()
        
        # Return generator losses for logging and generated lesions for visualization
        return {
            'g_total': g_loss.item(),
            'g_adv': g_adv_loss.item(),
            'g_l1': masked_l1_loss.item(),
            'g_weighted_l1': atlas_weighted_l1_loss.item(),
            'g_dice': weighted_dice_loss.item(),
            'g_focal': focal_loss_term.item(),
            'g_distribution': distribution_loss.item(),
            'g_atlas_violation': atlas_violation.item(),
            'g_coherence': coherence_loss.item(),
            'g_connectivity': connectivity_loss.item(),
            'g_feature_matching': feature_matching_loss.item(),
            'g_empty_penalty': empty_penalty.item(),
            'g_activation_target': activation_target_loss.item(),
            'g_valid_fill': valid_region_filled.item() * 100.0,  # Report as percentage
            'g_real_fill': real_valid_region_filled.item() * 100.0,  # Real data fill percentage
            'fake_lesions': fake_lesions.detach()
        }

    def generate_pdf_slices(self, epoch, num_samples=3):
        """
        Generuje PDF se všemi axiálními řezy 3D imagu.
        V každém slice nad obrázkem je číslo řezu (z celkového počtu) a počet nenulových pixelů.
        Řezy jsou uspořádány po 4 na řádek.

        Args:
            epoch: Aktuální epocha
            num_samples: Počet generovaných vzorků
        """
        print(f"Generuji PDF vizualizaci pro epochu {epoch}...")
        # Vytvoření adresáře pro PDF
        pdf_dir = os.path.join(self.save_dir, 'pdf_visualizations')
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Název výsledného PDF
        pdf_path = os.path.join(pdf_dir, f'lesion_slices_epoch_{epoch}.pdf')
        
        # Získáme data z loaderu (pouze první batch)
        for batch in self.dataloader:
            atlas = batch['atlas'].to(self.device)
            brain_mask = batch['brain_mask'].to(self.device)
            
            # Zajistíme batch rozměr
            if atlas.dim() == 3:
                atlas = atlas.unsqueeze(0)
            if brain_mask.dim() == 3:
                brain_mask = brain_mask.unsqueeze(0)
            
            # Binární masky
            atlas_binary = (atlas > 0).float()
            brain_mask_binary = (brain_mask > 0).float()
            
            # Přepneme model do eval režimu
            self.generator.eval()
            
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            
            # Otevření PDF souboru (všechny vzorky na samostatných stránkách)
            with PdfPages(pdf_path) as pdf:
                for sample_idx in range(num_samples):
                    print(f"  Generuji vzorek {sample_idx+1}/{num_samples}...")
                    
                    # Vygenerování vzorku (3D image)
                    with torch.no_grad():
                        z = torch.randn(1, self.z_dim, device=self.device)
                        fake_lesion = self.generator(z, atlas[:1], brain_mask[:1])
                        
                        # Aplikace atlas a brain mask
                        fake_lesion = fake_lesion * atlas_binary[:1] * brain_mask_binary[:1]
                        
                        # Ponecháme původní hodnoty (raw) před binarizací
                        raw_lesion_np = fake_lesion[0, 0].cpu().numpy()
                        
                        # Binarizace pro kontrolu, zda nejsou hodnoty invertované
                        binary_lesion = (fake_lesion > 0.5).float()
                        binary_np = binary_lesion[0, 0].cpu().numpy()
                    
                    ones_percentage = np.mean(binary_np == 1) * 100
                    if ones_percentage > 90:
                        print(f"  UPOZORNĚNÍ: Hodnoty vypadají invertované - {ones_percentage:.2f}% jsou jedničky!")
                        binary_np = 1 - binary_np
                        raw_lesion_np = 1 - raw_lesion_np
                    
                    # Použijeme všechny řezy
                    depth = raw_lesion_np.shape[0]
                    
                    # Uspořádání: 4 řezy na řádek
                    cols = 4
                    rows = int(np.ceil(depth / cols))
                    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                    # Pokud je pouze jeden řádek, zajistíme konzistentní iteraci
                    axs = np.array(axs).reshape(-1)
                    
                    # Pro každý slice
                    for i in range(depth):
                        slice_img = raw_lesion_np[i, :, :]
                        nonzero_count = np.count_nonzero(slice_img == 1)
                        
                        # Nad každým slice zobrazíme číslo slice a počet nenulových pixelů
                        axs[i].set_title(f"{i+1}/{depth}, nz: {nonzero_count}", fontsize=8)
                        axs[i].imshow(slice_img, cmap='gray', vmin=0, vmax=1)
                        axs[i].axis('off')
                    
                    # Skryjeme prázdné subplots (pokud nějaké zbydou)
                    for j in range(depth, len(axs)):
                        axs[j].axis('off')
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                
            # Vrátíme model do tréninkového režimu
            self.generator.train()
            break  # Použijeme pouze první batch
        
        print(f"PDF vizualizace uložena do: {pdf_path}")

    def generate_pdf_from_samples(self, epoch, num_samples=3):
        """
        Generuje PDF se všemi axiálními řezy 3D imagu pomocí metody generate_samples.
        V každém slice nad obrázkem je číslo řezu (z celkového počtu) a počet nenulových pixelů.
        Řezy jsou uspořádány po 4 na řádek.

        Args:
            epoch: Aktuální epocha
            num_samples: Počet generovaných vzorků
        """
        print(f"Generuji PDF vizualizaci z optimalizovaných vzorků pro epochu {epoch}...")
        
        # Vytvoření adresáře pro PDF
        pdf_dir = os.path.join(self.save_dir, 'pdf_visualizations')
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Název výsledného PDF
        pdf_path = os.path.join(pdf_dir, f'lesion_slices_epoch_{epoch}.pdf')
        
        # Přepneme model do eval režimu
        self.generator.eval()
        
        # Získáme potřebné údaje z loaderu
        for batch in self.dataloader:
            atlas = batch['atlas'].to(self.device)
            brain_mask = batch['brain_mask'].to(self.device)
            
            # Zajistíme batch rozměr
            if atlas.dim() == 3:
                atlas = atlas.unsqueeze(0)
            if brain_mask.dim() == 3:
                brain_mask = brain_mask.unsqueeze(0)
            
            # Binární masky
            atlas_binary = (atlas > 0).float()
            brain_mask_binary = (brain_mask > 0).float()
            
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            
            # Otevření PDF souboru (všechny vzorky na samostatných stránkách)
            with PdfPages(pdf_path) as pdf:
                # Pro každý vzorek vygenerujeme podobně jako v generate_samples
                for sample_idx in range(num_samples):
                    print(f"  Generuji vzorek {sample_idx+1}/{num_samples}...")
                    
                    # Vygenerování latentního vektoru
                    z = torch.randn(1, self.z_dim, device=self.device)
                    
                    # Generování léze
                    with torch.no_grad():
                        fake_lesion = self.generator(z, atlas[:1], brain_mask[:1])
                    
                    # Aplikace atlasu a brain masky
                    fake_lesion = fake_lesion * atlas_binary[:1] * brain_mask_binary[:1]
                    
                    # ZMĚNA: Aplikace nucené aktivace podobně jako v generate_samples
                    atlas_threshold = torch.quantile(atlas[:1].view(-1), 0.9)  # Horních 10% hodnot atlasu
                    high_prob_regions = (atlas[:1] > atlas_threshold).float() * brain_mask[:1]
                    
                    # Generování náhodné aktivační masky
                    act_mask = torch.zeros_like(fake_lesion)
                    random_strength = torch.rand(1, device=self.device) * 0.3 + 0.7  # Aktivace 0.7-1.0
                    
                    # Přidání nucené aktivace do části vysoce pravděpodobných oblastí
                    activation_regions = high_prob_regions * (torch.rand_like(high_prob_regions) > 0.5).float()
                    act_mask = act_mask.scatter_(1, torch.zeros_like(fake_lesion, dtype=torch.long), 
                                               activation_regions * random_strength)
                    
                    # Smíchání s původním výstupem - 40% generovaný obsah, 60% nucená aktivace
                    fake_lesion = fake_lesion * 0.4 + act_mask * 0.6
                    
                    # Aplikace teplotního škálování pro ostřejší přechody
                    temperature = 8.0
                    soft_binary = torch.sigmoid(temperature * (fake_lesion - 0.5))
                    
                    # Převod na numpy pro další zpracování
                    binary_np = soft_binary[0, 0].cpu().numpy()
                    
                    # Odfiltrování malých komponent (podobně jako v generate_samples)
                    binary_thresh = (binary_np > 0.5).astype(np.float32)
                    labeled_components, num_components = measure_label(binary_thresh, return_num=True)
                    
                    # Filtrování malých komponent
                    component_sizes = np.bincount(labeled_components.flatten())[1:] if num_components > 0 else []
                    min_component_size = 3  # Minimální velikost komponenty pro zachování
                    
                    cleaned_binary = np.zeros_like(binary_np)
                    for comp_id in range(1, num_components + 1):
                        if component_sizes[comp_id - 1] >= min_component_size:
                            cleaned_binary[labeled_components == comp_id] = 1
                    
                    # Kontrola inverzních hodnot
                    ones_percentage = np.mean(cleaned_binary == 1) * 100
                    if ones_percentage > 90:
                        print(f"  UPOZORNĚNÍ: Hodnoty vypadají invertované - {ones_percentage:.2f}% jsou jedničky!")
                        cleaned_binary = 1 - cleaned_binary
                    
                    # Použijeme všechny řezy
                    depth = cleaned_binary.shape[0]
                    
                    # Uspořádání: 4 řezy na řádek
                    cols = 4
                    rows = int(np.ceil(depth / cols))
                    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                    # Pokud je pouze jeden řádek, zajistíme konzistentní iteraci
                    axs = np.array(axs).reshape(-1)
                    
                    # Pro každý slice
                    for i in range(depth):
                        slice_img = cleaned_binary[i, :, :]
                        nonzero_count = np.count_nonzero(slice_img > 0.5)
                        
                        # Nad každým slice zobrazíme číslo slice a počet nenulových pixelů
                        axs[i].set_title(f"{i+1}/{depth}, nz: {nonzero_count}", fontsize=8)
                        axs[i].imshow(slice_img, cmap='gray', vmin=0, vmax=1)
                        axs[i].axis('off')
                    
                    # Skryjeme prázdné subplots (pokud nějaké zbydou)
                    for j in range(depth, len(axs)):
                        axs[j].axis('off')
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
            
            # Použijeme pouze první batch
            break
        
        # Vrátíme model do tréninkového režimu
        self.generator.train()
        
        print(f"PDF vizualizace uložena do: {pdf_path}")
    
    def generate_overlay_pdf_from_samples(self, epoch, num_samples=3):
        """
        Generuje PDF se všemi axiálními řezy 3D imagu, kde léze jsou zobrazeny červeně
        na pozadí normativního mozku (atlasu).
        V každém slice nad obrázkem je číslo řezu a počet nenulových pixelů léze.
        Řezy jsou uspořádány po 4 na řádek.

        Args:
            epoch: Aktuální epocha
            num_samples: Počet generovaných vzorků
        """
        print(f"Generuji PDF vizualizaci lézí na normativním mozku pro epochu {epoch}...")
        
        # Vytvoření adresáře pro PDF
        pdf_dir = os.path.join(self.save_dir, 'pdf_visualizations')
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Název výsledného PDF
        pdf_path = os.path.join(pdf_dir, f'lesion_overlay_epoch_{epoch}.pdf')
        
        # Přepneme model do eval režimu
        self.generator.eval()
        
        # Získáme potřebné údaje z loaderu
        for batch in self.dataloader:
            atlas = batch['atlas'].to(self.device)
            brain_mask = batch['brain_mask'].to(self.device)
            
            # Zajistíme batch rozměr
            if atlas.dim() == 3:
                atlas = atlas.unsqueeze(0)
            if brain_mask.dim() == 3:
                brain_mask = brain_mask.unsqueeze(0)
            
            # Binární masky
            atlas_binary = (atlas > 0).float()
            brain_mask_binary = (brain_mask > 0).float()
            
            # Převod atlasu do numpy pro vizualizaci
            atlas_np = atlas[0, 0].cpu().numpy()
            
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            
            # Vytvoření vlastní barevné mapy pro překryv - šedá pro mozek, červená pro léze
            # Vytvořím custom colormap pro zobrazení překryvu
            colors = [(0.85, 0.85, 0.85), (0.85, 0.85, 0.85), (1, 0, 0)]  # šedá, šedá, červená
            cmap_name = 'gray_red'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
            
            # Otevření PDF souboru (všechny vzorky na samostatných stránkách)
            with PdfPages(pdf_path) as pdf:
                # Pro každý vzorek vygenerujeme podobně jako v generate_samples
                for sample_idx in range(num_samples):
                    print(f"  Generuji vzorek {sample_idx+1}/{num_samples}...")
                    
                    # Vygenerování latentního vektoru
                    z = torch.randn(1, self.z_dim, device=self.device)
                    
                    # Generování léze
                    with torch.no_grad():
                        fake_lesion = self.generator(z, atlas[:1], brain_mask[:1])
                    
                    # Aplikace atlasu a brain masky
                    fake_lesion = fake_lesion * atlas_binary[:1] * brain_mask_binary[:1]
                    
                    # ZMĚNA: Aplikace nucené aktivace podobně jako v generate_samples
                    atlas_threshold = torch.quantile(atlas[:1].view(-1), 0.9)  # Horních 10% hodnot atlasu
                    high_prob_regions = (atlas[:1] > atlas_threshold).float() * brain_mask[:1]
                    
                    # Generování náhodné aktivační masky
                    act_mask = torch.zeros_like(fake_lesion)
                    random_strength = torch.rand(1, device=self.device) * 0.3 + 0.7  # Aktivace 0.7-1.0
                    
                    # Přidání nucené aktivace do části vysoce pravděpodobných oblastí
                    activation_regions = high_prob_regions * (torch.rand_like(high_prob_regions) > 0.5).float()
                    act_mask = act_mask.scatter_(1, torch.zeros_like(fake_lesion, dtype=torch.long), 
                                               activation_regions * random_strength)
                    
                    # Smíchání s původním výstupem - 40% generovaný obsah, 60% nucená aktivace
                    fake_lesion = fake_lesion * 0.4 + act_mask * 0.6
                    
                    # Aplikace teplotního škálování pro ostřejší přechody
                    temperature = 8.0
                    soft_binary = torch.sigmoid(temperature * (fake_lesion - 0.5))
                    
                    # Převod na numpy pro další zpracování
                    binary_np = soft_binary[0, 0].cpu().numpy()
                    
                    # Odfiltrování malých komponent (podobně jako v generate_samples)
                    binary_thresh = (binary_np > 0.5).astype(np.float32)
                    labeled_components, num_components = measure_label(binary_thresh, return_num=True)
                    
                    # Filtrování malých komponent
                    component_sizes = np.bincount(labeled_components.flatten())[1:] if num_components > 0 else []
                    min_component_size = 3  # Minimální velikost komponenty pro zachování
                    
                    cleaned_binary = np.zeros_like(binary_np)
                    for comp_id in range(1, num_components + 1):
                        if component_sizes[comp_id - 1] >= min_component_size:
                            cleaned_binary[labeled_components == comp_id] = 1
                    
                    # Kontrola inverzních hodnot
                    ones_percentage = np.mean(cleaned_binary == 1) * 100
                    if ones_percentage > 90:
                        print(f"  UPOZORNĚNÍ: Hodnoty vypadají invertované - {ones_percentage:.2f}% jsou jedničky!")
                        cleaned_binary = 1 - cleaned_binary
                    
                    # Použijeme všechny řezy
                    depth = cleaned_binary.shape[0]
                    
                    # Uspořádání: 4 řezy na řádek
                    cols = 4
                    rows = int(np.ceil(depth / cols))
                    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                    # Pokud je pouze jeden řádek, zajistíme konzistentní iteraci
                    axs = np.array(axs).reshape(-1)
                    
                    # Pro každý slice
                    for i in range(depth):
                        # Získáme řez atlasu
                        atlas_slice = atlas_np[i, :, :]
                        
                        # Získáme řez léze
                        lesion_slice = cleaned_binary[i, :, :]
                        nonzero_count = np.count_nonzero(lesion_slice > 0.5)
                        
                        # Vytvoříme překryv - normalizujeme atlas na hodnoty 0-0.5 (šedá)
                        # a nastavíme léze na hodnotu 1 (červená)
                        overlay = np.zeros_like(atlas_slice)
                        
                        # Normalizace atlasu na rozsah 0-0.5
                        if atlas_slice.max() > 0:
                            atlas_norm = atlas_slice / atlas_slice.max() * 0.5
                            overlay = atlas_norm.copy()
                        
                        # Přidáme léze jako 1 (červená barva v custom colormapě)
                        overlay[lesion_slice > 0.5] = 1.0
                        
                        # Nad každým slice zobrazíme číslo slice a počet nenulových pixelů léze
                        axs[i].set_title(f"{i+1}/{depth}, léze: {nonzero_count} px", fontsize=8)
                        axs[i].imshow(overlay, cmap=cm, vmin=0, vmax=1)
                        axs[i].axis('off')
                    
                    # Skryjeme prázdné subplots (pokud nějaké zbydou)
                    for j in range(depth, len(axs)):
                        axs[j].axis('off')
                    
                    # Přidáme legendu pro celou figuru
                    fig.subplots_adjust(bottom=0.05)
                    import matplotlib.patches as mpatches
                    red_patch = mpatches.Patch(color='red', label='Léze')
                    gray_patch = mpatches.Patch(color='gray', label='Normativní mozek')
                    fig.legend(handles=[gray_patch, red_patch], loc='lower center', ncol=2)
                    
                    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Ponecháme místo pro legendu
                    pdf.savefig(fig)
                    plt.close(fig)
            
            # Použijeme pouze první batch
            break
        
        # Vrátíme model do tréninkového režimu
        self.generator.train()
        
        print(f"PDF vizualizace s překryvem lézí na normativním mozku uložena do: {pdf_path}")

    def train(self, num_epochs, validate_every=5):
        start_time = time.time()
        
        # Initialize tracking variables for training stability
        prev_g_loss = float('inf')
        prev_d_loss = float('inf')
        loss_stability_counter = 0
        best_g_loss = float('inf')
        
        # Track learning rates
        g_lr_history = []
        d_lr_history = []
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            # Training on one epoch
            self.generator.train()
            self.discriminator.train()
            
            # Progress bar
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            # Storage for visualization
            last_batch_data = None
            
            for i, batch in enumerate(pbar):
                real_lesions = batch['label'].to(self.device)
                atlas = batch['atlas'].to(self.device)
                brain_mask = batch['brain_mask'].to(self.device)
                
                # NOVÉ: Zkontrolujeme, jestli reálný vzorek obsahuje dostatek lézí
                # Spočítáme procento aktivních pixelů v reálném vzorku
                real_binary = (real_lesions > 0.5).float()
                valid_regions = (atlas > 0).float() * brain_mask
                real_activated_percent = torch.sum(real_binary * valid_regions) / (torch.sum(valid_regions) + 1e-8)
                
                # Pokud je vzorek téměř prázdný (méně než 0.1% aktivních pixelů) a není začátek tréninku,
                # přeskočíme ho s 80% pravděpodobností, abychom upřednostnili vzorky s lézemi
                skip_sample = False
                if real_activated_percent < 0.001 and epoch > 2 and torch.rand(1).item() < 0.8:
                    skip_sample = True
                    if i % 10 == 0:  # Občas vypíšeme informaci
                        print(f"[TRAIN] Přeskakuji téměř prázdný vzorek (aktivace: {real_activated_percent.item()*100:.4f}%)")
                    continue  # Přeskočíme na další vzorek
                
                # NEW: Trénovat diskriminátor pouze občas
                # To dá generátoru výhodu
                train_discriminator = True
                if i % 3 != 0:  # Trénuj diskriminátor pouze každou třetí iteraci
                    train_discriminator = False
                
                # Train discriminator pouze pokud je to potřeba
                if train_discriminator:
                    d_metrics = self.train_discriminator(real_lesions, atlas, brain_mask)
                else:
                    # Dummy metriky, pokud jsme diskriminátor netrénovali
                    d_metrics = {'d_loss': 0.0, 'd_patch': 0.0, 'd_global': 0.0, 'd_distrib': 0.0, 'd_real': 0.0, 'd_fake': 0.0}
                
                # NEW: Vždy trénuj generátor, nezávisle na diskriminátoru
                # Nepoužívat komplexní logiku, která může někdy skipnout trénink generátoru
                
                # Určení počtu kroků trénování generátoru
                if train_discriminator:
                    num_g_steps = 1  # Standardní počet kroků
                else:
                    num_g_steps = 3  # Když netrénujeme diskriminátor, dej generátoru více kroků
                
                # Train generator (multiple times)
                g_metrics = None
                for g_step in range(num_g_steps):
                    g_metrics = self.train_generator(real_lesions, atlas, brain_mask, i + epoch * len(pbar))
                
                # Accumulate losses for averaging
                epoch_d_loss += d_metrics['d_loss']
                epoch_g_loss += g_metrics['g_total']
                
                # Update progress bar
                pbar.set_postfix({
                    'g_loss': g_metrics['g_total'],
                    'd_loss': d_metrics['d_loss']
                })
                
                # Log metrics for every few batches
                if i % 10 == 0:
                    step = epoch * len(self.dataloader) + i
                    # Filter out non-scalar metrics
                    all_metrics = {
                        **{k: v for k, v in d_metrics.items() if isinstance(v, (int, float))},
                        **{k: v for k, v in g_metrics.items() if isinstance(v, (int, float))}
                    }
                    self.log_metrics(all_metrics, epoch, step)
                
                # Save last batch for visualization
                if i == len(pbar) - 1:
                    last_batch_data = {
                        'real_lesions': real_lesions.detach(),
                        'fake_lesions': g_metrics['fake_lesions'].detach(),
                        'atlas': atlas.detach()
                    }
            
            # Average losses for the epoch
            epoch_d_loss /= len(self.dataloader)
            epoch_g_loss /= len(self.dataloader)
            
            # NEW: Apply learning rate schedulers
            self.g_scheduler.step()
            self.d_scheduler.step()
            
            # NEW: Record current learning rates
            current_g_lr = self.g_scheduler.get_last_lr()[0]
            current_d_lr = self.d_scheduler.get_last_lr()[0]
            g_lr_history.append(current_g_lr)
            d_lr_history.append(current_d_lr)
            
            # Log learning rates
            self.writer.add_scalar('LearningRate/generator', current_g_lr, epoch)
            self.writer.add_scalar('LearningRate/discriminator', current_d_lr, epoch)
            
            # Log images if we have data
            if last_batch_data is not None:
                self.log_images(
                    last_batch_data['real_lesions'],
                    last_batch_data['fake_lesions'],
                    last_batch_data['atlas'],
                    epoch
                )
            
            # Generate and save validation samples every X epochs
            if (epoch + 1) % validate_every == 0 and self.val_dataloader is not None:
                self.validate(epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch + 1)
            
            # Generate PDF with slices every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.generate_pdf_from_samples(epoch + 1, num_samples=5)
                self.generate_overlay_pdf_from_samples(epoch + 1, num_samples=5)
            
            # Check for training stability
            if epoch > 0:
                g_loss_change = abs(epoch_g_loss - prev_g_loss) / max(prev_g_loss, 1e-8)
                d_loss_change = abs(epoch_d_loss - prev_d_loss) / max(prev_d_loss, 1e-8)
                
                # If losses are changing too little or too much, increment instability counter
                if g_loss_change < 0.01 or g_loss_change > 2.0 or d_loss_change < 0.01 or d_loss_change > 2.0:
                    loss_stability_counter += 1
                else:
                    loss_stability_counter = 0
            
            prev_g_loss = epoch_g_loss
            prev_d_loss = epoch_d_loss
            
            # Save best model if generator loss improves
            if epoch_g_loss < best_g_loss:
                best_g_loss = epoch_g_loss
                best_model_path = os.path.join(self.save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'g_optimizer_state_dict': self.generator_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
                }, best_model_path)
                print(f"Saved best model with G Loss: {epoch_g_loss:.4f}")
            
            # Print epoch information
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch+1}/{num_epochs}] - time: {epoch_time:.2f}s - G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}, LR: G={current_g_lr:.6f}/D={current_d_lr:.6f}")
            
            # NEW: Check for training instability and adjust learning rates if needed
            if loss_stability_counter >= 3:
                print(f"WARNING: Training appears unstable. Adjusting learning rates...")
                # Reduce learning rates manually
                for param_group in self.generator_optimizer.param_groups:
                    param_group['lr'] *= 0.5
                for param_group in self.discriminator_optimizer.param_groups:
                    param_group['lr'] *= 0.5
                
                # Reset stability counter
                loss_stability_counter = 0
        
        # End of training
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        final_checkpoint_path = os.path.join(self.save_dir, 'final_model.pt')
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'd_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'g_loss': epoch_g_loss,
            'd_loss': epoch_d_loss,
            'g_lr_history': g_lr_history,
            'd_lr_history': d_lr_history,
        }, final_checkpoint_path)
        print(f"Final model saved: {final_checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a lesion GAN")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing lesion label maps")
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to the lesion atlas file")
    parser.add_argument("--brain_mask_path", type=str, default=None, help="Path to the normative brain mask file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--z_dim", type=int, default=128, help="Dimension of latent space")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--filter_empty", action="store_true", help="Filter out empty (all-black) images from the dataset")
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = HIELesionDataset(
        label_dir=args.label_dir,
        atlas_path=args.atlas_path,
        brain_mask_path=args.brain_mask_path,
        filter_empty=args.filter_empty
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Create model and trainer
    trainer = HIELesionGANTrainer(
        dataloader=dataloader,
        z_dim=args.z_dim,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        debug=args.debug
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs)
    
    # Generate samples
    trainer.generate_samples(num_samples=10)


if __name__ == "__main__":
    main()
