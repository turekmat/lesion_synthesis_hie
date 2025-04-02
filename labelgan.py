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

class HIELesionDataset(Dataset):
    def __init__(self, 
                 label_dir, 
                 atlas_path=None, 
                 brain_mask_path=None,
                 transform=None, 
                 use_random_affine=False, 
                 use_elastic_transform=False,
                 intensity_noise=0.1):
        self.label_dir = label_dir
        self.transform = transform
        self.use_random_affine = use_random_affine
        self.use_elastic_transform = use_elastic_transform
        self.intensity_noise = intensity_noise
        
        # Load all label file paths
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) 
                               if f.endswith('.nii.gz') or f.endswith('.nii')])
        
        print(f"Found {len(self.label_files)} lesion label files in {label_dir}")
        
        # Load lesion probability atlas if provided
        self.atlas = None
        if atlas_path and os.path.exists(atlas_path):
            print(f"Loading lesion atlas from {atlas_path}")
            atlas_nii = nib.load(atlas_path)
            self.atlas = atlas_nii.get_fdata()
            # Normalize atlas to [0, 1] if needed
            if self.atlas.max() > 1.0:
                self.atlas = self.atlas / self.atlas.max()
            
            # Binary atlas representation (any non-zero value is a potential lesion location)
            self.atlas_binary = (self.atlas > 0).astype(np.float32)
            
            print(f"Atlas shape: {self.atlas.shape}, min: {self.atlas.min()}, max: {self.atlas.max()}")
        else:
            print("No lesion atlas provided or file not found")
        
        # Load normative brain mask if provided
        self.brain_mask = None
        if brain_mask_path and os.path.exists(brain_mask_path):
            print(f"Loading brain mask from {brain_mask_path}")
            brain_mask_nii = nib.load(brain_mask_path)
            self.brain_mask = brain_mask_nii.get_fdata()
            
            # Ensure brain mask is binary
            self.brain_mask = (self.brain_mask > 0).astype(np.float32)
            
            print(f"Brain mask shape: {self.brain_mask.shape}, sum: {np.sum(self.brain_mask)}")
        else:
            print("No brain mask provided or file not found")
        
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        # Load label
        label_path = self.label_files[idx]
        label_nii = nib.load(label_path)
        label = label_nii.get_fdata()
        
        # Ensure label is binary
        label = (label > 0).astype(np.float32)
        
        # Convert to tensor and add channel dimension
        label_tensor = torch.from_numpy(label).float().unsqueeze(0)
        
        # Prepare atlas if available
        if self.atlas is not None:
            atlas_tensor = torch.from_numpy(self.atlas).float().unsqueeze(0)
        else:
            # If no atlas, create a dummy tensor of ones with the same shape as the label
            atlas_tensor = torch.ones_like(label_tensor)
        
        # Prepare brain mask if available
        if self.brain_mask is not None:
            brain_mask_tensor = torch.from_numpy(self.brain_mask).float().unsqueeze(0)
        else:
            # If no brain mask, create a dummy tensor of ones with the same shape as the label
            brain_mask_tensor = torch.ones_like(label_tensor)
        
        # Apply transformations if needed
        if self.transform:
            label_tensor = self.transform(label_tensor)
            atlas_tensor = self.transform(atlas_tensor)
            brain_mask_tensor = self.transform(brain_mask_tensor)
        
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
        
        # Náhodná rotace
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            for z in range(label.shape[0]):
                label_slice = label[z, :, :]
                label[z, :, :] = transforms.functional.rotate(
                    label_slice.unsqueeze(0), angle).squeeze(0)
        
        # Náhodný flip
        if random.random() < self.flip_prob:
            label = torch.flip(label, [1])  # Flip horizontálně
        
        if random.random() < self.flip_prob:
            label = torch.flip(label, [2])  # Flip vertikálně
        
        return {'label': label, 'atlas': atlas}


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
        # Operates on d2_skip (base_filters*2 channels)
        self.atlas_attention = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters * 2 + atlas_channels, base_filters * 2, 3, 1, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(base_filters * 2, base_filters * 2, 3, 1, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters * 2),
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
        batch_size = atlas.size(0)
        
        if self.debug:
            print(f"Input atlas shape: {atlas.shape}, z shape: {z.shape}")
            if brain_mask is not None:
                print(f"Brain mask shape: {brain_mask.shape}")
        
        # Encoder - atlas
        e1 = self.atlas_enc1(atlas)  # 64x64x32
        e2 = self.atlas_enc2(e1)     # 32x32x16
        e3 = self.atlas_enc3(e2)     # 16x16x8
        e4 = self.atlas_enc4(e3)     # 8x8x4
        
        if self.debug:
            print(f"Encoder outputs - e1: {e1.shape}, e2: {e2.shape}, e3: {e3.shape}, e4: {e4.shape}")
        
        # Zpracování náhodného šumu a reshape na 3D feature mapu
        z_flat = self.fc_z(z)
        
        # Shape fix: Get the actual shape from e4 to ensure compatibility
        _, _, D, H, W = e4.shape
        z_3d = z_flat.view(batch_size, self.base_filters, D, H, W)
        z_3d = self.reshape_z(z_3d)
        
        if self.debug:
            print(f"Latent z_3d shape: {z_3d.shape}")
        
        # Kombinace atlasu a šumu v bottlenecku
        combined = torch.cat([e4, z_3d], dim=1)
        if self.debug:
            print(f"Combined bottleneck shape: {combined.shape}")
            
        bottleneck = self.bottleneck_conv(combined)
        
        # Self-attention
        attended = self.self_attn(bottleneck)
        if self.debug:
            print(f"After self-attention: {attended.shape}")
        
        # Decoder s skip connections
        d4 = self.dec4(attended)                           # 16x16x8
        if self.debug:
            print(f"d4 shape: {d4.shape}, e3 shape: {e3.shape}")
            
        d4_skip = torch.cat([d4, e3], dim=1)
        if self.debug:
            print(f"d4_skip shape: {d4_skip.shape}")
        
        d3 = self.dec3(d4_skip)                            # 32x32x16
        if self.debug:
            print(f"d3 shape: {d3.shape}, e2 shape: {e2.shape}")
            
        d3_skip = torch.cat([d3, e2], dim=1)
        if self.debug:
            print(f"d3_skip shape: {d3_skip.shape}")
        
        d2 = self.dec2(d3_skip)                            # 64x64x32
        if self.debug:
            print(f"d2 shape: {d2.shape}, e1 shape: {e1.shape}")
            
        d2_skip = torch.cat([d2, e1], dim=1)               # Concatenate to get [batch, base_filters*2, D, H, W]
        if self.debug:
            print(f"d2_skip shape (before attention): {d2_skip.shape}")
        
        # Resize atlas for attention if needed
        if d2_skip.shape[2:] != atlas.shape[2:]:
            atlas_resized = F.interpolate(atlas, size=d2_skip.shape[2:], mode='trilinear', align_corners=False)
        else:
            atlas_resized = atlas
            
        # Probabilistic atlas-based attention - using continuous atlas values, not just binary
        # Concatenate feature maps with atlas to guide the attention mechanism
        attention_input = torch.cat([d2_skip, atlas_resized], dim=1)
        atlas_attention = self.atlas_attention(attention_input)
        
        if self.debug:
            print(f"atlas_attention shape: {atlas_attention.shape}")
            
        # Apply attention with residual connection
        d2_skip = d2_skip * atlas_attention + d2_skip
        
        # Finální dekódování do lesion mapy
        logits = self.dec1(d2_skip)                       # 128x128x64
        if self.debug:
            print(f"Logits output shape: {logits.shape}")
        
        # Apply hard sigmoid with temperature scaling to get sharper binary-like outputs
        # This makes the gradient more abrupt around 0, pushing values towards 0 or 1
        temperature = 20.0  # Higher value = sharper transition
        out = torch.sigmoid(logits * temperature)
        
        # Resize atlas to match output size if needed
        if out.shape[2:] != atlas.shape[2:]:
            atlas_final = F.interpolate(atlas, size=out.shape[2:], mode='trilinear', align_corners=False)
        else:
            atlas_final = atlas
            
        # Apply atlas modulation - enhance probability in regions with higher atlas values
        atlas_mod = self.atlas_modulation(atlas_final)
        
        # Blend the generated output with the atlas probability
        # This makes lesions more likely in high-probability regions
        out = out * (1.0 + atlas_mod)
        
        # Binarize the output to ensure truly binary masks during inference
        # During training we use the continuous version for gradient flow
        if not self.training:
            out = (out > 0.5).float()
        
        if self.debug:
            print(f"Output shape (before masking): {out.shape}")
        
        # Ensure lesions only appear where atlas is non-zero (binary constraint)
        atlas_mask = (atlas > 0).float()
        
        # Ensure atlas_mask has the same spatial dimensions as the output
        if out.shape[2:] != atlas.shape[2:]:
            atlas_mask = F.interpolate(atlas_mask, size=out.shape[2:], mode='nearest')
            if self.debug:
                print(f"Resized atlas_mask to match output: {atlas_mask.shape}")
        
        out = out * atlas_mask
        
        # Apply brain mask constraint if provided
        if brain_mask is not None:
            # Resize brain mask if necessary
            if out.shape[2:] != brain_mask.shape[2:]:
                brain_mask_resized = F.interpolate(brain_mask, size=out.shape[2:], mode='nearest')
                if self.debug:
                    print(f"Resized brain_mask to match output: {brain_mask_resized.shape}")
            else:
                brain_mask_resized = brain_mask
                
            # Apply brain mask to ensure lesions are only inside the brain
            out = out * brain_mask_resized
            
        if self.debug:
            print(f"Final output shape: {out.shape}")
        
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
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(AtlasGuidedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets, atlas):
        # Ensure dimensions match
        if inputs.shape[2:] != atlas.shape[2:]:
            atlas = F.interpolate(atlas, size=inputs.shape[2:], mode='trilinear', align_corners=False)
        
        # Regular focal loss term
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_weight = self.alpha * (1-pt)**self.gamma
        
        # Atlas-guided modulation - increase weight in high probability atlas regions
        # Normalize atlas values for weighting
        atlas_norm = atlas / (atlas.mean() + 1e-8)
        
        # Scale atlas influence
        atlas_scale = 2.0
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
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.z_dim = z_dim
        self.save_dir = save_dir
        self.validate_every = validate_every
        self.save_interval = save_interval
        
        # Set debug mode
        self.debug = debug
        
        # Create model components
        self.generator = Generator(z_dim=z_dim, atlas_channels=1, output_channels=1, base_filters=64)
        self.discriminator = Discriminator(input_channels=2, atlas_channels=1, base_filters=64)
        
        # Move models to device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # Set debug flags
        self.generator.debug = debug
        self.discriminator.debug = debug
        
        # Initialize optimizers
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=learning_rate, 
            betas=(beta1, beta2)
        )
        
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        
        # Initialize loss functions
        self.adversarial_loss = GANLoss(self.device, gan_mode='hinge')
        self.l1_loss = nn.L1Loss()
        self.dice_loss = DiceLoss()
        
        # Atlas-weighted loss functions
        self.atlas_weighted_l1 = AtlasWeightedLoss(nn.L1Loss(), reduction='mean')
        self.atlas_weighted_dice = AtlasWeightedLoss(DiceLoss(), reduction='mean')
        self.atlas_focal_loss = AtlasGuidedFocalLoss(alpha=0.75, gamma=2.0)
        self.atlas_distribution_loss = AtlasDistributionLoss()
        
        # Loss weights
        self.adv_loss_weight = 1.0
        self.l1_loss_weight = 10.0
        self.weighted_l1_weight = 5.0
        self.dice_loss_weight = 10.0
        self.focal_loss_weight = 2.0
        self.distribution_loss_weight = 1.0
        self.violation_penalty_weight = 5.0
        
        # Setup TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
        
        # Create necessary directories
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
        
        # Define helper function for debug printing
        self.debug_print = self.create_debug_printer()
        
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
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                real_lesions = batch['label'].to(self.device)
                atlas = batch['atlas'].to(self.device)
                brain_mask = batch['brain_mask'].to(self.device)
                
                batch_size = real_lesions.size(0)
                num_batches += 1
                
                # Generate fake lesions
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                fake_lesions = self.generator(z, atlas, brain_mask)
                
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
                
                # Accumulate validation losses
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
        if hasattr(self, 'writer') and self.writer is not None and len(real_lesions) > 0:
            with torch.no_grad():
                self.log_images(real_lesions, fake_lesions, atlas, epoch)
        
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
            
            # Set random seed for reproducibility
            if use_fixed_z:
                torch.manual_seed(42)
            
            # Generate samples
            with open(stats_path, 'w') as f:
                f.write("Statistiky generovaných lézí\n")
                f.write("==========================\n\n")
                
                # Generate samples
                for i in range(num_samples):
                    # Generate latent vectors
                    z = torch.randn(1, self.z_dim, device=self.device)
                    
                    # Generate lesions
                    with torch.no_grad():
                        fake_lesion = self.generator(z, atlas[:1], brain_mask[:1])
                    
                    # Apply atlas mask to ensure lesions only appear in atlas-positive regions
                    fake_lesion = fake_lesion * atlas_binary[:1]
                    
                    # Apply brain mask to ensure lesions are only within the brain
                    fake_lesion = fake_lesion * brain_mask_binary[:1]
                    
                    # Ensure lesions are truly binary (0 or 1)
                    binary_lesion = (fake_lesion > 0.5).float()
                    
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

    def train_discriminator(self, real_lesions, atlas):
        batch_size = real_lesions.size(0)
        
        # Reset discriminator gradients
        self.discriminator_optimizer.zero_grad()
        
        # Get brain mask from the class that contains atlas
        if hasattr(self.dataloader.dataset, 'brain_mask') and self.dataloader.dataset.brain_mask is not None:
            brain_mask = torch.from_numpy(self.dataloader.dataset.brain_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            brain_mask = torch.ones_like(atlas)
        
        # Debug prints if needed
        if self.debug:
            self.debug_print("Real lesions in train_discriminator", real_lesions)
            self.debug_print("Atlas in train_discriminator", atlas)
        
        # Real lesion forward pass
        real_patch_pred, real_global_pred, real_distribution_pred = self.discriminator(real_lesions, atlas)
        
        # Generate fake lesions
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        
        # Generate fake lesions
        with torch.no_grad():  # Do not compute gradients for generator during discriminator step
            fake_lesions = self.generator(z, atlas, brain_mask)
        
        # Debug print if needed
        if self.debug:
            self.debug_print("Fake lesions in train_discriminator", fake_lesions)
        
        # Fake lesion forward pass
        fake_patch_pred, fake_global_pred, fake_distribution_pred = self.discriminator(fake_lesions.detach(), atlas)
        
        # Adversarial loss with label smoothing (using soft labels for real: 0.9 instead of 1.0)
        real_patch_labels = torch.ones_like(real_patch_pred) * 0.9
        real_global_labels = torch.ones_like(real_global_pred) * 0.9
        real_dist_labels = torch.ones_like(real_distribution_pred) * 0.9
        
        fake_patch_labels = torch.zeros_like(fake_patch_pred) * 0.1
        fake_global_labels = torch.zeros_like(fake_global_pred) * 0.1
        fake_dist_labels = torch.zeros_like(fake_distribution_pred) * 0.1
        
        # Compute discriminator losses for real samples
        d_real_patch_loss = self.adversarial_loss(real_patch_pred, real_patch_labels)
        d_real_global_loss = self.adversarial_loss(real_global_pred, real_global_labels)
        d_real_dist_loss = self.adversarial_loss(real_distribution_pred, real_dist_labels)
        
        # Compute discriminator losses for fake samples
        d_fake_patch_loss = self.adversarial_loss(fake_patch_pred, fake_patch_labels)
        d_fake_global_loss = self.adversarial_loss(fake_global_pred, fake_global_labels)
        d_fake_dist_loss = self.adversarial_loss(fake_distribution_pred, fake_dist_labels)
        
        # Total discriminator losses
        d_real_loss = (d_real_patch_loss + d_real_global_loss + d_real_dist_loss) / 3.0
        d_fake_loss = (d_fake_patch_loss + d_fake_global_loss + d_fake_dist_loss) / 3.0
        
        # Total loss
        d_loss = (d_real_loss + d_fake_loss) / 2.0
        
        # Backpropagation
        d_loss.backward()
        self.discriminator_optimizer.step()
        
        # Return losses for logging
        return {
            'd_loss': d_loss.item(),
            'd_real_loss': d_real_loss.item(),
            'd_fake_loss': d_fake_loss.item()
        }

    def train_generator(self, real_lesions, atlas, brain_mask, iteration):
        batch_size = real_lesions.size(0)
        
        # Reset generator gradients
        self.generator_optimizer.zero_grad()
        
        # Generate latent vector z
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        
        # Generate fake lesions using the Generator
        fake_lesions = self.generator(z, atlas, brain_mask)
        
        # Debug prints if needed
        if self.debug:
            self.debug_print("Real lesions in train_generator", real_lesions)
            self.debug_print("Fake lesions in train_generator", fake_lesions)
            self.debug_print("Atlas in train_generator", atlas)
        
        # Discriminator forward pass on fake lesions
        fake_patch_pred, fake_global_pred, fake_distribution_pred = self.discriminator(fake_lesions, atlas)
        
        # Adversarial loss with label smoothing (using soft labels for real: 0.9 instead of 1.0)
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
        
        # Backpropagation
        g_loss.backward()
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
            'fake_lesions': fake_lesions.detach()  # Add this for visualization
        }

    def train(self, num_epochs, validate_every=5):
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            # Trénink na jedné epoše
            self.generator.train()
            self.discriminator.train()
            
            # Progress bar
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for i, batch in enumerate(pbar):
                real_lesions = batch['label'].to(self.device)
                atlas = batch['atlas'].to(self.device)
                brain_mask = batch['brain_mask'].to(self.device)
                
                # Trénink diskriminátoru
                d_metrics = self.train_discriminator(real_lesions, atlas)
                
                # Trénink generátoru
                g_metrics = self.train_generator(real_lesions, atlas, brain_mask, i)
                
                # Akumulace ztrát pro průměrování
                epoch_d_loss += d_metrics['d_loss']
                epoch_g_loss += g_metrics['g_total']
                
                # Aktualizace progress baru
                pbar.set_postfix({
                    'g_loss': g_metrics['g_total'],
                    'd_loss': d_metrics['d_loss']
                })
                
                # Logování metrik pro každý batch
                if i % 10 == 0:
                    step = epoch * len(self.dataloader) + i
                    all_metrics = {**d_metrics, **{k: v for k, v in g_metrics.items() if k != 'g_atlas_violation'}}
                    self.log_metrics(all_metrics, epoch, step)
            
            # Průměrování ztrát za epoch
            epoch_d_loss /= len(self.dataloader)
            epoch_g_loss /= len(self.dataloader)
            
            # Logování obrazů
            self.log_images(real_lesions, g_metrics['fake_lesions'], atlas, epoch)
            
            # Generate and save validation samples every X epochs
            if (epoch + 1) % validate_every == 0 and self.val_dataloader is not None:
                self.validate(epoch)
            
            # Uložení checkpointu
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch + 1)
            
            # Výpis informací o epoše
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch+1}/{num_epochs}] - čas: {epoch_time:.2f}s - G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}")
        
        # Konec tréninku
        total_time = time.time() - start_time
        print(f"Trénink dokončen za {total_time:.2f} sekund")
        
        # Uložení finálního modelu
        final_checkpoint_path = os.path.join(self.save_dir, 'final_model.pt')
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, final_checkpoint_path)
        print(f"Finální model uložen: {final_checkpoint_path}")

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
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = HIELesionDataset(
        label_dir=args.label_dir,
        atlas_path=args.atlas_path,
        brain_mask_path=args.brain_mask_path
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
