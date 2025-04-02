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

class HIELesionDataset(Dataset):
    def __init__(self, label_dir, lesion_atlas_path, transform=None):
        """
        Dataset pro HIE léze.
        
        Args:
            label_dir (str): Cesta k adresáři s registrovanými LABEL mapami
            lesion_atlas_path (str): Cesta k souboru s atlasem lézí
            transform: Transformace, které mají být aplikovány na data
        """
        self.label_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir) 
                           if f.endswith('.nii') or f.endswith('.nii.gz')]
        self.lesion_atlas_path = lesion_atlas_path
        self.transform = transform
        
        # Načtení lesion atlasu
        lesion_atlas_nib = nib.load(lesion_atlas_path)
        self.lesion_atlas = torch.from_numpy(lesion_atlas_nib.get_fdata()).float()
        
        # Normalizace atlasu do rozsahu [0, 1]
        if self.lesion_atlas.max() > 0:
            self.lesion_atlas = self.lesion_atlas / self.lesion_atlas.max()
    
    def __len__(self):
        return len(self.label_paths)
    
    def __getitem__(self, idx):
        # Načtení LABEL mapy
        label_path = self.label_paths[idx]
        label_nib = nib.load(label_path)
        label = torch.from_numpy(label_nib.get_fdata()).float()
        
        # Binarizace labelu (pokud není již binární)
        label = (label > 0).float()
        
        # Aplikace transformací
        if self.transform:
            sample = {'label': label, 'atlas': self.lesion_atlas}
            sample = self.transform(sample)
            label = sample['label']
        
        # Změna dimenzí pro pytorch (N, C, D, H, W)
        label = label.unsqueeze(0)  # Přidání kanálové dimenze
        atlas = self.lesion_atlas.unsqueeze(0)
        
        return {
            'label': label,
            'atlas': atlas,
            'path': label_path
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
        
        # Bottleneck s noise injection a concatenation
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
        self.dec1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(base_filters * 2, output_channels, 4, 2, 1), use_spectral_norm),
            nn.Sigmoid()  # Výstup v rozsahu [0, 1] pro binární segmentaci
        )
        
        # Atlas integration
        self.atlas_attention = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters, base_filters, 3, 1, 1), use_spectral_norm),
            nn.InstanceNorm3d(base_filters),
            nn.Sigmoid()
        )
        
    def forward(self, z, atlas):
        batch_size = atlas.size(0)
        
        # Encoder - atlas
        e1 = self.atlas_enc1(atlas)  # 64x64x32
        e2 = self.atlas_enc2(e1)     # 32x32x16
        e3 = self.atlas_enc3(e2)     # 16x16x8
        e4 = self.atlas_enc4(e3)     # 8x8x4
        
        # Debug: Print shape to understand the expected shape for z_3d
        # print(f"e4 shape: {e4.shape}")
        
        # Zpracování náhodného šumu a reshape na 3D feature mapu
        z_flat = self.fc_z(z)
        
        # Shape fix: Get the actual shape from e4 to ensure compatibility
        _, _, D, H, W = e4.shape
        z_3d = z_flat.view(batch_size, self.base_filters, D, H, W)
        z_3d = self.reshape_z(z_3d)
        
        # Kombinace atlasu a šumu v bottlenecku
        combined = torch.cat([e4, z_3d], dim=1)
        bottleneck = self.bottleneck_conv(combined)
        
        # Self-attention
        attended = self.self_attn(bottleneck)
        
        # Decoder s skip connections
        d4 = self.dec4(attended)                           # 16x16x8
        d4_skip = torch.cat([d4, e3], dim=1)
        
        d3 = self.dec3(d4_skip)                            # 32x32x16
        d3_skip = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3_skip)                            # 64x64x32
        d2_skip = torch.cat([d2, e1], dim=1)
        
        # Atlas-based attention mapa pro zdůraznění relevantních oblastí
        atlas_attention = self.atlas_attention(d2)
        d2_skip = d2_skip * atlas_attention + d2_skip
        
        # Finální dekódování do lesion mapy
        out = self.dec1(d2_skip)                           # 128x128x64
        
        # Ensure lesions only appear where atlas is non-zero
        atlas_mask = (atlas > 0).float()
        out = out * atlas_mask
        
        return out


class Discriminator(nn.Module):
    def __init__(self, input_channels=2, base_filters=64, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        
        # Vstup je konkatenovaný atlas a label/generovaná léze
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv3d(input_channels, base_filters, 4, 2, 1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv3d(base_filters, base_filters * 2, 4, 2, 1), use_spectral_norm),
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
        
        # Pro PatchGAN - lokální realism posouzení
        self.layer5 = spectral_norm(nn.Conv3d(base_filters * 8, 1, 4, 1, 1), use_spectral_norm)
        
        # Pro global decision - celkové posouzení věrohodnosti
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = spectral_norm(nn.Linear(base_filters * 8, 1), use_spectral_norm)
        
    def forward(self, x, atlas):
        # Konkatenace vstupu a atlasu podél kanálové dimenze
        x = torch.cat([x, atlas], dim=1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Self-attention
        x = self.self_attn(x)
        
        x = self.layer4(x)
        
        # PatchGAN výstup - mapa skóre
        patch_out = self.layer5(x)
        
        # Globální posouzení
        global_features = self.global_pool(x).view(x.size(0), -1)
        global_out = self.fc(global_features)
        
        return patch_out, global_out


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
            if target_is_real:
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


# Tréninková utilita s implementací tréninkové smyčky
class HIELesionGANTrainer:
    def __init__(self, 
                 generator, 
                 discriminator, 
                 device, 
                 dataloader,
                 val_dataloader=None,
                 z_dim=128,
                 lr_g=0.0002,
                 lr_d=0.0002,
                 beta1=0.5,
                 beta2=0.999,
                 lambda_adv=1.0,
                 lambda_fm=10.0,
                 lambda_dice=10.0,
                 lambda_l1=10.0,
                 label_smoothing=0.9,
                 log_dir='./logs',
                 save_dir='./checkpoints',
                 save_interval=10):
        
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.z_dim = z_dim
        
        # Optimizéry
        self.optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
        
        # Lambdy pro vážení složek loss funkce
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_dice = lambda_dice
        self.lambda_l1 = lambda_l1
        self.label_smoothing = label_smoothing
        
        # Loss funkce
        self.gan_loss = GANLoss(device, gan_mode='hinge')
        self.dice_loss = DiceLoss()
        self.l1_loss = nn.L1Loss()
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2)
        
        # Adresáře pro ukládání
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.save_interval = save_interval
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))
        
        # Vytvoření adresářů, pokud neexistují
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
    def get_random_z(self, batch_size):
        return torch.randn(batch_size, self.z_dim).to(self.device)
    
    def train_discriminator(self, real_lesions, atlas):
        batch_size = real_lesions.size(0)
        
        # Reset gradientů
        self.optimizer_d.zero_grad()
        
        # Forward pass s reálnými lézemi
        real_patch_pred, real_global_pred = self.discriminator(real_lesions, atlas)
        
        # Generování fake lézí
        z = self.get_random_z(batch_size)
        fake_lesions = self.generator(z, atlas)
        
        # Forward pass s falešnými lézemi
        fake_patch_pred, fake_global_pred = self.discriminator(fake_lesions.detach(), atlas)
        
        # Real a Fake loss
        d_real_loss = self.gan_loss(real_patch_pred, True) + self.gan_loss(real_global_pred, True)
        d_fake_loss = self.gan_loss(fake_patch_pred, False) + self.gan_loss(fake_global_pred, False)
        
        # Celková loss
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        
        # Backpropagation
        d_loss.backward()
        self.optimizer_d.step()
        
        return {
            'd_loss': d_loss.item(),
            'd_real_loss': d_real_loss.item(),
            'd_fake_loss': d_fake_loss.item()
        }
    
    def train_generator(self, real_lesions, atlas):
        batch_size = real_lesions.size(0)
        
        # Reset gradientů
        self.optimizer_g.zero_grad()
        
        # Generování fake lézí
        z = self.get_random_z(batch_size)
        fake_lesions = self.generator(z, atlas)
        
        # Ensure generated lesions respect the atlas - make atlas constraint more explicit
        atlas_mask = (atlas > 0).float()
        fake_lesions = fake_lesions * atlas_mask  # Enforce constraint again for safety
        
        # Forward pass diskriminátorem
        fake_patch_pred, fake_global_pred = self.discriminator(fake_lesions, atlas)
        
        # Adversarial loss
        g_adv_loss = self.gan_loss(fake_patch_pred, True) + self.gan_loss(fake_global_pred, True)
        
        # Další pomocné loss funkce pro lepší generování
        
        # Atlas-guided Focal Loss - podporuje generování v oblastech, kde se léze častěji vyskytují
        # Scale the loss by the atlas values to emphasize areas with higher lesion probability
        atlas_guided_focal = self.focal_loss(fake_lesions * atlas, real_lesions * atlas)
        
        # L1 Loss pro stabilitu tréninku a zachycení nízkofrekvečních detailů
        # Only compute L1 loss in atlas-positive regions to avoid penalizing negative areas
        masked_l1_loss = self.l1_loss(fake_lesions * atlas_mask, real_lesions * atlas_mask)
        
        # Dice Loss pro lepší segmentace
        g_dice_loss = self.dice_loss(fake_lesions, real_lesions)
        
        # Atlas violation penalty - strongly discourage generating lesions where atlas is 0
        atlas_zero_mask = (1.0 - atlas_mask)
        violation_penalty = torch.mean(fake_lesions * atlas_zero_mask) * 10.0  # Higher weight for violations
        
        # Kombinace všech loss funkcí
        g_loss = (self.lambda_adv * g_adv_loss + 
                 self.lambda_l1 * masked_l1_loss + 
                 self.lambda_dice * g_dice_loss + 
                 atlas_guided_focal + 
                 violation_penalty)
        
        # Backpropagation
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'g_loss': g_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_l1_loss': masked_l1_loss.item(),
            'g_dice_loss': g_dice_loss.item(),
            'g_focal_loss': atlas_guided_focal.item(),
            'atlas_violation': violation_penalty.item(),
            'fake_lesions': fake_lesions.detach()
        }
    
    def validate(self):
        if self.val_dataloader is None:
            return {}
        
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {
            'val_g_loss': 0,
            'val_g_adv_loss': 0,
            'val_g_l1_loss': 0,
            'val_g_dice_loss': 0,
            'val_atlas_violation': 0,
            'val_d_loss': 0
        }
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                real_lesions = batch['label'].to(self.device)
                atlas = batch['atlas'].to(self.device)
                batch_size = real_lesions.size(0)
                
                # Generování fake lézí
                z = self.get_random_z(batch_size)
                fake_lesions = self.generator(z, atlas)
                
                # Apply atlas mask
                atlas_mask = (atlas > 0).float()
                fake_lesions = fake_lesions * atlas_mask
                
                # Diskriminátor výstupy
                real_patch_pred, real_global_pred = self.discriminator(real_lesions, atlas)
                fake_patch_pred, fake_global_pred = self.discriminator(fake_lesions, atlas)
                
                # Adversarial Loss
                g_adv_loss = self.gan_loss(fake_patch_pred, True) + self.gan_loss(fake_global_pred, True)
                
                # L1 Loss only in atlas-positive regions
                masked_l1_loss = self.l1_loss(fake_lesions * atlas_mask, real_lesions * atlas_mask)
                
                # Dice Loss
                g_dice_loss = self.dice_loss(fake_lesions, real_lesions)
                
                # Atlas violation penalty
                atlas_zero_mask = (1.0 - atlas_mask)
                violation_penalty = torch.mean(fake_lesions * atlas_zero_mask) * 10.0
                
                # Kombinace
                g_loss = (self.lambda_adv * g_adv_loss + 
                         self.lambda_l1 * masked_l1_loss + 
                         self.lambda_dice * g_dice_loss +
                         violation_penalty)
                
                # Diskriminátor loss
                d_real_loss = self.gan_loss(real_patch_pred, True) + self.gan_loss(real_global_pred, True)
                d_fake_loss = self.gan_loss(fake_patch_pred, False) + self.gan_loss(fake_global_pred, False)
                d_loss = (d_real_loss + d_fake_loss) * 0.5
                
                val_losses['val_g_loss'] += g_loss.item()
                val_losses['val_g_adv_loss'] += g_adv_loss.item()
                val_losses['val_g_l1_loss'] += masked_l1_loss.item()
                val_losses['val_g_dice_loss'] += g_dice_loss.item()
                val_losses['val_atlas_violation'] += violation_penalty.item()
                val_losses['val_d_loss'] += d_loss.item()
        
        # Průměrování
        num_batches = len(self.val_dataloader)
        for k in val_losses.keys():
            val_losses[k] /= num_batches
        
        self.generator.train()
        self.discriminator.train()
        
        return val_losses
    
    def log_metrics(self, metrics, epoch, step=None):
        # Log do tensorboard
        prefix = ''
        if step is not None:
            prefix = f'step_{step}/'
        
        for key, value in metrics.items():
            self.writer.add_scalar(f'{prefix}{key}', value, epoch)
    
    def log_images(self, real_lesions, fake_lesions, atlas, epoch, n_samples=4):
        with torch.no_grad():
            # Výběr n_samples vzorků
            real_samples = real_lesions[:n_samples].cpu().detach()
            fake_samples = fake_lesions[:n_samples].cpu().detach()
            atlas_samples = atlas[:n_samples].cpu().detach()
            
            # Funkce pro vizualizaci středových řezů
            def visualize_slice(volume, slice_idx=None):
                if slice_idx is None:
                    slice_idx = volume.shape[2] // 2  # Střední řez podél Z
                return volume[0, :, slice_idx, :, :]  # Výběr pro jeden batch
            
            # Log středových řezů pro každý vzorek
            for i in range(n_samples):
                # Získání středového řezu
                real_slice = visualize_slice(real_samples[i:i+1])
                fake_slice = visualize_slice(fake_samples[i:i+1])
                atlas_slice = visualize_slice(atlas_samples[i:i+1])
                
                # Vytvoření gridu pro vizualizaci
                grid = torch.cat([real_slice, fake_slice, atlas_slice], dim=2)
                
                # Log do tensorboard
                self.writer.add_image(f'sample_{i}/real_fake_atlas', grid, epoch)
    
    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint uložen: {checkpoint_path}")
    
    def generate_samples(self, atlas, num_samples=10, output_dir='./generated_samples'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a directory for visualization
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Statistics dictionary
        stats = {
            'lesion_volumes': [],
            'lesion_counts': [],
            'atlas_coverage': []
        }
        
        self.generator.eval()
        with torch.no_grad():
            # Zajistíme, že atlas má batch rozměr
            if atlas.dim() == 4:  # [C, D, H, W]
                atlas = atlas.unsqueeze(0)  # [1, C, D, H, W]
            
            # Get atlas binary mask for constraint
            atlas_mask = (atlas > 0).float()
            
            # Uložení atlasu pro referenci
            atlas_np = atlas.squeeze().cpu().numpy()
            atlas_path = os.path.join(output_dir, 'atlas_reference.nii.gz')
            
            # Pokud máme informace o atlasu, použijeme stejné parametry
            if hasattr(self.dataloader.dataset, 'lesion_atlas_path'):
                atlas_nib = nib.load(self.dataloader.dataset.lesion_atlas_path)
                nib.save(nib.Nifti1Image(atlas_np, atlas_nib.affine, atlas_nib.header), atlas_path)
            
            # Vygenerování vzorků
            print(f"Generuji {num_samples} vzorků...")
            for i in range(num_samples):
                # Získání náhodného latentního vektoru
                z = self.get_random_z(1)
                
                # Generování léze
                fake_lesion = self.generator(z, atlas)
                
                # Enforce atlas constraint
                fake_lesion = fake_lesion * atlas_mask
                
                # Binarize for statistics (threshold at 0.5)
                binary_lesion = (fake_lesion > 0.5).float()
                
                # Compute statistics
                lesion_volume = binary_lesion.sum().item()
                stats['lesion_volumes'].append(lesion_volume)
                
                # Convert to numpy for connected component analysis
                binary_np = binary_lesion.squeeze().cpu().numpy()
                labeled_array, num_features = measure_label(binary_np, connectivity=3, return_num=True)
                stats['lesion_counts'].append(num_features)
                
                # Calculate atlas coverage (percentage of atlas covered by lesions)
                atlas_coverage = (binary_lesion * atlas_mask).sum() / atlas_mask.sum()
                stats['atlas_coverage'].append(atlas_coverage.item() * 100)  # as percentage
                
                # Převod na numpy array a binarizace pro uložení jako nifti
                lesion_np = fake_lesion.squeeze().cpu().numpy()
                
                # Uložení jako .nii soubor
                sample_path = os.path.join(output_dir, f'generated_lesion_{i}.nii.gz')
                
                # Vytvoření nifti objektu - použijeme stejné parametry jako u atlasu
                if hasattr(self.dataloader.dataset, 'lesion_atlas_path'):
                    # Získání header informací z atlasu
                    atlas_nib = nib.load(self.dataloader.dataset.lesion_atlas_path)
                    lesion_nib = nib.Nifti1Image(lesion_np, atlas_nib.affine, atlas_nib.header)
                else:
                    # Pokud nemáme informace o atlasu, vytvoříme základní nifti
                    lesion_nib = nib.Nifti1Image(lesion_np, np.eye(4))
                
                # Uložení
                nib.save(lesion_nib, sample_path)
                
                # Create visualization - mid-slice for 3 planes
                vis_path = os.path.join(vis_dir, f'viz_lesion_{i}.png')
                
                # Create mid slices for visualization
                D, H, W = lesion_np.shape
                mid_z = D // 2
                mid_y = H // 2
                mid_x = W // 2
                
                # Create figure with 3 subplots for 3 orthogonal planes
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Axial view
                axes[0].imshow(lesion_np[mid_z, :, :], cmap='jet')
                axes[0].set_title(f"Axial (z={mid_z})")
                axes[0].imshow(atlas_np[mid_z, :, :], cmap='gray', alpha=0.3)
                
                # Coronal view
                axes[1].imshow(lesion_np[:, mid_y, :], cmap='jet')
                axes[1].set_title(f"Coronal (y={mid_y})")
                axes[1].imshow(atlas_np[:, mid_y, :], cmap='gray', alpha=0.3)
                
                # Sagittal view
                axes[2].imshow(lesion_np[:, :, mid_x], cmap='jet')
                axes[2].set_title(f"Sagittal (x={mid_x})")
                axes[2].imshow(atlas_np[:, :, mid_x], cmap='gray', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(vis_path)
                plt.close()
                
                print(f"Vygenerován vzorek: {sample_path}")
                print(f"  - Počet lézí: {num_features}")
                print(f"  - Objem lézí: {lesion_volume} voxelů")
                print(f"  - Pokrytí atlasu: {atlas_coverage.item() * 100:.2f}%")
        
        # Create a summary statistics file
        stats_path = os.path.join(output_dir, 'lesion_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("=== Statistiky generovaných lézí ===\n\n")
            f.write(f"Počet vygenerovaných vzorků: {num_samples}\n\n")
            
            avg_vol = sum(stats['lesion_volumes']) / num_samples
            avg_count = sum(stats['lesion_counts']) / num_samples
            avg_coverage = sum(stats['atlas_coverage']) / num_samples
            
            f.write(f"Průměrný počet lézí na vzorek: {avg_count:.2f}\n")
            f.write(f"Průměrný objem lézí na vzorek: {avg_vol:.2f} voxelů\n")
            f.write(f"Průměrné pokrytí atlasu: {avg_coverage:.2f}%\n\n")
            
            f.write("Detaily jednotlivých vzorků:\n")
            for i in range(num_samples):
                f.write(f"Vzorek {i}:\n")
                f.write(f"  - Počet lézí: {stats['lesion_counts'][i]}\n")
                f.write(f"  - Objem lézí: {stats['lesion_volumes'][i]} voxelů\n")
                f.write(f"  - Pokrytí atlasu: {stats['atlas_coverage'][i]:.2f}%\n\n")
        
        print(f"\nStatistiky byly uloženy do: {stats_path}")
        print(f"Vizualizace byly uloženy do: {vis_dir}")
        
        self.generator.train()
    
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
                
                # Trénink diskriminátoru
                d_metrics = self.train_discriminator(real_lesions, atlas)
                
                # Trénink generátoru
                g_metrics = self.train_generator(real_lesions, atlas)
                
                # Akumulace ztrát pro průměrování
                epoch_d_loss += d_metrics['d_loss']
                epoch_g_loss += g_metrics['g_loss']
                
                # Aktualizace progress baru
                pbar.set_postfix({
                    'g_loss': g_metrics['g_loss'],
                    'd_loss': d_metrics['d_loss']
                })
                
                # Logování metrik pro každý batch
                if i % 10 == 0:
                    step = epoch * len(self.dataloader) + i
                    all_metrics = {**d_metrics, **{k: v for k, v in g_metrics.items() if k != 'fake_lesions'}}
                    self.log_metrics(all_metrics, epoch, step)
            
            # Průměrování ztrát za epoch
            epoch_d_loss /= len(self.dataloader)
            epoch_g_loss /= len(self.dataloader)
            
            # Logování obrazů
            self.log_images(real_lesions, g_metrics['fake_lesions'], atlas, epoch)
            
            # Validace
            if (epoch + 1) % validate_every == 0 and self.val_dataloader is not None:
                val_metrics = self.validate()
                self.log_metrics(val_metrics, epoch)
                print(f"Validace (Epoch {epoch+1}): G Loss: {val_metrics['val_g_loss']:.4f}, D Loss: {val_metrics['val_d_loss']:.4f}")
            
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
    parser = argparse.ArgumentParser(description='HIE Lesion GAN Training')
    parser.add_argument('--label_dir', type=str, required=True, 
                        help='Directory containing registered LABEL maps')
    parser.add_argument('--lesion_atlas_path', type=str, required=True, 
                        help='Path to the lesion atlas file')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Output directory for models and results')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of training epochs')
    parser.add_argument('--lr_g', type=float, default=0.0001, 
                        help='Learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=0.0004, 
                        help='Learning rate for discriminator')
    parser.add_argument('--z_dim', type=int, default=128, 
                        help='Dimension of latent space')
    parser.add_argument('--base_filters', type=int, default=64, 
                        help='Number of base filters in networks')
    parser.add_argument('--val_split', type=float, default=0.1, 
                        help='Fraction of data to use for validation')
    parser.add_argument('--use_spectral_norm', action='store_true', 
                        help='Use spectral normalization')
    parser.add_argument('--save_interval', type=int, default=10, 
                        help='Save model every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate samples from a trained model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model for generation')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Nastavení seedy pro reprodukovatelnost
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Vytvoření výstupních adresářů
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    save_dir = os.path.join(args.output_dir, 'checkpoints')
    samples_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Vytvoření transformace pro augmentaci
    transform = RandomAugmentation3D(rotation_range=10, flip_prob=0.5)
    
    # Načtení datasetu
    dataset = HIELesionDataset(args.label_dir, args.lesion_atlas_path, transform=transform)
    print(f"Načteno {len(dataset)} lesion map.")
    
    # Rozdělení na trénovací a validační data
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    ) if val_size > 0 else None
    
    # Inicializace modelů
    device = torch.device(args.device)
    generator = Generator(
        z_dim=args.z_dim,
        atlas_channels=1,
        output_channels=1,
        base_filters=args.base_filters,
        use_spectral_norm=args.use_spectral_norm
    )
    
    discriminator = Discriminator(
        input_channels=2,  # Lesion + Atlas
        base_filters=args.base_filters,
        use_spectral_norm=args.use_spectral_norm
    )
    
    # Pokud máme cestu k natrénovanému modelu, načteme jej
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        if 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        print(f"Model načten z: {args.model_path}")
    
    # Inicializace tréninkové třídy
    trainer = HIELesionGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        dataloader=train_loader,
        val_dataloader=val_loader,
        z_dim=args.z_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        log_dir=log_dir,
        save_dir=save_dir,
        save_interval=args.save_interval
    )
    
    # Pokud chceme pouze generovat vzorky
    if args.generate_only:
        if args.model_path is None:
            print("Pro generování je potřeba specifikovat cestu k natrénovanému modelu (--model_path)")
            return
        
        # Nahrání atlasu
        atlas = dataset.lesion_atlas.unsqueeze(0).to(device)
        
        # Generování vzorků
        print(f"Generuji {args.num_samples} vzorků...")
        trainer.generate_samples(atlas, num_samples=args.num_samples, output_dir=samples_dir)
        print(f"Vzorky byly uloženy do: {samples_dir}")
    else:
        # Trénink modelu
        print(f"Začínám trénink na {len(train_loader)} batches, validuji na {len(val_loader) if val_loader else 0} batches.")
        print(f"Velikost batche: {args.batch_size}, Počet epoch: {args.epochs}")
        print(f"Zařízení: {device}, Základní počet filtrů: {args.base_filters}")
        
        trainer.train(num_epochs=args.epochs)
        
        # Generování několika ukázkových vzorků po tréninku
        atlas = dataset.lesion_atlas.unsqueeze(0).to(device)
        trainer.generate_samples(atlas, num_samples=args.num_samples, output_dir=samples_dir)
        print(f"Trénink dokončen. Vzorky byly uloženy do: {samples_dir}")


if __name__ == "__main__":
    main()
