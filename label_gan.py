import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import nibabel as nib
from torch.nn import functional as F
from pathlib import Path
import scipy.ndimage as ndimage

# Třída pro jednoduchou 3D datovou augmentaci
class Simple3DAugmentation:
    """Jednoduchá datová augmentace pro 3D objemová data."""
    
    def __init__(self, 
                 rotation_range=(-5, 5),     # Rozsah rotace ve stupních
                 shift_range=(-4, 4),        # Rozsah posunu v pixelech
                 zoom_range=(0.95, 1.05),    # Rozsah změny měřítka (zoom)
                 intensity_noise=0.05):      # Intenzita šumu (jako procento z rozsahu)
        """
        Args:
            rotation_range: Rozsah pro náhodné rotace ve stupních
            shift_range: Rozsah pro náhodné posuny v pixelech
            zoom_range: Rozsah pro náhodné změny měřítka (zoom)
            intensity_noise: Intenzita Gaussovského šumu (jako procento z rozsahu)
        """
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.intensity_noise = 0
    
    def __call__(self, tensor):
        """
        Aplikuje augmentaci na PyTorch tensor.
        
        Args:
            tensor: PyTorch tensor ve tvaru [1, D, H, W]
            
        Returns:
            Augmentovaný tensor ve stejném tvaru
        """
        # Převod na numpy array pro jednodušší manipulaci
        data = tensor.squeeze(0).cpu().numpy()  # Odstraníme kanálovou dimenzi [D, H, W]
        
        # Náhodné rotace (malé, aby zůstala anatomická struktura)
        if self.rotation_range:
            angle_x = np.random.uniform(*self.rotation_range)
            angle_y = np.random.uniform(*self.rotation_range)
            angle_z = np.random.uniform(*self.rotation_range)
            
            data = ndimage.rotate(data, angle_x, axes=(1, 2), reshape=False, order=1, mode='nearest')
            data = ndimage.rotate(data, angle_y, axes=(0, 2), reshape=False, order=1, mode='nearest')
            data = ndimage.rotate(data, angle_z, axes=(0, 1), reshape=False, order=1, mode='nearest')
        
        # Náhodné posuny
        if self.shift_range:
            shift_x = np.random.uniform(*self.shift_range)
            shift_y = np.random.uniform(*self.shift_range)
            shift_z = np.random.uniform(*self.shift_range)
            
            data = ndimage.shift(data, (shift_z, shift_y, shift_x), order=0, mode='constant')
        
        # Náhodné změny měřítka (zoom)
        if self.zoom_range:
            zoom_factor = np.random.uniform(*self.zoom_range)
            # Použijeme stejný zoom faktor pro všechny dimenze, aby se zachoval poměr stran
            data = ndimage.zoom(data, zoom_factor, order=1, mode='nearest')
            
            # Pokud změníme velikost, musíme upravit velikost na původní
            if zoom_factor != 1.0:
                shape = tensor.squeeze(0).shape
                # Ořez nebo padding, aby výsledek měl správnou velikost
                if data.shape[0] > shape[0]:
                    diff = data.shape[0] - shape[0]
                    data = data[diff//2:diff//2+shape[0], :, :]
                elif data.shape[0] < shape[0]:
                    diff = shape[0] - data.shape[0]
                    pad_width = ((diff//2, diff-diff//2), (0, 0), (0, 0))
                    data = np.pad(data, pad_width, mode='constant')
                    
                if data.shape[1] > shape[1]:
                    diff = data.shape[1] - shape[1]
                    data = data[:, diff//2:diff//2+shape[1], :]
                elif data.shape[1] < shape[1]:
                    diff = shape[1] - data.shape[1]
                    pad_width = ((0, 0), (diff//2, diff-diff//2), (0, 0))
                    data = np.pad(data, pad_width, mode='constant')
                    
                if data.shape[2] > shape[2]:
                    diff = data.shape[2] - shape[2]
                    data = data[:, :, diff//2:diff//2+shape[2]]
                elif data.shape[2] < shape[2]:
                    diff = shape[2] - data.shape[2]
                    pad_width = ((0, 0), (0, 0), (diff//2, diff-diff//2))
                    data = np.pad(data, pad_width, mode='constant')
        
        # Přidání Gaussovského šumu
        if self.intensity_noise > 0:
            noise = np.random.normal(0, self.intensity_noise, data.shape)
            data = data + noise
            # Ořízneme hodnoty zpět do rozsahu [0, 1]
            data = np.clip(data, 0, 1)
        
        # Převod zpět na PyTorch tensor
        return torch.FloatTensor(data).unsqueeze(0)  # Přidáme zpět kanálovou dimenzi [1, D, H, W]

class LabelGANDataset(Dataset):
    """Dataset pro LabelGAN, který generuje LABEL mapy HIE lézí"""
    
    def __init__(self, normal_atlas_path, label_dir, lesion_atlas_path=None, transform=None,
                 use_augmentation=False):
        """
        Args:
            normal_atlas_path: Cesta k normativnímu atlasu
            label_dir: Adresář s registrovanými LABEL mapami
            lesion_atlas_path: Volitelná cesta k atlasu frekvence lézí
            transform: Volitelné transformace
            use_augmentation: Použít datovou augmentaci
        """
        self.normal_atlas_path = normal_atlas_path
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                                  if f.endswith('.mha') or f.endswith('.nii.gz') or f.endswith('.nii')])
        self.lesion_atlas_path = lesion_atlas_path
        self.transform = transform
        self.use_augmentation = use_augmentation
        
        # Vytvoření augmentátoru s mírnými parametry, pokud je povoleno
        if self.use_augmentation:
            self.augmentor = Simple3DAugmentation(
                rotation_range=(-3, 3),    # Malé rotace max 3 stupně
                shift_range=(-2, 2),       # Malé posuny max 2 pixely
                zoom_range=(0.98, 1.02),   # Malé změny velikosti (±2%)
                intensity_noise=0.02       # Malý šum (2%)
            )
        
        # Načtení normativního atlasu
        if self.normal_atlas_path.endswith('.nii.gz') or self.normal_atlas_path.endswith('.nii'):
            self.normal_atlas = nib.load(normal_atlas_path).get_fdata()
        else:
            self.normal_atlas = sitk.GetArrayFromImage(sitk.ReadImage(normal_atlas_path))
        
        # Normalizace do rozsahu [0, 1]
        self.normal_atlas = (self.normal_atlas - self.normal_atlas.min()) / (self.normal_atlas.max() - self.normal_atlas.min())
        
        # Načtení atlasu frekvence lézí, pokud je dostupný
        if self.lesion_atlas_path:
            if self.lesion_atlas_path.endswith('.nii.gz') or self.lesion_atlas_path.endswith('.nii'):
                self.lesion_atlas = nib.load(self.lesion_atlas_path).get_fdata()
            else:
                self.lesion_atlas = sitk.GetArrayFromImage(sitk.ReadImage(self.lesion_atlas_path))
            
            # Normalizace do rozsahu [0, 1]
            self.lesion_atlas = (self.lesion_atlas - self.lesion_atlas.min()) / (self.lesion_atlas.max() - self.lesion_atlas.min())
        else:
            self.lesion_atlas = None
        
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        # Načtení LABEL mapy
        if self.label_files[idx].endswith('.nii.gz') or self.label_files[idx].endswith('.nii'):
            label = nib.load(self.label_files[idx]).get_fdata()
        else:
            label = sitk.GetArrayFromImage(sitk.ReadImage(self.label_files[idx]))
            
        # Binarizace LABEL mapy (zajistíme, že jsou jen hodnoty 0 a 1)
        label = (label > 0).astype(np.float32)
        
        # Převod na PyTorch tensory
        normal_atlas = torch.FloatTensor(self.normal_atlas).unsqueeze(0)  # Přidání kanálové dimenze
        label = torch.FloatTensor(label).unsqueeze(0)  # Přidání kanálové dimenze
        
        # Aplikace datové augmentace, pokud je povolena
        if self.use_augmentation:
            # Pro každý vzorek vybereme s 50% pravděpodobností, zda budeme augmentovat
            # Tímto zajistíme, že v datasetu budou jak originální, tak augmentované vzorky
            if np.random.random() > 0.5:
                # Použijeme stejnou augmentaci pro atlas i label, aby zůstaly zarovnané
                # Nejprve vytvoříme parametry augmentace
                normal_atlas = self.augmentor(normal_atlas)
                label = self.augmentor(label)  # Pro label používáme stejnou transformaci
                
                # Zajistíme, že label zůstává binární
                label = (label > 0.5).float()
        
        if self.transform:
            normal_atlas = self.transform(normal_atlas)
            label = self.transform(label)
        
        return_dict = {
            'normal_atlas': normal_atlas,
            'label': label
        }
        
        # Přidání atlasu frekvence lézí, pokud je k dispozici
        if self.lesion_atlas is not None:
            lesion_atlas = torch.FloatTensor(self.lesion_atlas).unsqueeze(0)
            if self.transform:
                lesion_atlas = self.transform(lesion_atlas)
            return_dict['lesion_atlas'] = lesion_atlas
            
        return return_dict

class LabelGenerator(nn.Module):
    """Generator pro syntézu LABEL map HIE lézí"""
    
    def __init__(self, in_channels=2, features=64, dropout_rate=0.3, use_self_attention=False):
        """
        Args:
            in_channels: Počet vstupních kanálů (normativní atlas + šum)
            features: Počet základních feature map
            dropout_rate: Míra dropout v generátoru
            use_self_attention: Použít self-attention mechanismus pro lepší detaily
        """
        super(LabelGenerator, self).__init__()
        
        self.use_self_attention = use_self_attention
        
        # U-Net architektura
        # Encoder
        self.enc1 = nn.Conv3d(in_channels, features, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv3d(features, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*2)
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv3d(features*2, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*4)
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv3d(features*4, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*8)
        )
        
        # Self-attention vrstvy (volitelné)
        if use_self_attention:
            self.self_attn = SelfAttention3D(features*8)
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*8, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*4),
            nn.Dropout3d(dropout_rate)
        )
        self.dec2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*4*2, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*2),
            nn.Dropout3d(dropout_rate)
        )
        self.dec3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*2*2, features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features),
            nn.Dropout3d(dropout_rate)
        )
        
        # Výstupní vrstva pro LABEL mapy
        self.label_output = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*2, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Výstup v rozsahu [0, 1] pro LABEL
        )
        
    def forward(self, x, noise, lesion_atlas=None):
        """
        Args:
            x: Normativní atlas (B, 1, D, H, W)
            noise: Náhodný šum (B, 1, D, H, W)
            lesion_atlas: Volitelný atlas frekvence lézí (B, 1, D, H, W)
        Returns:
            LABEL mapa
        """
        # Spojení vstupu a šumu (a případně atlasu lézí)
        if lesion_atlas is not None:
            # Pokud máme atlas lézí, vážíme šum podle něj
            weighted_noise = noise * lesion_atlas
            x = torch.cat([x, weighted_noise], dim=1)
        else:
            x = torch.cat([x, noise], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Self-attention (volitelné)
        if self.use_self_attention:
            e4 = self.self_attn(e4)
        
        # Decoder s skip connections
        d1 = self.dec1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        
        # Generování LABEL mapy
        label_map = self.label_output(d3)
        
        return label_map

# Self-Attention mechanismus pro lepší detaily
class SelfAttention3D(nn.Module):
    """Self-attention mechanismus pro 3D data"""
    def __init__(self, in_dim):
        super(SelfAttention3D, self).__init__()
        self.query_conv = nn.Conv3d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        
        # Flatten prostorové dimenze
        proj_query = self.query_conv(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, D * H * W)
        
        # Výpočet attention mapy
        attention = F.softmax(torch.bmm(proj_query, proj_key), dim=2)
        
        proj_value = self.value_conv(x).view(batch_size, -1, D * H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)
        
        # Gamma je naučitelný parametr, který určuje váhu self-attention
        out = self.gamma * out + x
        return out

class LabelDiscriminator(nn.Module):
    """Discriminator pro rozlišení reálných a syntetických LABEL map"""
    
    def __init__(self, in_channels=2, features=64, use_spectral_norm=False):
        """
        Args:
            in_channels: Počet vstupních kanálů (normativní atlas + LABEL mapa)
            features: Počet základních feature map
            use_spectral_norm: Použít spektrální normalizaci pro stabilnější trénink
        """
        super(LabelDiscriminator, self).__init__()
        
        # Použití spektrální normalizace pro stabilnější trénink
        if use_spectral_norm:
            norm_layer = lambda x: nn.utils.spectral_norm(x)
        else:
            norm_layer = lambda x: x
        
        self.model = nn.Sequential(
            # První blok bez normalizace
            norm_layer(nn.Conv3d(in_channels, features, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            
            # Další bloky
            norm_layer(nn.Conv3d(features, features*2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm3d(features*2),
            nn.LeakyReLU(0.2),
            
            norm_layer(nn.Conv3d(features*2, features*4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm3d(features*4),
            nn.LeakyReLU(0.2),
            
            norm_layer(nn.Conv3d(features*4, features*8, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm3d(features*8),
            nn.LeakyReLU(0.2),
            
            # Výstupní vrstva
            norm_layer(nn.Conv3d(features*8, 1, kernel_size=4, stride=1, padding=0))
        )
    
    def forward(self, atlas, label):
        """
        Args:
            atlas: Normativní atlas (B, 1, D, H, W)
            label: Reálná nebo syntetická LABEL mapa (B, 1, D, H, W)
        Returns:
            Skóre pravděpodobnosti (B, 1, D', H', W')
        """
        # Spojení normativního atlasu a LABEL
        x = torch.cat([atlas, label], dim=1)
        return self.model(x)

class DiceLoss(nn.Module):
    """Dice loss pro segmentační úlohy"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

def save_sample(data, path):
    """Uložení vzorku jako .nii.gz soubor"""
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, path)

# Gradient penalty pro WGAN-GP
def compute_gradient_penalty(discriminator, real_samples, fake_samples, atlas):
    """
    Výpočet gradient penalty pro WGAN-GP.
    
    Args:
        discriminator: Model diskriminátoru
        real_samples: Reálné LABEL mapy
        fake_samples: Generované LABEL mapy
        atlas: Normativní atlas
        
    Returns:
        Hodnota gradient penalty
    """
    device = real_samples.device
    batch_size = real_samples.size(0)
    
    # Náhodné váhy pro interpolaci mezi reálnými a generovanými vzorky
    alpha = torch.rand(batch_size, 1, 1, 1, 1).to(device)
    
    # Interpolace vzorků
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Skóre diskriminátoru pro interpolované vzorky
    d_interpolated = discriminator(atlas, interpolated)
    
    # Fake výstupy pro výpočet gradientů
    fake = torch.ones(d_interpolated.size()).to(device)
    
    # Výpočet gradientů
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Výpočet L2 normy gradientů
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # Návrat gradient penalty
    return ((gradient_norm - 1) ** 2).mean()

def train_label_gan(args):
    """Hlavní trénovací funkce pro LabelGAN model"""
    
    # Nastavení zařízení
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Načtení datasetu s augmentací
    dataset = LabelGANDataset(
        normal_atlas_path=args.normal_atlas_path,
        label_dir=args.label_dir,
        lesion_atlas_path=args.lesion_atlas_path,
        use_augmentation=args.use_augmentation  # Nový parametr pro zapnutí augmentace
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Inicializace modelů
    generator = LabelGenerator(
        in_channels=2,  # Atlas + šum
        features=args.generator_filters,
        dropout_rate=args.dropout_rate,
        use_self_attention=args.use_self_attention
    ).to(device)
    
    discriminator = LabelDiscriminator(
        in_channels=2,  # Atlas + LABEL
        features=args.discriminator_filters,
        use_spectral_norm=args.use_spectral_norm
    ).to(device)
    
    # Optimizátory
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    # Decay learning rate - přidáme schedulery
    milestones = [args.epochs // 3, 2 * args.epochs // 3]  # Milníky pro snížení learning rate
    g_scheduler = optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=milestones, gamma=0.5)
    d_scheduler = optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=milestones, gamma=0.5)
    
    # Loss funkce
    adversarial_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()  # Pro lepší segmentaci lézí
    
    # Trénovací smyčka
    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            # Přesun dat na správné zařízení
            normal_atlas = batch['normal_atlas'].to(device)
            real_label = batch['label'].to(device)
            
            # Příprava lesion_atlas, pokud je k dispozici
            lesion_atlas = batch.get('lesion_atlas', None)
            if lesion_atlas is not None:
                lesion_atlas = lesion_atlas.to(device)
            
            # Vytvoření náhodného šumu
            batch_size = normal_atlas.size(0)
            noise = torch.randn_like(normal_atlas).to(device)
            
            # -------------------------
            # Trénink diskriminátoru
            # -------------------------
            for _ in range(args.n_critic):  # Trénujeme diskriminator vícekrát za jeden krok generátoru
                d_optimizer.zero_grad()
                
                # Reálné vzorky
                real_output = discriminator(normal_atlas, real_label)
                real_labels = torch.ones_like(real_output)
                d_real_loss = adversarial_loss(real_output, real_labels)
                
                # Generované vzorky
                fake_label = generator(normal_atlas, noise, lesion_atlas)
                fake_output = discriminator(normal_atlas, fake_label.detach())
                fake_labels = torch.zeros_like(fake_output)
                d_fake_loss = adversarial_loss(fake_output, fake_labels)
                
                # Gradient penalty
                gradient_penalty = 0
                if args.use_gradient_penalty:
                    gp_weight = args.gradient_penalty_weight
                    gradient_penalty = compute_gradient_penalty(
                        discriminator, real_label, fake_label.detach(), normal_atlas
                    ) * gp_weight
                
                # Celková ztráta diskriminátoru
                d_loss = d_real_loss + d_fake_loss + gradient_penalty
                d_loss.backward()
                d_optimizer.step()
            
            # -------------------------
            # Trénink generátoru
            # -------------------------
            g_optimizer.zero_grad()
            
            # Nové generování vzorků pro G krok (pro lepší stabilitu)
            noise = torch.randn_like(normal_atlas).to(device)
            fake_label = generator(normal_atlas, noise, lesion_atlas)
            
            # Adversarial loss
            fake_output = discriminator(normal_atlas, fake_label)
            g_adv_loss = adversarial_loss(fake_output, real_labels)
            
            # Dice loss pro segmentaci lézí
            g_dice_loss = dice_loss(fake_label, real_label)
            
            # Přidáváme anatomicky konzistentní loss
            anatomically_informed_loss = 0.0
            if lesion_atlas is not None:
                threshold = 0.2
                low_probability_mask = (lesion_atlas < threshold).float()
                anatomically_informed_loss = (fake_label * low_probability_mask).mean() * 10.0
            
            # Celková ztráta generátoru
            g_loss = (
                g_adv_loss +
                args.lambda_dice * g_dice_loss +
                anatomically_informed_loss
            )
            g_loss.backward()
            g_optimizer.step()
            
            # Výpis stavu tréninku
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Batch {i}/{len(dataloader)}, "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                      f"ADV: {g_adv_loss.item():.4f}, DICE: {g_dice_loss.item():.4f}, "
                      f"LR: {g_optimizer.param_groups[0]['lr']:.6f}")
        
        # Aktualizace learning rate na konci každé epochy
        g_scheduler.step()
        d_scheduler.step()
        
        # Uložení checkpointu
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(args.output_dir, f'labelgan_checkpoint_epoch{epoch}.pt'))
            
            # Uložení vygenerovaného vzorku
            with torch.no_grad():
                sample_atlas = normal_atlas[0:1]
                sample_noise = torch.randn_like(sample_atlas).to(device)
                
                if lesion_atlas is not None:
                    sample_lesion_atlas = lesion_atlas[0:1]
                    sample_fake_label = generator(sample_atlas, sample_noise, sample_lesion_atlas)
                else:
                    sample_fake_label = generator(sample_atlas, sample_noise)
                
                # Uložení vzorků jako .nii.gz soubory
                sample_path = os.path.join(args.output_dir, f'labelgan_sample_epoch{epoch}')
                save_sample(sample_fake_label[0, 0].cpu().numpy(), f"{sample_path}_fake_label.nii.gz")
                save_sample(sample_atlas[0, 0].cpu().numpy(), f"{sample_path}_atlas.nii.gz")
                save_sample(real_label[0, 0].cpu().numpy(), f"{sample_path}_real_label.nii.gz")
    
    print("LabelGAN Training completed!")

def generate_label_samples(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.normal_atlas_path.endswith('.nii.gz') or args.normal_atlas_path.endswith('.nii'):
        normal_atlas = nib.load(args.normal_atlas_path).get_fdata()
    else:
        normal_atlas = sitk.GetArrayFromImage(sitk.ReadImage(args.normal_atlas_path))
    normal_atlas = (normal_atlas - normal_atlas.min()) / (normal_atlas.max() - normal_atlas.min())
    normal_atlas = torch.FloatTensor(normal_atlas).unsqueeze(0).unsqueeze(0).to(device)
    
    lesion_atlas = None
    if args.lesion_atlas_path:
        if args.lesion_atlas_path.endswith('.nii.gz') or args.lesion_atlas_path.endswith('.nii'):
            lesion_atlas = nib.load(args.lesion_atlas_path).get_fdata()
        else:
            lesion_atlas = sitk.GetArrayFromImage(sitk.ReadImage(args.lesion_atlas_path))
        lesion_atlas = (lesion_atlas - lesion_atlas.min()) / (lesion_atlas.max() - lesion_atlas.min())
        lesion_atlas = torch.FloatTensor(lesion_atlas).unsqueeze(0).unsqueeze(0).to(device)
    
    generator = LabelGenerator(
        in_channels=2,
        features=args.generator_filters,
        dropout_rate=args.dropout_rate,
        use_self_attention=args.use_self_attention
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    label_output_dir = os.path.join(args.output_dir, "label")
    os.makedirs(label_output_dir, exist_ok=True)
    
    generated_labels = []
    for i in range(args.num_samples):
        with torch.no_grad():
            noise = torch.randn_like(normal_atlas).to(device)
            if lesion_atlas is not None:
                fake_label = generator(normal_atlas, noise, lesion_atlas)
            else:
                fake_label = generator(normal_atlas, noise)
            fake_label_binary = (fake_label > 0.5).float()
            label_np = fake_label_binary[0, 0].cpu().numpy()  # [D, H, W]
            generated_labels.append(label_np)
            label_output_path = os.path.join(label_output_dir, f'sample_{i}_label.nii.gz')
            save_sample(label_np, label_output_path)
    
    if args.vizualize_generated:
        pdf_output_path = os.path.join(args.output_dir, "generated_labels.pdf")
        visualize_all_generated_labels(generated_labels, pdf_output_path)
    
    print(f"Generated {args.num_samples} LABEL maps")
    print(f"LABEL maps saved to: {label_output_dir}")
    
    
def visualize_all_generated_labels(generated_labels, output_pdf_path, slices_per_row=5):
    """
    Vytvoří jeden PDF dokument s vizualizací všech generovaných LABEL map.
    Každá LABEL mapa je zobrazena na samostatných stránkách, přičemž každá stránka obsahuje
    mřížku řezu (slices_per_row řezu na řádek).
    
    Args:
        generated_labels: List numpy polí s vygenerovanými LABEL mapami, každé s tvarem [D, H, W]
        output_pdf_path: Cesta k uložení výsledného PDF
        slices_per_row: Počet řezu na řádek
    """
    with PdfPages(output_pdf_path) as pdf:
        for idx, label_np in enumerate(generated_labels):
            D, H, W = label_np.shape
            n_rows = int(np.ceil(D / slices_per_row))
            fig = plt.figure(figsize=(slices_per_row*3, n_rows*3))
            gs = gridspec.GridSpec(n_rows, slices_per_row)
            for slice_idx in range(D):
                row = slice_idx // slices_per_row
                col = slice_idx % slices_per_row
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(label_np[slice_idx, :, :], cmap='gray')
                ax.set_title(f"Sample {idx+1}, Slice {slice_idx+1}")
                ax.axis('off')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Všechny generované LABEL mapy byly uloženy do PDF: {output_pdf_path}")

def main():
    """Hlavní funkce skriptu"""
    parser = argparse.ArgumentParser(description="LabelGAN for HIE Lesion LABEL Map Synthesis")
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Společné argumenty
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--normal_atlas_path', type=str, required=True,
                              help='Cesta k normativnímu atlasu')
    parent_parser.add_argument('--lesion_atlas_path', type=str, default=None,
                              help='Cesta k atlasu frekvence lézí (volitelné)')
    parent_parser.add_argument('--output_dir', type=str, default='./output',
                              help='Výstupní adresář pro uložení modelů a vzorků')
    parent_parser.add_argument('--generator_filters', type=int, default=64,
                              help='Počet základních filtrů generátoru')
    parent_parser.add_argument('--discriminator_filters', type=int, default=64,
                              help='Počet základních filtrů diskriminátoru')
    parent_parser.add_argument('--dropout_rate', type=float, default=0.3,
                              help='Míra dropout v generátoru')
    parent_parser.add_argument('--use_self_attention', action='store_true',
                              help='Použít self-attention mechanismus v generátoru')
    parent_parser.add_argument('--use_spectral_norm', action='store_true',
                              help='Použít spektrální normalizaci v diskriminátoru')
    parent_parser.add_argument('--latent_dim', type=int, default=128,
                              help='Dimenze latentního prostoru (pro budoucí rozšíření)')
    
    # Parser pro trénink
    train_parser = subparsers.add_parser('train', parents=[parent_parser],
                                        help='Trénink LabelGAN modelu')
    train_parser.add_argument('--label_dir', type=str, required=True,
                             help='Adresář s registrovanými LABEL mapami')
    train_parser.add_argument('--batch_size', type=int, default=2,
                             help='Velikost dávky pro trénink')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Počet epoch tréninku')
    train_parser.add_argument('--lr', type=float, default=0.0002,
                             help='Learning rate')
    train_parser.add_argument('--beta1', type=float, default=0.5,
                             help='Beta1 parametr pro Adam optimizér')
    train_parser.add_argument('--beta2', type=float, default=0.999,
                             help='Beta2 parametr pro Adam optimizér')
    train_parser.add_argument('--lambda_dice', type=float, default=50.0,
                             help='Váha pro Dice loss (segmentace lézí)')
    # Nové parametry pro vylepšení stability
    train_parser.add_argument('--use_augmentation', action='store_true',
                              help='Použít mírnou datovou augmentaci pro zvýšení variability vzorků')
    train_parser.add_argument('--use_gradient_penalty', action='store_true',
                              help='Použít gradient penalty pro stabilnější trénink')
    train_parser.add_argument('--gradient_penalty_weight', type=float, default=10.0,
                              help='Váha pro gradient penalty')
    train_parser.add_argument('--n_critic', type=int, default=1,
                              help='Počet kroků diskriminátoru na jeden krok generátoru')
    
    # Parser pro generování
    generate_parser = subparsers.add_parser('generate', parents=[parent_parser],
                                           help='Generování vzorků LABEL map pomocí natrénovaného modelu')
    generate_parser.add_argument('--checkpoint_path', type=str, required=True,
                                help='Cesta k checkpointu modelu')
    generate_parser.add_argument('--num_samples', type=int, default=10,
                                help='Počet vzorků k vygenerování')
    generate_parser.add_argument('--vizualize_generated', action='store_true',
                                help='Vytvoří PDF s vizualizací vygenerovaných LABEL map')
    
    args = parser.parse_args()
    
    if args.action == 'train':
        train_label_gan(args)
    elif args.action == 'generate':
        generate_label_samples(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 