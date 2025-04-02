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
        
        # Výstupní vrstva pro LABEL mapy - upravená pro lepší aktivaci lézí
        self.label_output = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*2, 1, kernel_size=4, stride=2, padding=1),
            # Nepoužíváme standardní sigmoid, ale upravený pro lepší aktivaci lézí
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
        
        # Generování LABEL mapy s upravený aktivací pro lepší detekci lézí
        raw_output = self.label_output(d3)
        
        # Upravená aktivační funkce - místo standardního sigmoidu používáme 
        # upravenou verzi, která má vyšší pravděpodobnost vytvoření lézí
        label_map = torch.sigmoid(raw_output * 1.2)  # Zvýšená citlivost pro léze
        
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

# Přidáváme vylepšenou třídu pro vážený Dice loss, který lépe zvládá extrémní nevyváženost tříd
class WeightedDiceLoss(nn.Module):
    """Vylepšená vážená Dice loss pro lépe zvládání extrémní nevyváženosti tříd"""
    def __init__(self, smooth=1.0, pos_weight=5.0, bg_weight=0.1):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.pos_weight = pos_weight  # Váha pro léze (pozitivní třída)
        self.bg_weight = bg_weight    # Váha pro pozadí (negativní třída)
        
    def forward(self, pred, target):
        # Zploštíme predikce a cílové hodnoty
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Vytvoříme masky pro pozitivní a negativní třídu
        pos_mask = (target_flat > 0.5).float()
        neg_mask = 1.0 - pos_mask
        
        # Výpočet vážené intersekce a unie pro pozitivní třídu (léze)
        pos_pred = pred_flat * pos_mask
        pos_target = target_flat * pos_mask
        pos_intersection = (pos_pred * pos_target).sum()
        pos_union = pos_pred.sum() + pos_target.sum() + self.smooth
        pos_dice = (2. * pos_intersection + self.smooth) / pos_union
        
        # Výpočet vážené intersekce a unie pro negativní třídu (pozadí)
        neg_pred = pred_flat * neg_mask
        neg_target = target_flat * neg_mask
        neg_intersection = (neg_pred * neg_target).sum()
        neg_union = neg_pred.sum() + neg_target.sum() + self.smooth
        neg_dice = (2. * neg_intersection + self.smooth) / neg_union
        
        # Kombinace pozitivní a negativní Dice s vahami
        weighted_dice = self.pos_weight * pos_dice + self.bg_weight * neg_dice
        weighted_dice = weighted_dice / (self.pos_weight + self.bg_weight)
        
        # Místo přímého vrácení (1 - dice), použijeme -log(dice) pro stabilnější gradienty
        # Tato úprava způsobí, že ztráta nikdy není příliš blízko nule a poskytuje lepší signál
        # pro trénink v případech s extrémní nevyvážeností tříd
        log_dice_loss = -torch.log(torch.clamp(weighted_dice, min=0.001, max=0.999))
        
        return log_dice_loss

# Přidáváme třídu pro Focal loss - lepší pro nevyvážené třídy
class FocalLoss(nn.Module):
    """Focal loss pro nevyvážené třídy"""
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Trestná funkce pro situace, kdy model generuje jen černé pixely
class BlackImagePenaltyLoss(nn.Module):
    """Penalizuje prázdné/černé výstupy"""
    def __init__(self, threshold=0.01, target_percentage=0.01):
        super(BlackImagePenaltyLoss, self).__init__()
        self.threshold = threshold
        self.target_percentage = target_percentage
        
    def forward(self, pred):
        # Procento pixelů nad prahem
        activated_percentage = (pred > self.threshold).float().mean()
        
        # Penalizace když je aktivních pixelů méně než cílové procento
        penalty = torch.relu(self.target_percentage - activated_percentage)
        return penalty * 10.0  # Silná penalizace

# Vylepšená trestná funkce s dodatečnými kontrolami pro generování lézí
class EnhancedLesionPenaltyLoss(nn.Module):
    """Vylepšená penalizace pro kontrolu generování lézí"""
    def __init__(self, min_threshold=0.01, max_threshold=0.5, 
                 target_min_percentage=0.005, target_max_percentage=0.03,
                 component_weights=None):
        super(EnhancedLesionPenaltyLoss, self).__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.target_min_percentage = target_min_percentage
        self.target_max_percentage = target_max_percentage
        
        # Váhy pro různé složky ztráty
        self.component_weights = {
            'min_penalty': 15.0,  # Penalizace za příliš málo lézí
            'max_penalty': 5.0,   # Penalizace za příliš mnoho lézí 
            'continuity': 5.0,    # Penalizace za nespojité léze
            'size_var': 7.0       # Penalizace za příliš uniformní velikost lézí
        }
        
        # Přepíšeme výchozí váhy, pokud jsou zadané
        if component_weights is not None:
            for key, value in component_weights.items():
                if key in self.component_weights:
                    self.component_weights[key] = value
        
    def forward(self, pred):
        batch_size = pred.size(0)
        total_loss = 0.0
        
        for b in range(batch_size):
            # Pro každý vzorek v batchi
            sample = pred[b, 0]  # Vzorek má tvar [D, H, W]
            
            # 1. Kontrola minimálního procenta lézí
            activated_percentage = (sample > self.min_threshold).float().mean()
            min_penalty = torch.relu(self.target_min_percentage - activated_percentage)
            total_loss += min_penalty * self.component_weights['min_penalty']
            
            # 2. Kontrola maximálního procenta lézí
            high_activation = (sample > self.max_threshold).float().mean()
            max_penalty = torch.relu(high_activation - self.target_max_percentage)
            total_loss += max_penalty * self.component_weights['max_penalty']
            
            # 3. Kontrola spojitosti lézí (léze by měly být spojité, ne roztroušené jednotlivé pixely)
            # Detekujeme léze s použitím středního prahu
            lesion_mask = (sample > 0.3).float()
            
            # Pokud máme léze, kontrolujeme jejich spojitost pomocí gradientů
            if lesion_mask.sum() > 0:
                # Výpočet gradientů v každém směru (d, h, w) - vyšší gradienty znamenají méně spojité léze
                d_grad = torch.abs(sample[1:, :, :] - sample[:-1, :, :]).mean()
                h_grad = torch.abs(sample[:, 1:, :] - sample[:, :-1, :]).mean()
                w_grad = torch.abs(sample[:, :, 1:] - sample[:, :, :-1]).mean()
                
                # Průměrný gradient - vyšší hodnota znamená méně spojité léze
                avg_grad = (d_grad + h_grad + w_grad) / 3.0
                continuity_penalty = torch.min(avg_grad, torch.tensor(1.0, device=pred.device))
                total_loss += continuity_penalty * self.component_weights['continuity']
            
            # 4. Variabilita velikosti lézí - chceme různě velké léze
            # Pokud máme dostatek aktivovaných pixelů, kontrolujeme variabilitu
            if activated_percentage > 0.001:
                # Extrahujeme hodnoty aktivních pixelů
                active_values = sample[sample > self.min_threshold]
                if len(active_values) > 1:
                    # Vypočítáme variabilitu jako relativní směrodatnou odchylku
                    value_std = torch.std(active_values)
                    value_mean = torch.mean(active_values)
                    relative_std = value_std / (value_mean + 1e-6)
                    
                    # Penalizace za nízkou variabilitu (uniformní léze)
                    size_var_penalty = torch.exp(-5.0 * relative_std)
                    total_loss += size_var_penalty * self.component_weights['size_var']
        
        # Vrátíme průměrnou ztrátu přes všechny vzorky v batchi
        return total_loss / batch_size

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
    
    # Loss funkce - použijeme vylepšené verze
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    # Vážený Dice loss s vylepšenou verzí pro extrémní nevyváženost tříd
    dice_loss_fn = WeightedDiceLoss(
        pos_weight=args.pos_weight, 
        bg_weight=args.bg_weight
    )
    
    # Focal loss pro lepší zvládání nevyvážených tříd
    focal_loss_fn = FocalLoss(alpha=0.8, gamma=2.0)
    
    # Vylepšená penalizace pro kontrolu generování lézí
    lesion_penalty_loss_fn = EnhancedLesionPenaltyLoss(
        min_threshold=0.1,
        max_threshold=0.5,
        target_min_percentage=args.target_lesion_percentage,
        target_max_percentage=args.target_lesion_percentage * 3.0  # Horní limit 3x větší než minimální
    )
    
    # Penalizace prázdných výstupů - cílíme na zadané procento aktivních pixelů
    black_penalty_loss_fn = BlackImagePenaltyLoss(threshold=0.1, target_percentage=args.target_lesion_percentage)
    
    # Gradient clipování pro stabilnější trénink
    max_grad_norm = 1.0

    # Upravíme learning rate - začneme s vyšší hodnotou a postupně ji snížíme
    initial_lr = args.lr
    for param_group in g_optimizer.param_groups:
        param_group['lr'] = initial_lr
    for param_group in d_optimizer.param_groups:
        param_group['lr'] = initial_lr
    
    # Sledování statistik lézí během tréninku
    best_lesion_percentage = 0.0
    best_epoch = 0
    
    # Trénovací smyčka
    for epoch in range(args.epochs):
        # Statistiky pro tuto epochu
        epoch_lesion_percentages = []
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
            
            # Strukturovaný šum který podpoří vznik lézí - použijeme směs náhodného šumu 
            # a koncentrovaných "seedů" lézí pro podporu generování aktivních regionů
            noise = torch.randn_like(normal_atlas).to(device)
            
            # Přidáme "seedy" lézí - malé aktivní oblasti které pomáhají generátoru vytvořit léze
            # Vytvoříme 1-3 náhodné "seedy" lézí v každém vzorku
            batch_size = normal_atlas.size(0)
            for b in range(batch_size):
                num_seeds = torch.randint(1, 4, (1,)).item()  # 1-3 seedy
                for _ in range(num_seeds):
                    # Náhodná pozice v objemu
                    d, h, w = normal_atlas.shape[2:]
                    seed_d = torch.randint(0, d, (1,)).item()
                    seed_h = torch.randint(0, h, (1,)).item()
                    seed_w = torch.randint(0, w, (1,)).item()
                    
                    # Vytvoření "seedu" léze (Gaussovský blob)
                    seed_size = torch.randint(2, 5, (1,)).item()
                    d_indices = torch.arange(max(0, seed_d-seed_size), min(d, seed_d+seed_size+1)).long()
                    h_indices = torch.arange(max(0, seed_h-seed_size), min(h, seed_h+seed_size+1)).long()
                    w_indices = torch.arange(max(0, seed_w-seed_size), min(w, seed_w+seed_size+1)).long()
                    
                    for id in d_indices:
                        for ih in h_indices:
                            for iw in w_indices:
                                dist = ((id-seed_d)**2 + (ih-seed_h)**2 + (iw-seed_w)**2) ** 0.5
                                if dist <= seed_size:
                                    intensity = 2.0 * (1.0 - dist/seed_size)
                                    noise[b, 0, id, ih, iw] = intensity
            
            # Pokud máme lesion_atlas, zvýšíme šum v oblastech s vyšší pravděpodobností lézí
            if lesion_atlas is not None:
                # Vážíme šum podle pravděpodobnosti lézí
                noise = noise * (1.0 + lesion_atlas * 2.0)
            
            fake_label = generator(normal_atlas, noise, lesion_atlas)
            
            # Adversarial loss
            fake_output = discriminator(normal_atlas, fake_label)
            g_adv_loss = adversarial_loss(fake_output, real_labels)
            
            # Vážený Dice loss pro lepší segmentaci lézí
            g_dice_loss = dice_loss_fn(fake_label, real_label)
            
            # Focal loss pro ještě lepší zvládání nevyvážených tříd
            g_focal_loss = focal_loss_fn(fake_label, real_label)
            
            # Penalizace prázdných/černých výstupů
            g_black_loss = black_penalty_loss_fn(fake_label)
            
            # Přidáváme anatomicky konzistentní loss
            anatomically_informed_loss = 0.0
            if lesion_atlas is not None:
                threshold = 0.2
                low_probability_mask = (lesion_atlas < threshold).float()
                anatomically_informed_loss = (fake_label * low_probability_mask).mean() * 10.0
            
            # Vylepšená penalizace pro kontrolu generování lézí
            g_lesion_loss = lesion_penalty_loss_fn(fake_label)
            
            # Vypočítáme jednotlivé složky ztráty s jejich vahami
            weighted_adv_loss_value = g_adv_loss  # Adversarial loss váha 1.0
            weighted_dice_loss_value = args.lambda_dice * g_dice_loss
            weighted_focal_loss_value = args.lambda_focal * g_focal_loss
            weighted_black_loss_value = args.lambda_black * g_black_loss
            weighted_lesion_loss_value = args.lambda_black * g_lesion_loss
            
            # Pro lepší debugging vypisujeme všechny složky ztráty
            if i % 10 == 0:
                print(f"Loss components: ADV={weighted_adv_loss_value:.4f}, DICE={weighted_dice_loss_value:.4f}, "
                      f"FOCAL={weighted_focal_loss_value:.4f}, LESION={weighted_lesion_loss_value:.4f}, "
                      f"ANATOM={anatomically_informed_loss:.4f}")
            
            # Celková ztráta jako součet všech složek
            g_loss = (
                weighted_adv_loss_value +
                weighted_dice_loss_value +
                weighted_focal_loss_value +
                weighted_lesion_loss_value +
                anatomically_informed_loss
            )
            
            # Aplikujeme normalizaci, aby ztráta nebyla příliš velká a nezpůsobovala nestabilní gradienty
            # To pomůže zlepšit trénink a konvergenci
            if g_loss > 100:
                g_loss_factor = 100 / g_loss.item()
                g_loss = g_loss * g_loss_factor
                
            g_loss.backward()
            
            # Gradient clipping pro lepší stabilitu tréninku
            nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
            
            g_optimizer.step()
            
            # Výpočet statistik léze
            with torch.no_grad():
                lesion_percentage = (fake_label > 0.5).float().mean().item() * 100.0
                epoch_lesion_percentages.append(lesion_percentage)
            
            # Výpočet skutečného DICE skóre (ne ztráty) pro monitorování tréninku
            with torch.no_grad():
                # Binarizace predikce a ground truth
                pred_binary = (fake_label > 0.5).float()
                target_binary = (real_label > 0.5).float()
                
                # Flatten pro výpočet DICE
                pred_flat = pred_binary.view(-1)
                target_flat = target_binary.view(-1)
                
                # Výpočet DICE koeficientu
                intersection = (pred_flat * target_flat).sum()
                dice_score = (2. * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)
            
            # Výpis stavu tréninku
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Batch {i}/{len(dataloader)}, "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                      f"DICE Score: {dice_score.item():.4f}, "  # Skutečný DICE koeficient (vyšší je lepší)
                      f"Lesion %: {lesion_percentage:.2f}%, "
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
        
        # Aktualizace statistik lézí
        lesion_percentage = (fake_label > 0.5).float().mean().item()
        epoch_lesion_percentages.append(lesion_percentage)
        
        # Aktualizace nejlepšího modelu
        if lesion_percentage > best_lesion_percentage:
            best_lesion_percentage = lesion_percentage
            best_epoch = epoch
    
    print("LabelGAN Training completed!")
    print(f"Best lesion percentage: {best_lesion_percentage:.4f} at epoch {best_epoch}")

def generate_label_samples(args):
    """Generování vzorků LABEL map pomocí natrénovaného modelu"""
    
    # Nastavení zařízení
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Načtení normativního atlasu
    if args.normal_atlas_path.endswith('.nii.gz') or args.normal_atlas_path.endswith('.nii'):
        normal_atlas = nib.load(args.normal_atlas_path).get_fdata()
    else:
        normal_atlas = sitk.GetArrayFromImage(sitk.ReadImage(args.normal_atlas_path))
    
    # Normalizace
    normal_atlas = (normal_atlas - normal_atlas.min()) / (normal_atlas.max() - normal_atlas.min())
    normal_atlas = torch.FloatTensor(normal_atlas).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D, H, W]
    
    # Načtení atlasu frekvence lézí (pokud je dostupný)
    lesion_atlas = None
    if args.lesion_atlas_path:
        if args.lesion_atlas_path.endswith('.nii.gz') or args.lesion_atlas_path.endswith('.nii'):
            lesion_atlas = nib.load(args.lesion_atlas_path).get_fdata()
        else:
            lesion_atlas = sitk.GetArrayFromImage(sitk.ReadImage(args.lesion_atlas_path))
        
        # Normalizace
        lesion_atlas = (lesion_atlas - lesion_atlas.min()) / (lesion_atlas.max() - lesion_atlas.min())
        lesion_atlas = torch.FloatTensor(lesion_atlas).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D, H, W]
    
    # Inicializace a načtení generátoru
    generator = LabelGenerator(
        in_channels=2,
        features=args.generator_filters,
        dropout_rate=args.dropout_rate,
        use_self_attention=args.use_self_attention
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Generování vzorků
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Vytvoření adresáře pro LABEL mapy
    label_output_dir = os.path.join(args.output_dir, "label")
    os.makedirs(label_output_dir, exist_ok=True)
    
    # Sledování úspěšnosti generování lézí
    generated_lesion_percentages = []
    
    for i in range(args.num_samples):
        with torch.no_grad():
            # Vytvoření základního náhodného šumu
            noise = torch.randn_like(normal_atlas).to(device)
            
            # Přidáme "seedy" lézí - náhodné aktivní oblasti pro pomoc generátoru
            # Vytvoříme 2-5 náhodných "seedů" lézí
            d, h, w = normal_atlas.shape[2:]
            num_seeds = torch.randint(2, 6, (1,)).item()  # 2-5 seedů
            
            for _ in range(num_seeds):
                # Náhodná pozice v objemu
                seed_d = torch.randint(0, d, (1,)).item()
                seed_h = torch.randint(0, h, (1,)).item()
                seed_w = torch.randint(0, w, (1,)).item()
                
                # Vytvoříme Gaussovský blob jako "seed" léze
                seed_size = torch.randint(2, 6, (1,)).item()  # Mírně větší seedy pro generování
                
                # Vytvoření indexových rozsahů s kontrolou hranic
                d_indices = torch.arange(max(0, seed_d-seed_size), min(d, seed_d+seed_size+1)).long()
                h_indices = torch.arange(max(0, seed_h-seed_size), min(h, seed_h+seed_size+1)).long()
                w_indices = torch.arange(max(0, seed_w-seed_size), min(w, seed_w+seed_size+1)).long()
                
                # Vytvoření Gaussovského blobu
                for id in d_indices:
                    for ih in h_indices:
                        for iw in w_indices:
                            dist = ((id-seed_d)**2 + (ih-seed_h)**2 + (iw-seed_w)**2) ** 0.5
                            if dist <= seed_size:
                                intensity = 2.5 * (1.0 - dist/seed_size)  # Silnější intenzita pro lepší inicializaci
                                noise[0, 0, id, ih, iw] = intensity
            
            # Pokud máme atlas lézí, vážíme šum podle pravděpodobnosti výskytu lézí
            if lesion_atlas is not None:
                # Posílíme šum v oblastech s vyšší pravděpodobností lézí
                noise = noise * (1.0 + lesion_atlas * 3.0)  # Silnější váhování pro generování
                # Generujeme label
                fake_label = generator(normal_atlas, noise, lesion_atlas)
            else:
                # Generujeme bez atlasu
                fake_label = generator(normal_atlas, noise)
            
            # Použijeme nižší práh pro pozitivnější detekci lézí
            threshold = 0.3  # Začínáme s nižším prahem než při tréninku
            fake_label_binary = (fake_label > threshold).float()
            
            # Kontrola, zda máme nějaké léze - pokud ne, snížíme práh
            lesion_percentage = fake_label_binary.mean().item() * 100.0
            print(f"Sample {i}: Lesion percentage = {lesion_percentage:.2f}%")
            
            # Pokud je příliš málo lézí, snížíme práh
            if lesion_percentage < 0.05:
                threshold = 0.2
                print(f"  Very few lesions detected, lowering threshold to {threshold}...")
                fake_label_binary = (fake_label > threshold).float()
                lesion_percentage = fake_label_binary.mean().item() * 100.0
                print(f"  After threshold adjustment: Lesion percentage = {lesion_percentage:.2f}%")
                
                # Pokud stále nemáme léze, zkusíme ještě nižší práh
                if lesion_percentage < 0.02:
                    threshold = 0.1
                    print(f"  Still insufficient lesions, lowering threshold to {threshold}...")
                    fake_label_binary = (fake_label > threshold).float()
                    lesion_percentage = fake_label_binary.mean().item() * 100.0
                    print(f"  After second threshold adjustment: Lesion percentage = {lesion_percentage:.2f}%")
            
            # Sledování průměrného procenta lézí
            generated_lesion_percentages.append(lesion_percentage)
            
            # Uložení výsledné binární label mapy
            label_output_path = os.path.join(label_output_dir, f'sample_{i}_label.nii.gz')
            save_sample(fake_label_binary[0, 0].cpu().numpy(), label_output_path)
            
            # Uložení také syrové predikce pro možnou analýzu
            raw_output_path = os.path.join(label_output_dir, f'sample_{i}_raw_pred.nii.gz')
            save_sample(fake_label[0, 0].cpu().numpy(), raw_output_path)
    
    # Výpis statistik generování
    avg_lesion_percentage = sum(generated_lesion_percentages) / len(generated_lesion_percentages)
    print(f"Generated {args.num_samples} LABEL maps")
    print(f"Average lesion percentage: {avg_lesion_percentage:.2f}%")
    print(f"LABEL maps saved to: {label_output_dir}")

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
    train_parser.add_argument('--lambda_focal', type=float, default=10.0,
                             help='Váha pro Focal loss (lepší zvládání nevyvážených tříd)')
    train_parser.add_argument('--lambda_black', type=float, default=15.0,
                             help='Váha pro Black Image Penalty (prevence generování černých výstupů)')
    train_parser.add_argument('--pos_weight', type=float, default=20.0,
                             help='Váha pro pozitivní třídu ve váženém Dice loss')
    train_parser.add_argument('--bg_weight', type=float, default=0.1,
                             help='Váha pro negativní třídu (pozadí) ve váženém Dice loss')
    
    # Nové parametry pro vylepšení stability
    train_parser.add_argument('--use_augmentation', action='store_true', default=True,
                              help='Použít mírnou datovou augmentaci pro zvýšení variability vzorků')
    train_parser.add_argument('--use_gradient_penalty', action='store_true', default=True,
                              help='Použít gradient penalty pro stabilnější trénink')
    train_parser.add_argument('--gradient_penalty_weight', type=float, default=10.0,
                              help='Váha pro gradient penalty')
    train_parser.add_argument('--n_critic', type=int, default=2,
                              help='Počet kroků diskriminátoru na jeden krok generátoru')
    
    # Parametry pro kontrolu generování lézí
    train_parser.add_argument('--target_lesion_percentage', type=float, default=0.01,
                             help='Cílové procento lézí (1% = 0.01)')
    
    # Parser pro generování
    generate_parser = subparsers.add_parser('generate', parents=[parent_parser],
                                           help='Generování vzorků LABEL map pomocí natrénovaného modelu')
    generate_parser.add_argument('--checkpoint_path', type=str, required=True,
                                help='Cesta k checkpointu modelu')
    generate_parser.add_argument('--num_samples', type=int, default=10,
                                help='Počet vzorků k vygenerování')
    
    args = parser.parse_args()
    
    if args.action == 'train':
        train_label_gan(args)
    elif args.action == 'generate':
        generate_label_samples(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 