import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from monai.networks.nets import SwinUNETR
from monai.losses import SSIMLoss
from monai.transforms import (
    LoadImaged,
    ScaleIntensityd,
    Orientationd,
    Spacingd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    CropForegroundd,
    ToTensord,
    Compose,
    EnsureChannelFirstd
)
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import glob
from pathlib import Path
import argparse
from torch.amp import GradScaler, autocast
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
from matplotlib.backends.backend_pdf import PdfPages
import random
import matplotlib.gridspec as gridspec
import pandas as pd
import json
from typing import Dict, List, Optional, Union, Tuple
from contextlib import nullcontext  # Add nullcontext import for the training loop

# Nastavení zařízení pro trénink
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Používám zařízení: {device}")

class LesionInpaintingDataset(Dataset):
    """
    Dataset pro načítání trojic snímků: pseudo-zdravý mozek (vstup), ADC mapa s lézemi (cíl) 
    a binární maska léze (podmínka)
    
    Tato verze pracuje s celými objemy (full volume) místo patchů.
    """
    def __init__(
        self,
        pseudo_healthy_dir,
        adc_dir,
        lesion_mask_dir,
        transform=None,
        norm_range=(0, 1),
        crop_foreground=True,
        target_size=None
    ):
        """
        Inicializace datasetu
        
        Args:
            pseudo_healthy_dir (str): Adresář s pseudo-zdravými ADC mapami
            adc_dir (str): Adresář s ADC mapami s lézemi
            lesion_mask_dir (str): Adresář s binárními maskami lézí
            transform: Transformace pro augmentaci dat (MONAI transformace)
            norm_range (tuple): Rozsah pro normalizaci hodnot (min, max)
            crop_foreground (bool): Zda ořezat data na bounding box mozku
            target_size (tuple): Cílová velikost výstupních objemů (D, H, W), pokud None, zachová se původní velikost
                                 s paddingem na nejbližší mocninu 2 pro každý rozměr
        """
        self.pseudo_healthy_dir = Path(pseudo_healthy_dir)
        self.adc_dir = Path(adc_dir)
        self.lesion_mask_dir = Path(lesion_mask_dir)
        self.transform = transform
        self.norm_range = norm_range
        self.crop_foreground = crop_foreground
        self.target_size = target_size
        
        # Najít všechny pseudo-zdravé soubory jako základ
        self.pseudo_healthy_files = sorted(list(self.pseudo_healthy_dir.glob("*.mha")))
        print(f"Nalezeno {len(self.pseudo_healthy_files)} pseudo-zdravých mozků")
        
        # Vypsat několik příkladů nalezených souborů pro diagnostiku
        if len(self.pseudo_healthy_files) > 0:
            print("Příklady pseudo-zdravých souborů:")
            for i, f in enumerate(self.pseudo_healthy_files[:3]):
                print(f"  {i+1}. {f.name}")
            
            # Získat všechny ADC a lesion mask soubory
            all_adc_files = list(self.adc_dir.glob("*.mha"))
            all_lesion_files = list(self.lesion_mask_dir.glob("*.mha"))
            
            print(f"Nalezeno celkem {len(all_adc_files)} ADC souborů a {len(all_lesion_files)} masek lézí")
        
        # Sestavit seznam všech trojic souborů
        self.file_triplets = []
        
        for ph_file in self.pseudo_healthy_files:
            # Extrahovat ID pacienta z názvu pseudo-zdravého souboru
            # Tento kód je flexibilnější - hledáme ID pacienta před "-PSEUDO_HEALTHY"
            ph_filename = ph_file.name
            if "-PSEUDO_HEALTHY" in ph_filename:
                patient_id = ph_filename.split('-PSEUDO_HEALTHY')[0]
            else:
                # Pokud nenajdeme přesný formát, použijeme název souboru bez přípony jako ID
                patient_id = ph_file.stem.split('_')[0]  # Bereme první část před podtržítkem jako ID
            
            # Vypsat ID pacienta pro kontrolu
            
            # Hledáme ADC soubory s různými možnými přípony
            adc_file = []
            for pattern in [f"{patient_id}*-ADC*.mha", f"{patient_id}*adc*.mha", f"{patient_id}*.mha"]:
                adc_file = list(self.adc_dir.glob(pattern))
            
            # Hledáme masky lézí s různými možnými přípony
            lesion_file = []
            for pattern in [f"{patient_id}*_lesion*.mha", f"{patient_id}*mask*.mha", f"{patient_id}*lesion*.mha", f"{patient_id}*.mha"]:
                lesion_file = list(self.lesion_mask_dir.glob(pattern))
                if lesion_file:
                    break
            
            if adc_file and lesion_file:
                self.file_triplets.append({
                    'pseudo_healthy': ph_file,
                    'adc': adc_file[0],
                    'lesion_mask': lesion_file[0],
                    'patient_id': patient_id
                })
        
        print(f"Nalezeno {len(self.file_triplets)} kompletních tripletů souborů")
        
    
    def __len__(self):
        return len(self.file_triplets)
    
    def _load_and_preprocess(self, file_path):
        """Načte .mha soubor a provede předzpracování"""
        img = sitk.ReadImage(str(file_path))
        data = sitk.GetArrayFromImage(img)
        
        # Převeď na float32 pro lepší zpracování
        data = data.astype(np.float32)
        
        # Ořízni záporné hodnoty na 0
        data = np.clip(data, 0, None)
        
        return data, img
    
    def _normalize(self, data):
        """Normalizuje data na zadaný rozsah"""
        min_val, max_val = self.norm_range
        if np.max(data) - np.min(data) > 0:
            normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            normalized = normalized * (max_val - min_val) + min_val
            return normalized
        return data
    
    def _get_bounding_box(self, binary_mask, margin=5):
        """Získá bounding box s marží pro neprázdnou oblast"""
        # Najít souřadnice neprázdných voxelů
        if np.any(binary_mask):
            z, y, x = np.where(binary_mask > 0)
            
            z_min, z_max = max(0, np.min(z) - margin), min(binary_mask.shape[0], np.max(z) + margin + 1)
            y_min, y_max = max(0, np.min(y) - margin), min(binary_mask.shape[1], np.max(y) + margin + 1)
            x_min, x_max = max(0, np.min(x) - margin), min(binary_mask.shape[2], np.max(x) + margin + 1)
            
            return (z_min, z_max, y_min, y_max, x_min, x_max)
        else:
            # Pokud maska je prázdná, vrať celý objem
            return (0, binary_mask.shape[0], 0, binary_mask.shape[1], 0, binary_mask.shape[2])
    
    def _crop_to_bounding_box(self, data, bbox):
        """Ořízne data podle bounding boxu"""
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        return data[z_min:z_max, y_min:y_max, x_min:x_max]
    
    def _pad_to_power_of_2(self, data):
        """
        Doplní každý rozměr na nejbližší násobek 32 (patch size pro SwinUNETR).
        Pokud je rozměr již násobkem 32, zůstává beze změny.
        Garantuje minimální velikost 32x32x32 pro všechny rozměry.
        """
        # Získat aktuální rozměry
        d, h, w = data.shape
        
        # Najít nejbližší vyšší násobek 32 pro každý rozměr, pokud již není násobkem 32
        # a zároveň zajistit minimální velikost 32 pro všechny rozměry
        patch_size = 32
        target_d = max(patch_size, ((d + patch_size - 1) // patch_size) * patch_size)
        target_h = max(patch_size, ((h + patch_size - 1) // patch_size) * patch_size)
        target_w = max(patch_size, ((w + patch_size - 1) // patch_size) * patch_size)
        
        print(f"Padding from {data.shape} to target size: ({target_d}, {target_h}, {target_w})")
        
        # Vypočítat padding pro každý rozměr
        pad_d = max(0, target_d - d)
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        
        # Aplikovat padding symetricky
        pad_d_before, pad_d_after = pad_d // 2, pad_d - (pad_d // 2)
        pad_h_before, pad_h_after = pad_h // 2, pad_h - (pad_h // 2)
        pad_w_before, pad_w_after = pad_w // 2, pad_w - (pad_w // 2)
        
        # Padding v numpy
        padded_data = np.pad(
            data,
            ((pad_d_before, pad_d_after), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
            mode='constant',
            constant_values=0
        )
        
        return padded_data
    
    def __getitem__(self, idx):
        """Vrátí položku datasetu podle indexu"""
        # Získat cestu k souborům
        triplet = self.file_triplets[idx]
        
        # Načíst a předzpracovat data
        pseudo_healthy_data, ph_img = self._load_and_preprocess(triplet['pseudo_healthy'])
        adc_data, _ = self._load_and_preprocess(triplet['adc'])
        lesion_mask_data, _ = self._load_and_preprocess(triplet['lesion_mask'])
        
        # Binarizovat masku léze
        lesion_mask_data = (lesion_mask_data > 0).astype(np.float32)
        
        # Pokud je požadováno, ořízni data pomocí bounding boxu
        if self.crop_foreground:
            # Použít kombinaci pseudo-zdravého mozku a masky léze pro určení bounding boxu
            combined_mask = (pseudo_healthy_data > 0) | (lesion_mask_data > 0)
            bbox = self._get_bounding_box(combined_mask)
            
            pseudo_healthy_data = self._crop_to_bounding_box(pseudo_healthy_data, bbox)
            adc_data = self._crop_to_bounding_box(adc_data, bbox)
            lesion_mask_data = self._crop_to_bounding_box(lesion_mask_data, bbox)
        
        # Normalizovat hodnoty
        pseudo_healthy_data = self._normalize(pseudo_healthy_data)
        adc_data = self._normalize(adc_data)
        
        # Doplnit na mocninu 2 pomocí paddingu
        pseudo_healthy_data = self._pad_to_power_of_2(pseudo_healthy_data)
        adc_data = self._pad_to_power_of_2(adc_data)
        lesion_mask_data = self._pad_to_power_of_2(lesion_mask_data)
        
        # Ujistit se, že všechna data mají stejné rozměry
        assert pseudo_healthy_data.shape == adc_data.shape == lesion_mask_data.shape, \
            f"Nekonzistentní rozměry: pseudo_healthy={pseudo_healthy_data.shape}, adc={adc_data.shape}, mask={lesion_mask_data.shape}"
        
        # Vytvořit sample slovník pro MONAI transformace
        sample = {
            'pseudo_healthy': pseudo_healthy_data,
            'adc': adc_data,
            'lesion_mask': lesion_mask_data,
            'patient_id': triplet['patient_id'],  # Použít přímo patient_id z tripletu
            'volume_shape': pseudo_healthy_data.shape  # Uložit původní rozměr pro pozdější použití
        }
        
        # Převést numerická data na torch tenzory a přidat kanálovou dimenzi
        for key in ['pseudo_healthy', 'adc', 'lesion_mask']:
            sample[key] = torch.from_numpy(sample[key]).unsqueeze(0)
        
        # Aplikovat další transformace, pokud jsou definované
        if self.transform:
            # MONAI transformace očekává slovník metadat, proto nejprve převedeme na vhodný formát
            transform_sample = {}
            for key in ['pseudo_healthy', 'adc', 'lesion_mask']:
                transform_sample[key] = sample[key]
            
            # Aplikovat transformace
            transform_sample = self.transform(transform_sample)
            
            # Vrátit transformovaná data zpět do původního sample
            for key in ['pseudo_healthy', 'adc', 'lesion_mask']:
                sample[key] = transform_sample[key]
        
        return sample

class Generator(nn.Module):
    """
    U-Net styl generátoru pro inpainting lézí
    Vstup: Pseudo-zdravý mozek a maska léze
    Výstup: Generovaná ADC mapa léze
    """
    def __init__(
        self, 
        in_channels=2,  # 1 pro pseudo-zdravý + 1 pro masku léze
        out_channels=1,  # 1 pro ADC mapu
        base_filters=64,
        depth=5,
        min_size=32
    ):
        super().__init__()
        
        # Přidáme kontrolu minimální velikosti vstupu
        self.min_input_size = min_size
        
        # Encoder - první konvoluce bez normalizace
        self.encoder_layers = nn.ModuleList()
        
        # První vrstva s menším kernelem a stride=1 pro lepší zpracování malých vstupů
        self.encoder_layers.append(
            nn.Sequential(
                nn.Conv3d(in_channels, base_filters, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        
        # Přidat zbytek enkodéru
        current_filters = base_filters
        for i in range(1, depth):
            # Od druhé vrstvy použijeme standardní parametry
            stride = 2
            kernel_size = 4
            
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        current_filters, 
                        current_filters * 2, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=1,
                        bias=False
                    ),
                    nn.BatchNorm3d(current_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            current_filters *= 2
        
        # Decoder s přeskočenými spojeními
        self.decoder_layers = nn.ModuleList()
        
        for i in range(depth - 1):
            # Pro největší hloubku použijeme menší kernel
            kernel_size = 3 if i == 0 else 4
            
            if i == 0:  # Nejhlubší vrstva bez přeskočeného spojení
                self.decoder_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(
                            current_filters, 
                            current_filters // 2,
                            kernel_size=kernel_size, 
                            stride=2, 
                            padding=1,
                            bias=False
                        ),
                        nn.BatchNorm3d(current_filters // 2),
                        nn.ReLU(inplace=True)
                    )
                )
            else:  # Ostatní vrstvy s přeskočeným spojením
                self.decoder_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(
                            current_filters, 
                            current_filters // 2,
                            kernel_size=kernel_size, 
                            stride=2, 
                            padding=1,
                            bias=False
                        ),
                        nn.BatchNorm3d(current_filters // 2),
                        nn.ReLU(inplace=True)
                    )
                )
            current_filters //= 2
        
        # Finální konvoluce
        self.final_conv = nn.Sequential(
            nn.Conv3d(current_filters, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Normalizovaný výstup v rozmezí [-1, 1]
        )
        
    def forward(self, pseudo_healthy, lesion_mask):
        """
        Forward pass generátoru
        
        Args:
            pseudo_healthy (torch.Tensor): Tensor pseudo-zdravého mozku [B, 1, D, H, W]
            lesion_mask (torch.Tensor): Tensor binární masky léze [B, 1, D, H, W]
        
        Returns:
            torch.Tensor: Generovaná ADC mapa léze [B, 1, D, H, W]
        """
        # Kontrola minimální velikosti vstupu
        min_size = self.min_input_size
        d, h, w = pseudo_healthy.shape[2:5]
        
        if d < min_size or h < min_size or w < min_size:
            print(f"Warning: Input size {(d, h, w)} is smaller than minimum size {min_size}. Results may be unpredictable.")
        
        # Spojit vstupy pro generátor
        x = torch.cat([pseudo_healthy, lesion_mask], dim=1)
        
        # Uložit výstupy enkodéru pro přeskočená spojení
        skip_connections = []
        
        # Encoder
        for encoder in self.encoder_layers:
            x = encoder(x)
            skip_connections.append(x)
        
        # Odstranit nejhlubší vrstvu, která nebude mít přeskočené spojení
        skip_connections.pop()
        
        # Decoder s přeskočenými spojeními
        for i, decoder in enumerate(self.decoder_layers):
            x = decoder(x)
            
            # Přidat přeskočené spojení, kromě poslední vrstvy enkodéru
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                
                # Zajistit, že velikosti jsou kompatibilní pro konkatenaci
                if x.shape[2:] != skip.shape[2:]:
                    # Interpolace pro zajištění stejné velikosti
                    x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
                
                x = torch.cat([x, skip], dim=1)
        
        # Finální konvoluce pro generování ADC mapy
        x = self.final_conv(x)
        
        return x


class Discriminator(nn.Module):
    """
    PatchGAN diskriminátor s 3D konvolucemi pro rozlišení reálných a generovaných lézí
    Vstup: Pseudo-zdravý mozek, maska léze a ADC mapa (reálná nebo generovaná)
    Výstup: 3D pole skóre, které indikují, zda každý patch obsahuje reálnou nebo generovanou lézi
    """
    def __init__(
        self,
        in_channels=3,  # 1 pro pseudo-zdravý + 1 pro masku léze + 1 pro ADC mapu
        base_filters=64,
        n_layers=4,
        min_size=32
    ):
        super().__init__()
        
        # Přidáme kontrolu minimální velikosti vstupu
        self.min_input_size = min_size
        
        # Menší kernel a stride pro první vrstvu, aby lépe zvládala malé vstupy
        layers = [
            nn.Conv3d(in_channels, base_filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Přidat další vrstvy s normalizací
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            # Menší kernel a stride pro další vrstvy
            kernel_size = 3 if n == 1 else 4
            stride = 1 if n == 1 else 2
            
            layers += [
                nn.Conv3d(
                    base_filters * nf_mult_prev,
                    base_filters * nf_mult,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm3d(base_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Závěrečná vrstva pro klasifikaci patchů
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        layers += [
            nn.Conv3d(
                base_filters * nf_mult_prev,
                base_filters * nf_mult,
                kernel_size=3,  # Menší kernel pro finální vrstvu
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(base_filters * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                base_filters * nf_mult,
                1,
                kernel_size=3,  # Menší kernel pro výstupní vrstvu
                stride=1,
                padding=1
            )
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, pseudo_healthy, lesion_mask, adc_map):
        """
        Forward pass diskriminátoru
        
        Args:
            pseudo_healthy (torch.Tensor): Tensor pseudo-zdravého mozku [B, 1, D, H, W]
            lesion_mask (torch.Tensor): Tensor binární masky léze [B, 1, D, H, W]
            adc_map (torch.Tensor): Tensor ADC mapy (reálné nebo generované) [B, 1, D, H, W]
        
        Returns:
            torch.Tensor: Mapa skóre pravděpodobnosti reálné/generované léze [B, 1, D', H', W']
        """
        # Kontrola minimální velikosti vstupu
        min_size = self.min_input_size
        d, h, w = pseudo_healthy.shape[2:5]
        
        if d < min_size or h < min_size or w < min_size:
            print(f"Warning: Input size {(d, h, w)} is smaller than minimum size {min_size}. Results may be unpredictable.")
        
        # Spojit všechny vstupy pro diskriminátor
        x = torch.cat([pseudo_healthy, lesion_mask, adc_map], dim=1)
        
        # Získat mapu skóre
        return self.model(x)


class LesionInpaintingGAN(nn.Module):
    """
    CGAN pro inpainting lézí do mozku s využitím SwinUNETR
    """
    def __init__(
        self,
        img_size=(96, 96, 96),
        gen_features=48,
        disc_base_filters=64,
        disc_n_layers=4,
        lambda_l1=10.0,
        lambda_identity=5.0,
        lambda_ssim=1.0,
        lambda_edge=1.0
    ):
        super().__init__()
        
        # Generátor a diskriminátor
        self.generator = Generator(
            img_size=img_size,
            in_channels=2,  # pseudo-healthy + mask
            out_channels=1,  # ADC s lézí
            feature_size=gen_features
        )
        
        self.discriminator = Discriminator(
            in_channels=3,  # pseudo-healthy + mask + ADC
            base_filters=disc_base_filters,
            n_layers=disc_n_layers
        )
        
        # Váhy loss funkcí
        self.lambda_l1 = lambda_l1
        self.lambda_identity = lambda_identity
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        
        # Loss funkce
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(spatial_dims=3, win_size=7)
        
        # Inicializace vah
        self.init_weights()
    
    def init_weights(self):
        """Inicializace vah sítě"""
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        
        # Inicializovat diskriminátor
        self.discriminator.apply(init_func)
    
    def compute_edge_loss(self, pred, target, mask, kernel_size=3):
        """
        Výpočet edge loss pro zajištění plynulého přechodu na hranici masky
        Implementace bez gradientů, která se zaměřuje pouze na hodnoty v oblasti masky
        
        Args:
            pred (torch.Tensor): Predikovaná ADC mapa [B, 1, D, H, W]
            target (torch.Tensor): Cílová ADC mapa [B, 1, D, H, W]
            mask (torch.Tensor): Binární maska léze [B, 1, D, H, W]
            kernel_size (int): Nepoužívá se v této implementaci
        
        Returns:
            torch.Tensor: Skalární hodnota edge loss
        """
        # Nejprve se ujistíme, že všechny tenzory mají stejný tvar
        if pred.shape != target.shape or pred.shape != mask.shape:
            # Reportujeme rozdíly ve velikostech pro diagnostiku
            print(f"Warning: Rozdílné velikosti tensorů - pred: {pred.shape}, target: {target.shape}, mask: {mask.shape}")
            
            # Ořezat všechny tensory na nejmenší společnou velikost
            min_depth = min(pred.shape[2], target.shape[2], mask.shape[2])
            min_height = min(pred.shape[3], target.shape[3], mask.shape[3])
            min_width = min(pred.shape[4], target.shape[4], mask.shape[4])
            
            pred = pred[:, :, :min_depth, :min_height, :min_width]
            target = target[:, :, :min_depth, :min_height, :min_width]
            mask = mask[:, :, :min_depth, :min_height, :min_width]
            
            print(f"Tensory ořezány na společnou velikost: {pred.shape}")
        
        # Jednoduchý L1 loss v oblasti masky
        masked_loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        
        # Normalizace podle počtu voxelů v masce
        mask_sum = torch.sum(mask) + 1e-8
        edge_loss = masked_loss / mask_sum
        
        return edge_loss
    
    def compute_localized_ssim_loss(self, pred, target, mask):
        """
        Výpočet SSIM loss pouze v oblasti léze
        
        Args:
            pred (torch.Tensor): Predikovaná ADC mapa [B, 1, D, H, W]
            target (torch.Tensor): Cílová ADC mapa [B, 1, D, H, W]
            mask (torch.Tensor): Binární maska léze [B, 1, D, H, W]
        
        Returns:
            torch.Tensor: Skalární hodnota lokalizovaného SSIM loss
        """
        # Výpočet SSIM loss pouze v oblasti léze
        ssim_loss_val = 1.0 - self.ssim_loss(pred * mask, target * mask)
        return ssim_loss_val
    
    def generator_loss(self, pseudo_healthy, lesion_mask, real_adc, fake_adc, disc_fake_pred):
        """
        Výpočet loss funkce pro generátor
        
        Args:
            pseudo_healthy (torch.Tensor): Pseudo-zdravý mozek [B, 1, D, H, W]
            lesion_mask (torch.Tensor): Binární maska léze [B, 1, D, H, W]
            real_adc (torch.Tensor): Reálná ADC mapa s lézí [B, 1, D, H, W]
            fake_adc (torch.Tensor): Generovaná ADC mapa s lézí [B, 1, D, H, W]
            disc_fake_pred (torch.Tensor): Predikce diskriminátoru pro generovaná data
        
        Returns:
            tuple: (celkový loss, adversarial loss, L1 loss, identity loss, SSIM loss, edge loss)
        """
        # Adversarial loss pro generátor - motivovat generátor k vytváření věrohodných lézí
        target_real = torch.ones_like(disc_fake_pred).to(disc_fake_pred.device)
        adv_loss = self.adversarial_loss(disc_fake_pred, target_real)
        
        # Rekonstrukční L1 loss - pouze v oblasti léze
        l1_loss_masked = self.l1_loss(fake_adc * lesion_mask, real_adc * lesion_mask)
        
        # Identity loss - oblast mimo masku by měla zůstat nezměněná
        identity_loss = self.l1_loss(fake_adc * (1 - lesion_mask), pseudo_healthy * (1 - lesion_mask))
        
        # Lokalizovaný SSIM loss pro strukturální podobnost v oblasti léze
        local_ssim_loss = self.compute_localized_ssim_loss(fake_adc, real_adc, lesion_mask)
        
        # Edge/Gradient loss pro plynulý přechod na hranici léze
        edge_loss = self.compute_edge_loss(fake_adc, real_adc, lesion_mask)
        
        # Celkový loss pro generátor
        total_loss = (
            adv_loss + 
            self.lambda_l1 * l1_loss_masked + 
            self.lambda_identity * identity_loss + 
            self.lambda_ssim * local_ssim_loss + 
            self.lambda_edge * edge_loss
        )
        
        return total_loss, adv_loss, l1_loss_masked, identity_loss, local_ssim_loss, edge_loss
    
    def discriminator_loss(self, pseudo_healthy, lesion_mask, real_adc, fake_adc, disc_real_pred, disc_fake_pred):
        """
        Výpočet loss funkce pro diskriminátor
        
        Args:
            pseudo_healthy (torch.Tensor): Pseudo-zdravý mozek [B, 1, D, H, W]
            lesion_mask (torch.Tensor): Binární maska léze [B, 1, D, H, W]
            real_adc (torch.Tensor): Reálná ADC mapa s lézí [B, 1, D, H, W]
            fake_adc (torch.Tensor): Generovaná ADC mapa s lézí [B, 1, D, H, W]
            disc_real_pred (torch.Tensor): Predikce diskriminátoru pro reálná data
            disc_fake_pred (torch.Tensor): Predikce diskriminátoru pro generovaná data
        
        Returns:
            torch.Tensor: Loss hodnota pro diskriminátor
        """
        # Cílové hodnoty
        target_real = torch.ones_like(disc_real_pred).to(disc_real_pred.device)
        target_fake = torch.zeros_like(disc_fake_pred).to(disc_fake_pred.device)
        
        # Loss pro reálná data
        loss_real = self.adversarial_loss(disc_real_pred, target_real)
        
        # Loss pro generovaná data
        loss_fake = self.adversarial_loss(disc_fake_pred, target_fake)
        
        # Celkový loss pro diskriminátor
        loss_d = 0.5 * (loss_real + loss_fake)
        
        return loss_d
    
    def forward(self, pseudo_healthy, lesion_mask):
        """
        Forward pass celého modelu
        
        Args:
            pseudo_healthy (torch.Tensor): Pseudo-zdravý mozek [B, 1, D, H, W]
            lesion_mask (torch.Tensor): Binární maska léze [B, 1, D, H, W]
        
        Returns:
            torch.Tensor: Generovaná ADC mapa s lézí [B, 1, D, H, W]
        """
        return self.generator(pseudo_healthy, lesion_mask)

class LesionInpaintingTrainer:
    """Třída pro trénování GAN modelu pro inpainting lézí"""
    
    def __init__(self, model, optimizer_g, optimizer_d, device, output_dir, config=None, use_amp=False, visualize_interval=5):
        """
        Inicializace traineru
        
        Args:
            model: Instance modelu pro inpainting
            optimizer_g: Optimizer pro generátor
            optimizer_d: Optimizer pro diskriminátor
            device: Zařízení pro výpočty (CPU/GPU)
            output_dir: Adresář pro ukládání výstupů
            config: Konfigurace tréninku
            use_amp: Použít Automatic Mixed Precision
            visualize_interval: Interval pro vizualizaci výsledků
        """
        self.model = model
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config
        self.use_amp = use_amp
        self.visualize_interval = visualize_interval
        
        # Vytvoření adresáře pro výstupy, pokud neexistuje
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vytvořit adresář pro PDF vizualizace
        self.pdf_visualization_dir = self.output_dir / "pdf_visualizations"
        self.pdf_visualization_dir.mkdir(exist_ok=True)
        
        # Vytvořit adresář pro checkpointy
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Vytvořit adresář pro vzorky
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        
        # Inicializace AMP scaler pokud používáme mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Inicializace logů
        self.train_log = {
            'epoch': [],
            'g_loss': [],
            'g_adv_loss': [],
            'g_l1_loss': [],
            'g_identity_loss': [],
            'g_ssim_loss': [],
            'g_edge_loss': [],
            'd_loss': []
        }
        
        self.val_log = {
            'epoch': [],
            'g_loss': [],
            'g_adv_loss': [],
            'g_l1_loss': [],
            'g_identity_loss': [],
            'g_ssim_loss': [],
            'g_edge_loss': [],
            'd_loss': []
        }
    
    def save_model(self, epoch, is_best=False):
        """Uloží checkpoint modelu"""
        if is_best:
            checkpoint_path = self.checkpoints_dir / f"best_model.pth"
        else:
            checkpoint_path = self.checkpoints_dir / f"model_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'train_log': self.train_log,
            'val_log': self.val_log
        }, checkpoint_path)
        
        print(f"Model uložen do {checkpoint_path}")
    
    def load_model(self, checkpoint_path):
        """Načte checkpoint modelu"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        
        if 'train_log' in checkpoint:
            self.train_log = checkpoint['train_log']
        
        if 'val_log' in checkpoint:
            self.val_log = checkpoint['val_log']
        
        print(f"Model načten z {checkpoint_path}, pokračuji od epochy {start_epoch}")
        return start_epoch
    
    def visualize_results(self, epoch, dataloader, num_samples=4):
        """Vizualizace výsledků modelu na validačních datech"""
        self.model.eval()
        
        # Vytvořit adresář pro aktuální epochu
        vis_epoch_dir = self.samples_dir / f"epoch_{epoch}"
        vis_epoch_dir.mkdir(exist_ok=True)
        
        # Vytvořit CSV soubor pro metriky
        csv_path = vis_epoch_dir / "metrics.csv"
        with open(csv_path, 'w') as f:
            f.write("patient_id,slice_idx,ssim,mae\n")
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                
                # Přesunout data na správné zařízení
                pseudo_healthy = batch['pseudo_healthy'].to(self.device)
                lesion_mask = batch['lesion_mask'].to(self.device)
                real_adc = batch['adc'].to(self.device)
                patient_id = batch['patient_id'][0]
                
                # Generovat falešnou ADC mapu
                fake_adc = self.model(pseudo_healthy, lesion_mask)
                
                # Výpočet SSIM a MAE na celém objemu
                # Počítáme SSIM v oblasti léze
                ssim_value = self.model.ssim_loss(fake_adc * lesion_mask, real_adc * lesion_mask).item()
                
                # Počítáme MAE (mean absolute error) v oblasti léze
                masked_fake = fake_adc * lesion_mask
                masked_real = real_adc * lesion_mask
                mae_value = torch.sum(torch.abs(masked_fake - masked_real)) / (torch.sum(lesion_mask) + 1e-8)
                mae_value = mae_value.item()
                
                # Vybrat prostřední řez pro zobrazení
                slice_idx = pseudo_healthy.shape[2] // 2
                
                # Extrahovat 2D řezy z 3D objemů
                ph_slice = pseudo_healthy[0, 0, slice_idx].cpu().numpy()
                mask_slice = lesion_mask[0, 0, slice_idx].cpu().numpy()
                real_slice = real_adc[0, 0, slice_idx].cpu().numpy()
                fake_slice = fake_adc[0, 0, slice_idx].cpu().numpy()
                
                # Zapsat metriky do CSV
                with open(csv_path, 'a') as f:
                    f.write(f"{patient_id},{slice_idx},{ssim_value:.4f},{mae_value:.4f}\n")
                
                # Vytvořit obrazek s výsledky
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Zobrazit pseudo-zdravý mozek
                im0 = axes[0, 0].imshow(ph_slice, cmap='gray')
                axes[0, 0].set_title(f'Pseudo-zdravý mozek')
                plt.colorbar(im0, ax=axes[0, 0])
                
                # Zobrazit skutečnou ADC mapu s lézí
                im1 = axes[0, 1].imshow(real_slice, cmap='gray')
                axes[0, 1].set_title(f'Skutečná ADC mapa')
                plt.colorbar(im1, ax=axes[0, 1])
                
                # Zobrazit masku léze
                im2 = axes[1, 0].imshow(mask_slice, cmap='Reds', alpha=0.7)
                axes[1, 0].imshow(ph_slice, cmap='gray', alpha=0.5)
                axes[1, 0].set_title(f'Maska léze')
                plt.colorbar(im2, ax=axes[1, 0])
                
                # Zobrazit generovanou ADC mapu
                im3 = axes[1, 1].imshow(fake_slice, cmap='gray')
                axes[1, 1].set_title(f'Generovaná ADC mapa')
                plt.colorbar(im3, ax=axes[1, 1])
                
                # Uložit obrázek
                fig.suptitle(f'Pacient {patient_id} - SSIM: {ssim_value:.4f}, MAE: {mae_value:.4f}')
                plt.tight_layout()
                plt.savefig(vis_epoch_dir / f"{patient_id}_slice_{slice_idx}.png")
                plt.close(fig)
                
                # Také uložit rozdíl mezi reálnou a generovanou ADC mapou
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Zobrazit skutečnou ADC mapu
                im0 = axes[0].imshow(real_slice, cmap='gray')
                axes[0].set_title(f'Skutečná ADC mapa')
                plt.colorbar(im0, ax=axes[0])
                
                # Zobrazit generovanou ADC mapu
                im1 = axes[1].imshow(fake_slice, cmap='gray')
                axes[1].set_title(f'Generovaná ADC mapa')
                plt.colorbar(im1, ax=axes[1])
                
                # Zobrazit rozdíl
                diff = np.abs(real_slice - fake_slice)
                im2 = axes[2].imshow(diff, cmap='hot')
                axes[2].set_title(f'Rozdíl (abs)')
                plt.colorbar(im2, ax=axes[2])
                
                # Uložit obrázek
                fig.suptitle(f'Pacient {patient_id} - Porovnání (SSIM: {ssim_value:.4f}, MAE: {mae_value:.4f})')
                plt.tight_layout()
                plt.savefig(vis_epoch_dir / f"{patient_id}_diff_slice_{slice_idx}.png")
                plt.close(fig)
                
                # Uložit také 3D data jako .npy soubory pro další analýzu
                np.save(vis_epoch_dir / f"{patient_id}_pseudo_healthy.npy", 
                       pseudo_healthy[0, 0].cpu().numpy())
                np.save(vis_epoch_dir / f"{patient_id}_lesion_mask.npy", 
                       lesion_mask[0, 0].cpu().numpy())
                np.save(vis_epoch_dir / f"{patient_id}_real_adc.npy", 
                       real_adc[0, 0].cpu().numpy())
                np.save(vis_epoch_dir / f"{patient_id}_fake_adc.npy", 
                       fake_adc[0, 0].cpu().numpy())
    
    def train_epoch(self, dataloader, epoch):
        """Trénování jedné epochy"""
        self.model.train()
        
        epoch_g_loss = 0
        epoch_g_adv_loss = 0
        epoch_g_l1_loss = 0
        epoch_g_identity_loss = 0
        epoch_g_ssim_loss = 0
        epoch_g_edge_loss = 0
        epoch_d_loss = 0
        
        # Progress bar pro sledování průběhu
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Přesunout data na správné zařízení
            pseudo_healthy = batch['pseudo_healthy'].to(self.device)
            lesion_mask = batch['lesion_mask'].to(self.device)
            real_adc = batch['adc'].to(self.device)
            
            # ===== Trénink diskriminátoru =====
            self.optimizer_d.zero_grad()
            
            # Generovat falešnou ADC mapu
            with torch.amp.autocast('cuda') if self.use_amp else nullcontext():
                fake_adc = self.model.generator(pseudo_healthy, lesion_mask)
                
                # Predikce diskriminátoru pro reálná a falešná data
                disc_real_pred = self.model.discriminator(pseudo_healthy, lesion_mask, real_adc)
                disc_fake_pred = self.model.discriminator(pseudo_healthy, lesion_mask, fake_adc.detach())
                
                # Loss diskriminátoru
                d_loss = self.model.discriminator_loss(
                    pseudo_healthy, lesion_mask, real_adc, fake_adc, 
                    disc_real_pred, disc_fake_pred
                )
            
            # Zpětná propagace pro diskriminátor
            if self.use_amp:
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.optimizer_d)
            else:
                d_loss.backward()
                self.optimizer_d.step()
            
            # ===== Trénink generátoru =====
            self.optimizer_g.zero_grad()
            
            # Generovat falešnou ADC mapu znovu (pro generátor)
            with torch.amp.autocast('cuda') if self.use_amp else nullcontext():
                fake_adc = self.model.generator(pseudo_healthy, lesion_mask)
                
                # Predikce diskriminátoru pro falešná data
                disc_fake_pred = self.model.discriminator(pseudo_healthy, lesion_mask, fake_adc)
                
                # Loss generátoru
                g_loss, g_adv_loss, g_l1_loss, g_identity_loss, g_ssim_loss, g_edge_loss = \
                    self.model.generator_loss(
                        pseudo_healthy, lesion_mask, real_adc, fake_adc, disc_fake_pred
                    )
            
            # Zpětná propagace pro generátor
            if self.use_amp:
                self.scaler.scale(g_loss).backward()
                self.scaler.step(self.optimizer_g)
                self.scaler.update()
            else:
                g_loss.backward()
                self.optimizer_g.step()
            
            # Akumulovat loss hodnoty
            epoch_g_loss += g_loss.item()
            epoch_g_adv_loss += g_adv_loss.item()
            epoch_g_l1_loss += g_l1_loss.item()
            epoch_g_identity_loss += g_identity_loss.item()
            epoch_g_ssim_loss += g_ssim_loss.item()
            epoch_g_edge_loss += g_edge_loss.item()
            epoch_d_loss += d_loss.item()
            
            # Aktualizovat progress bar
            pbar.set_postfix({
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item()
            })
        
        # Vypočítat průměrné hodnoty
        num_batches = len(dataloader)
        epoch_g_loss /= num_batches
        epoch_g_adv_loss /= num_batches
        epoch_g_l1_loss /= num_batches
        epoch_g_identity_loss /= num_batches
        epoch_g_ssim_loss /= num_batches
        epoch_g_edge_loss /= num_batches
        epoch_d_loss /= num_batches
        
        # Aktualizovat log
        self.train_log['epoch'].append(epoch)
        self.train_log['g_loss'].append(epoch_g_loss)
        self.train_log['g_adv_loss'].append(epoch_g_adv_loss)
        self.train_log['g_l1_loss'].append(epoch_g_l1_loss)
        self.train_log['g_identity_loss'].append(epoch_g_identity_loss)
        self.train_log['g_ssim_loss'].append(epoch_g_ssim_loss)
        self.train_log['g_edge_loss'].append(epoch_g_edge_loss)
        self.train_log['d_loss'].append(epoch_d_loss)
        
        return {
            'g_loss': epoch_g_loss,
            'd_loss': epoch_d_loss
        }
    
    def validate(self, val_loader, epoch):
        """
        Validace modelu na validační sadě
        
        Args:
            val_loader (DataLoader): DataLoader s validačními daty
            epoch (int): Číslo aktuální epochy
            
        Returns:
            dict: Slovník s průměrnými metrikami na validační sadě
        """
        self.model.eval()
        val_metrics = {
            'g_loss': 0.0,
            'g_adv_loss': 0.0,
            'g_l1_loss': 0.0,
            'g_identity_loss': 0.0,
            'g_ssim_loss': 0.0,
            'g_edge_loss': 0.0,
            'd_loss': 0.0
        }
        
        # Uložit všechny batche pro pozdější náhodný výběr
        all_batches = []
        
        # Validace na všech datech
        with torch.no_grad():
            for batch in val_loader:
                # Uložit batch pro pozdější výběr
                all_batches.append(batch)
                
                # Přesunout data na správné zařízení
                pseudo_healthy = batch['pseudo_healthy'].to(self.device)
                lesion_mask = batch['lesion_mask'].to(self.device)
                real_adc = batch['adc'].to(self.device)
                
                # Forward pass
                with torch.amp.autocast('cuda') if self.use_amp else nullcontext():
                    fake_adc = self.model.generator(pseudo_healthy, lesion_mask)
                    
                    # Získat predikci diskriminátoru pro generovaná data
                    disc_fake_pred = self.model.discriminator(pseudo_healthy, lesion_mask, fake_adc)
                    
                    # Výpočet loss pro generátor se správnými argumenty
                    g_loss, g_adv_loss, g_l1_loss, g_identity_loss, g_ssim_loss, g_edge_loss = \
                        self.model.generator_loss(
                            pseudo_healthy, lesion_mask, real_adc, fake_adc, disc_fake_pred
                        )
                    
                    # Výpočet loss pro diskriminátor
                    # Získat predikci pro reálná data
                    disc_real_pred = self.model.discriminator(pseudo_healthy, lesion_mask, real_adc)
                    d_loss = self.model.discriminator_loss(
                        pseudo_healthy, lesion_mask, real_adc, fake_adc,
                        disc_real_pred, disc_fake_pred
                    )
                
                # Aktualizace metrik
                val_metrics['g_loss'] += g_loss.item()
                val_metrics['g_adv_loss'] += g_adv_loss.item()
                val_metrics['g_l1_loss'] += g_l1_loss.item()
                val_metrics['g_identity_loss'] += g_identity_loss.item()
                val_metrics['g_ssim_loss'] += g_ssim_loss.item()
                val_metrics['g_edge_loss'] += g_edge_loss.item() 
                val_metrics['d_loss'] += d_loss.item()
                
        # Průměrování přes všechny batche
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
            
        # Ukládání do val_log
        self.val_log['epoch'].append(epoch)
        for key in val_metrics:
            self.val_log[key].append(val_metrics[key])
            
        # Vizualizace na náhodně vybraném obrazu
        if all_batches:
            # Náhodně vybrat jeden batch
            random_batch = random.choice(all_batches)
            
            # Vytvořit PDF vizualizaci pro vybraný batch
            pdf_path = self.create_full_volume_pdf_visualization(epoch, random_batch)
            print(f"Full volume PDF visualization created at: {pdf_path}")
            
        return val_metrics
    
    def train(self, train_dataloader, val_dataloader, num_epochs, start_epoch=0, checkpoint_interval=5):
        """
        Trénování modelu
        
        Args:
            train_dataloader: DataLoader pro trénovací data
            val_dataloader: DataLoader pro validační data
            num_epochs (int): Počet epoch
            start_epoch (int): Počáteční epocha (pro pokračování v tréninku)
            checkpoint_interval (int): Interval pro ukládání checkpointu
        """
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, num_epochs):
            # Trénovat jednu epochu
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validovat model
            val_metrics = self.validate(val_dataloader, epoch)
            
            # Vypsat metriky
            print(f"Epocha {epoch}/{num_epochs-1}")
            print(f"  Train: G Loss: {train_metrics['g_loss']:.4f}, D Loss: {train_metrics['d_loss']:.4f}")
            print(f"  Val: G Loss: {val_metrics['g_loss']:.4f}, D Loss: {val_metrics['d_loss']:.4f}")
            print(f"  Val Metriky: SSIM: {val_metrics['g_ssim_loss']:.4f}, MAE: {val_metrics['g_l1_loss']:.4f}")
            
            # Uložit checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_model(epoch)
            
            # Uložit nejlepší model podle validační loss
            if val_metrics['g_loss'] < best_val_loss:
                print(f"  Zlepšení validační loss: {best_val_loss:.4f} -> {val_metrics['g_loss']:.4f}")
                best_val_loss = val_metrics['g_loss']
                self.save_model(epoch, is_best=True)
            
            # Vizualizovat výsledky
            if (epoch + 1) % self.visualize_interval == 0:
                self.visualize_results(epoch, val_dataloader)
        
        # Uložit finální model
        self.save_model(num_epochs - 1)
        
        # Vizualizovat finální výsledky
        self.visualize_results(num_epochs - 1, val_dataloader)
    
    def create_full_volume_pdf_visualization(self, epoch, batch):
        """
        Vytvoří PDF vizualizaci celého objemu pro jeden vzorek z validační sady
        
        Args:
            epoch (int): Aktuální epocha
            batch (dict): Batch dat obsahující 'pseudo_healthy', 'adc', 'lesion_mask' a 'patient_id'
            
        Returns:
            str: Cesta k vytvořenému PDF souboru
        """
        # Přesunout data na správné zařízení
        pseudo_healthy = batch['pseudo_healthy'].to(self.device)
        lesion_mask = batch['lesion_mask'].to(self.device)
        real_adc = batch['adc'].to(self.device)
        patient_id = batch['patient_id'][0]  # Bereme první pacienta z batche
        
        # Generovat falešnou ADC mapu
        with torch.no_grad():
            fake_adc = self.model.generator(pseudo_healthy, lesion_mask)
        
        # Vypočítat metriky
        # Počítáme SSIM v oblasti léze
        ssim_val = self.model.ssim_loss(fake_adc * lesion_mask, real_adc * lesion_mask).item()
        
        # Počítáme MAE (mean absolute error) v oblasti léze
        masked_fake = fake_adc * lesion_mask
        masked_real = real_adc * lesion_mask
        mae_val = torch.sum(torch.abs(masked_fake - masked_real)) / (torch.sum(lesion_mask) + 1e-8)
        mae_val = mae_val.item()
        
        # Cesta k výstupnímu PDF souboru
        pdf_filename = f"epoch_{epoch}_patient_{patient_id}_full_volume.pdf"
        pdf_path = self.pdf_visualization_dir / pdf_filename
        
        # Konvertovat data na CPU a numpy pro vizualizaci
        pseudo_healthy_np = pseudo_healthy.squeeze(0).cpu().numpy()  # [Z, H, W]
        real_adc_np = real_adc.squeeze(0).cpu().numpy()  # [Z, H, W]
        fake_adc_np = fake_adc.squeeze(0).cpu().numpy()  # [Z, H, W]
        lesion_mask_np = lesion_mask.squeeze(0).cpu().numpy()  # [Z, H, W]
        
        num_slices = pseudo_healthy_np.shape[0]
        max_slices_per_page = 5  # Maximální počet řezů na stránku
        
        with PdfPages(pdf_path) as pdf:
            # Pro každou stránku
            for page_start in range(0, num_slices, max_slices_per_page):
                page_end = min(page_start + max_slices_per_page, num_slices)
                slices_on_page = page_end - page_start
                
                # Vytvořit obrázek s řádky (řezy) a sloupci (typy obrazů)
                fig = plt.figure(figsize=(12, 3 * slices_on_page))
                
                # Přidat titulek s metrikami
                fig.suptitle(f'Patient {patient_id}, Epoch {epoch}, SSIM: {ssim_val:.4f}, MAE: {mae_val:.4f}', 
                            fontsize=14)
                
                # Vytvořit grid pro umístění subplotů
                gs = gridspec.GridSpec(slices_on_page, 4, figure=fig)
                
                # Přidat záhlaví pro sloupce
                column_titles = ['Pseudo-Healthy', 'Real ADC', 'Generated ADC', 'Lesion Mask']
                for col, title in enumerate(column_titles):
                    ax = fig.add_subplot(gs[0, col])
                    ax.set_title(title)
                    ax.axis('off')
                    
                    # Pokud je to první sloupec, přidáme texty
                    if col == 0:
                        # Zobrazit první řádek dat
                        ax.imshow(pseudo_healthy_np[page_start], cmap='gray')
                    elif col == 1:
                        ax.imshow(real_adc_np[page_start], cmap='gray')
                    elif col == 2:
                        ax.imshow(fake_adc_np[page_start], cmap='gray')
                    else:
                        ax.imshow(lesion_mask_np[page_start], cmap='gray')
                
                # Projít všechny řezy na aktuální stránce
                for i in range(slices_on_page):
                    slice_idx = page_start + i
                    
                    # Pro každý typ obrazu vytvořit subplot
                    ax1 = fig.add_subplot(gs[i, 0])
                    ax1.imshow(pseudo_healthy_np[slice_idx], cmap='gray')
                    ax1.set_title(f'Slice {slice_idx}')
                    ax1.axis('off')
                    
                    ax2 = fig.add_subplot(gs[i, 1])
                    ax2.imshow(real_adc_np[slice_idx], cmap='gray')
                    ax2.axis('off')
                    
                    ax3 = fig.add_subplot(gs[i, 2])
                    ax3.imshow(fake_adc_np[slice_idx], cmap='gray')
                    ax3.axis('off')
                    
                    ax4 = fig.add_subplot(gs[i, 3])
                    ax4.imshow(lesion_mask_np[slice_idx], cmap='gray')
                    ax4.axis('off')
                
                plt.tight_layout(rect=[0, 0, 1, 0.97])  # Nastavení okrajů pro titulek
                pdf.savefig(fig)
                plt.close(fig)
        
        return str(pdf_path)


def train_model(args):
    """Funkce pro trénování modelu s parametry z args"""
    # Nastavit seed pro reprodukovatelnost
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Vytvořit výstupní adresář
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Zjistit typickou velikost dat pro nastavení modelu
    # Načteme jeden soubor pro zjištění rozměrů
    sample_file = next(Path(args.pseudo_healthy_dir).glob("*PSEUDO_HEALTHY.mha"))
    sample_img = sitk.ReadImage(str(sample_file))
    sample_data = sitk.GetArrayFromImage(sample_img)
    
    # Zjistíme nejbližší násobek 32 pro každý rozměr
    d, h, w = sample_data.shape
    patch_size = 32
    target_d = ((d + patch_size - 1) // patch_size) * patch_size
    target_h = ((h + patch_size - 1) // patch_size) * patch_size
    target_w = ((w + patch_size - 1) // patch_size) * patch_size
    
    # Nastavit cílovou velikost pro zpracování
    target_size = (target_d, target_h, target_w)
    print(f"Původní velikost dat prvního vzorku: {sample_data.shape}")
    print(f"Zaokrouhleno na násobky {patch_size}: {target_size}")
    print(f"Tato velikost bude použita pro inicializaci modelu, ale dataset bude dynamicky zpracovávat každý vzorek individuálně.")
    
    if args.target_size:
        # Použít zadanou cílovou velikost, pokud je specifikována
        target_size = tuple(map(int, args.target_size.split(',')))
        print(f"Použita ručně specifikovaná cílová velikost pro model: {target_size}")
    
    # Definovat transformace pro augmentaci dat (pouze pro tréninkovou část)
    train_transforms = None
    if args.use_augmentations:
        from monai.transforms import (
            RandRotate90d,
            RandFlipd,
            RandAffined,
            RandScaleIntensityd,
            RandShiftIntensityd,
            RandGaussianNoised,
            Compose
        )
        
        # Klíče pro všechny data tensory v sample
        keys = ['pseudo_healthy', 'adc', 'lesion_mask']
        
        # Definice transformací pro data augmentaci
        train_transforms = Compose([
            # Náhodné rotace o 90 stupňů v rovině (2D řezy)
            RandRotate90d(
                keys=keys,
                prob=args.aug_rotate_prob,
                max_k=3,  # maximálně 3 rotace (0, 90, 180, 270 stupňů)
                spatial_axes=(1, 2)  # rotace v osách y-z (2D řezy)
            ),
            
            # Náhodné překlápění (zrcadlení)
            RandFlipd(
                keys=keys,
                prob=args.aug_flip_prob,
                spatial_axis=None  # náhodný výběr os
            ),
            
            # Affinní transformace (rotace, škálování, posuny)
            RandAffined(
                keys=keys,
                prob=args.aug_affine_prob,
                rotate_range=(np.pi/36, np.pi/36, np.pi/36),  # max +/- 5 stupňů ve všech osách
                scale_range=(0.05, 0.05, 0.05),  # škálování o +/- 5%
                mode=('bilinear', 'bilinear', 'nearest'),  # interpolace pro každý typ dat
                padding_mode='zeros'
            ),
            
            # Úpravy intenzit - jen pro intenzitní mapy, ne pro masky
            RandScaleIntensityd(
                keys=['pseudo_healthy', 'adc'],
                prob=args.aug_intensity_prob,
                factors=0.1  # násobení intenzit faktorem v rozsahu [0.9, 1.1]
            ),
            
            # Posuvy intenzit
            RandShiftIntensityd(
                keys=['pseudo_healthy', 'adc'],
                prob=args.aug_intensity_prob,
                offsets=0.1  # přičtení hodnoty v rozsahu [-0.1, 0.1]
            ),
            
            # Gaussovský šum
            RandGaussianNoised(
                keys=['pseudo_healthy', 'adc'],
                prob=args.aug_noise_prob,
                mean=0.0,
                std=0.05  # standardní odchylka šumu
            )
        ])
        
        print("Aktivovány augmentace dat pro trénink.")
    
    # Debug výpis pro zobrazení dostupných souborů v adresářích
    print("Kontrola souborů v adresářích:")
    adc_files = list(Path(args.adc_dir).glob('*.mha'))
    lesion_files = list(Path(args.lesion_mask_dir).glob('*.mha'))
    print(f"ADC soubory: {adc_files[:5] if adc_files else []}... (zobrazeno prvních 5)")
    print(f"Lesion mask soubory: {lesion_files[:5] if lesion_files else []}... (zobrazeno prvních 5)")
    
    # Načíst dataset
    train_dataset = LesionInpaintingDataset(
        pseudo_healthy_dir=args.pseudo_healthy_dir,
        adc_dir=args.adc_dir,
        lesion_mask_dir=args.lesion_mask_dir,
        norm_range=(0, 1),
        crop_foreground=args.crop_foreground,
        transform=train_transforms
    )
    
    # Kontrola, zda máme dostatek dat pro trénink
    if len(train_dataset.file_triplets) == 0:
        print("ERROR: Nebyly nalezeny žádné kompletní triplety souborů pro trénink.")
        print("Zkontrolujte cesty k adresářům a formáty názvů souborů.")
        print("Očekávaný formát ADC souborů: {patient_id}*-ADC_ss.mha")
        print("Očekávaný formát lesion mask souborů: {patient_id}*_lesion.mha")
        print("Ukončuji trénink.")
        return
    
    # Rozdělit dataset na trénovací a validační část
    dataset_size = len(train_dataset)
    val_size = max(1, int(dataset_size * args.val_ratio))
    train_size = dataset_size - val_size
    
    # Zajistit, že máme alespoň jeden vzorek pro trénování a jeden pro validaci
    if train_size <= 0 or val_size <= 0:
        print(f"ERROR: Nedostatek dat pro rozdělení na trénovací a validační část. Dataset má pouze {dataset_size} vzorků.")
        print("Upravte val_ratio nebo použijte větší dataset.")
        print("Ukončuji trénink.")
        return
    
    # Pro validační dataset nepoužíváme augmentace, musíme proto vytvořit nový dataset pro validaci
    if args.use_augmentations and val_size > 0:
        # Vytvoříme indexy pro rozdělení datasetu
        indices = list(range(dataset_size))
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Vytvoříme zvlášť validační dataset bez augmentací
        val_dataset = LesionInpaintingDataset(
            pseudo_healthy_dir=args.pseudo_healthy_dir,
            adc_dir=args.adc_dir,
            lesion_mask_dir=args.lesion_mask_dir,
            norm_range=(0, 1),
            crop_foreground=args.crop_foreground,
            transform=None  # bez augmentací pro validaci
        )
        
        # Použijeme SubsetRandomSampler pro vytvoření dataloaderu pouze s požadovanými indexy
        from torch.utils.data import SubsetRandomSampler
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # Dataloadery s příslušnými samplery
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,  # Pro validaci je vhodnější batch size 1
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        # Pokud nepoužíváme augmentace nebo nemáme validaci, použijeme původní přístup
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,  # Pro validaci je vhodnější batch size 1
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    print(f"Velikost trénovacího datasetu: {train_size}")
    print(f"Velikost validačního datasetu: {val_size}")
    
    # Inicializovat model s novými rozměry
    model = LesionInpaintingGAN(
        img_size=target_size,
        gen_features=args.gen_features,
        disc_base_filters=args.disc_base_filters,
        disc_n_layers=args.disc_n_layers,
        lambda_l1=args.lambda_l1,
        lambda_identity=args.lambda_identity,
        lambda_ssim=args.lambda_ssim,
        lambda_edge=args.lambda_edge
    ).to(device)
    
    # Vytvořit optimalizátory
    optimizer_g = torch.optim.Adam(
        model.generator.parameters(),
        lr=args.lr_g,
        betas=(args.beta1, args.beta2)
    )
    
    optimizer_d = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=args.lr_d,
        betas=(args.beta1, args.beta2)
    )
    
    # Vytvořit scheduler pro learning rate
    if args.use_lr_scheduler:
        scheduler_g = torch.optim.lr_scheduler.StepLR(
            optimizer_g, step_size=args.lr_step, gamma=args.lr_gamma
        )
        
        scheduler_d = torch.optim.lr_scheduler.StepLR(
            optimizer_d, step_size=args.lr_step, gamma=args.lr_gamma
        )
    else:
        scheduler_g = None
        scheduler_d = None
    
    # Inicializovat trainer
    trainer = LesionInpaintingTrainer(
        model=model,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=device,
        output_dir=args.output_dir,
        use_amp=args.use_amp,
        visualize_interval=args.visualize_interval
    )
    
    # Načíst checkpoint, pokud je zadán
    start_epoch = 0
    if args.checkpoint:
        start_epoch = trainer.load_model(args.checkpoint)
    
    # Trénovat model
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        start_epoch=start_epoch,
        checkpoint_interval=args.checkpoint_interval
    )


def save_mha_file(data, reference_image, output_path):
    """Uloží numpy array jako MHA soubor s použitím metadat z reference_image"""
    out_img = sitk.GetImageFromArray(data)
    out_img.CopyInformation(reference_image)
    sitk.WriteImage(out_img, str(output_path))


def generate_inpainted_brain(model, pseudo_healthy_path, lesion_mask_path, output_path, existing_lesion_path=None, output_combined_mask_path=None, device='cuda'):
    """
    Generuje inpaintovaný mozek s lézí pomocí naučeného modelu
    
    Args:
        model (LesionInpaintingGAN): Naučený model
        pseudo_healthy_path (str): Cesta k pseudo-zdravému mozku
        lesion_mask_path (str): Cesta k masce léze, kterou chceme přidat
        output_path (str): Cesta pro uložení výstupu (ADC mapa s lézemi)
        existing_lesion_path (str, optional): Cesta k masce již existujících lézí
        output_combined_mask_path (str, optional): Cesta pro uložení kombinované masky lézí
        device (str): Zařízení pro inferenci
    """
    # Načíst data
    ph_img = sitk.ReadImage(str(pseudo_healthy_path))
    ph_data = sitk.GetArrayFromImage(ph_img).astype(np.float32)
    
    mask_img = sitk.ReadImage(str(lesion_mask_path))
    mask_data = sitk.GetArrayFromImage(mask_img).astype(np.float32)
    
    # Binarizovat masku nové léze
    mask_data = (mask_data > 0).astype(np.float32)
    
    # Načíst existující léze, pokud jsou specifikovány
    if existing_lesion_path:
        existing_lesion_img = sitk.ReadImage(str(existing_lesion_path))
        existing_lesion_data = sitk.GetArrayFromImage(existing_lesion_img).astype(np.float32)
        existing_lesion_data = (existing_lesion_data > 0).astype(np.float32)
        
        # Vytvořit kombinovanou masku, která zachovává existující léze
        # a přidává nové léze pouze tam, kde neexistují původní
        combined_mask_data = np.maximum(existing_lesion_data, mask_data)
        
        # Vytvořit masku pro nové léze (pouze oblasti, kde nejsou existující léze)
        new_lesion_mask_data = mask_data * (1 - existing_lesion_data)
    else:
        # Pokud neexistují existující léze, použijeme původní masku nové léze
        combined_mask_data = mask_data
        new_lesion_mask_data = mask_data
    
    # Normalizovat hodnoty
    ph_data = np.clip(ph_data, 0, None)
    if np.max(ph_data) - np.min(ph_data) > 0:
        ph_data = (ph_data - np.min(ph_data)) / (np.max(ph_data) - np.min(ph_data))
    
    # Uložit původní tvar pro pozdější obnovení
    original_shape = ph_data.shape
    
    # Dynamicky doplnit na nejbližší násobek 32 pro každý rozměr
    d, h, w = ph_data.shape
    patch_size = 32
    target_d = ((d + patch_size - 1) // patch_size) * patch_size
    target_h = ((h + patch_size - 1) // patch_size) * patch_size
    target_w = ((w + patch_size - 1) // patch_size) * patch_size
    
    print(f"Původní velikost dat: {original_shape}")
    print(f"Paddovaná velikost pro inference: ({target_d}, {target_h}, {target_w})")
    
    # Pad data
    pad_d = max(0, target_d - d)
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    # Aplikovat padding symetricky
    pad_d_before, pad_d_after = pad_d // 2, pad_d - (pad_d // 2)
    pad_h_before, pad_h_after = pad_h // 2, pad_h - (pad_h // 2)
    pad_w_before, pad_w_after = pad_w // 2, pad_w - (pad_w // 2)
    
    # Padding pro pseudo-zdravý mozek
    padded_ph_data = np.pad(
        ph_data,
        ((pad_d_before, pad_d_after), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
        mode='constant',
        constant_values=0
    )
    
    # Padding pro masku nové léze (bez existujících lézí)
    padded_new_lesion_mask_data = np.pad(
        new_lesion_mask_data,
        ((pad_d_before, pad_d_after), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
        mode='constant',
        constant_values=0
    )
    
    # Převést na torch tensor a přidat dimenze pro batch a kanál
    ph_tensor = torch.from_numpy(padded_ph_data).unsqueeze(0).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(padded_new_lesion_mask_data).unsqueeze(0).unsqueeze(0).to(device)
    
    # Generovat inpaintovaný mozek pouze pro oblasti nových lézí
    model.eval()
    with torch.no_grad():
        inpainted = model(ph_tensor, mask_tensor)
    
    # Převést zpět na numpy array
    inpainted_data = inpainted.squeeze().cpu().numpy()
    
    # Ořezat zpět na původní velikost
    inpainted_data = inpainted_data[
        pad_d_before:pad_d_before + d,
        pad_h_before:pad_h_before + h,
        pad_w_before:pad_w_before + w
    ]
    
    # Aplikovat původní existující léze na výstup, pokud existují
    if existing_lesion_path:
        # Získat původní data s existujícími lézemi
        original_adc_data = ph_data.copy()  # Zde případně načíst původní ADC s existujícími lézemi
        
        # Kombinovat výsledek: zachováme existující léze a přidáme nové jen tam, kde nejsou existující
        final_output = (1 - existing_lesion_data) * inpainted_data + existing_lesion_data * original_adc_data
    else:
        final_output = inpainted_data
    
    # Ověřit, že jsme zpět na původní velikosti
    assert final_output.shape == original_shape, \
        f"Velikost výstupu {final_output.shape} neodpovídá původní velikosti {original_shape}"
    
    # Uložit výsledek jako MHA soubor
    save_mha_file(final_output, ph_img, output_path)
    
    # Uložit kombinovanou masku lézí, pokud je specifikována cesta
    if output_combined_mask_path:
        save_mha_file(combined_mask_data, ph_img, output_combined_mask_path)
    
    print(f"Inpaintovaný mozek uložen do {output_path}")
    if output_combined_mask_path:
        print(f"Kombinovaná maska lézí uložena do {output_combined_mask_path}")


def main():
    # Vytvořit parser argumentů
    parser = argparse.ArgumentParser(description='HIE Lesion Inpainting GAN')
    
    # Cesty k datům
    parser.add_argument('--pseudo_healthy_dir', type=str, required=True,
                        help='Adresář s pseudo-zdravými ADC mapami')
    parser.add_argument('--adc_dir', type=str, required=True,
                        help='Adresář s ADC mapami s lézemi')
    parser.add_argument('--lesion_mask_dir', type=str, required=True,
                        help='Adresář s binárními maskami lézí')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Výstupní adresář pro ukládání výsledků')
    
    # Parametry tréninku
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Počet epoch pro trénink')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Velikost batche pro trénink (pro full volume je typicky 1-2)')
    parser.add_argument('--crop_foreground', action='store_true',
                        help='Ořezat objemy na bounding box mozku')
    parser.add_argument('--target_size', type=str, default=None,
                        help='Cílová velikost výstupních objemů, např. "32,128,128"')
    parser.add_argument('--lr_g', type=float, default=0.0002,
                        help='Learning rate pro generátor')
    parser.add_argument('--lr_d', type=float, default=0.0001,
                        help='Learning rate pro diskriminátor')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 parametr pro Adam optimalizátor')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 parametr pro Adam optimalizátor')
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help='Použít scheduler pro learning rate')
    parser.add_argument('--lr_step', type=int, default=30,
                        help='Počet epoch pro step LR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Faktor snížení learning rate pro scheduler')
    parser.add_argument('--val_ratio', type=float, default=0.12,
                        help='Poměr dat pro validaci')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='Interval epoch pro ukládání checkpointu')
    parser.add_argument('--visualize_interval', type=int, default=5,
                        help='Interval epoch pro vizualizaci výsledků')
    parser.add_argument('--use_amp', action='store_true',
                        help='Použít Automatic Mixed Precision')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Počet workerů pro data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed pro reprodukovatelnost')
    
    # Parametry augmentací
    parser.add_argument('--use_augmentations', action='store_true',
                        help='Použít augmentace dat pro trénink')
    parser.add_argument('--aug_rotate_prob', type=float, default=0.3,
                        help='Pravděpodobnost použití náhodné rotace')
    parser.add_argument('--aug_flip_prob', type=float, default=0.3,
                        help='Pravděpodobnost použití náhodného zrcadlení')
    parser.add_argument('--aug_affine_prob', type=float, default=0.2,
                        help='Pravděpodobnost použití affinních transformací')
    parser.add_argument('--aug_intensity_prob', type=float, default=0.3,
                        help='Pravděpodobnost úpravy intenzit jasu')
    parser.add_argument('--aug_noise_prob', type=float, default=0.2,
                        help='Pravděpodobnost přidání Gaussovského šumu')
    
    # Parametry modelu
    parser.add_argument('--gen_features', type=int, default=48,
                        help='Počet featur pro generátor')
    parser.add_argument('--disc_base_filters', type=int, default=64,
                        help='Počet základních filtrů pro diskriminátor')
    parser.add_argument('--disc_n_layers', type=int, default=4,
                        help='Počet vrstev diskriminátoru')
    
    # Parametry loss funkcí
    parser.add_argument('--lambda_l1', type=float, default=10.0,
                        help='Váha L1 loss funkce')
    parser.add_argument('--lambda_identity', type=float, default=5.0,
                        help='Váha identity loss funkce')
    parser.add_argument('--lambda_ssim', type=float, default=1.0,
                        help='Váha SSIM loss funkce')
    parser.add_argument('--lambda_edge', type=float, default=1.0,
                        help='Váha edge loss funkce')
    
    # Inferenční mód
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='Mód běhu (trénink nebo inference)')
    parser.add_argument('--checkpoint', type=str,
                        help='Cesta k checkpointu pro načtení')
    parser.add_argument('--inference_input', type=str,
                        help='Cesta k pseudo-zdravému mozku pro inferenci')
    parser.add_argument('--inference_mask', type=str,
                        help='Cesta k masce léze pro inferenci')
    parser.add_argument('--inference_output', type=str,
                        help='Cesta pro uložení výsledku inference')
    parser.add_argument('--existing_lesion_mask', type=str,
                        help='Cesta k masce existujících lézí (volitelné)')
    parser.add_argument('--output_combined_mask', type=str,
                        help='Cesta pro uložení kombinované masky lézí (volitelné)')
    
    args = parser.parse_args()
    
    # Spustit trénink nebo inferenci podle zadaného módu
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'inference':
        if not args.checkpoint or not args.inference_input or not args.inference_mask or not args.inference_output:
            print("Pro inferenci je nutné zadat --checkpoint, --inference_input, --inference_mask a --inference_output")
            return
        
        # Načíst model
        # Pro inferenci potřebujeme znát velikost dat, se kterými byl model natrénován
        # - to by bylo ideálně uložené v checkpointu, ale pro teď použijeme detekci
        sample_img = sitk.ReadImage(args.inference_input)
        sample_data = sitk.GetArrayFromImage(sample_img)
        d, h, w = sample_data.shape
        patch_size = 32
        target_d = ((d + patch_size - 1) // patch_size) * patch_size
        target_h = ((h + patch_size - 1) // patch_size) * patch_size
        target_w = ((w + patch_size - 1) // patch_size) * patch_size
        target_size = (target_d, target_h, target_w)
        
        # Inicializovat model s odpovídající velikostí
        model = LesionInpaintingGAN(
            img_size=target_size,
            # Tyto hodnoty ideálně načíst z checkpointu nebo nastavit stejně jako při tréninku
            gen_features=args.gen_features,  
            disc_base_filters=args.disc_base_filters,
            disc_n_layers=args.disc_n_layers,
            lambda_l1=args.lambda_l1,
            lambda_identity=args.lambda_identity,
            lambda_ssim=args.lambda_ssim,
            lambda_edge=args.lambda_edge
        ).to(device)
        
        checkpoint = torch.load(args.checkpoint)
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        # Generovat inpaintovaný mozek
        generate_inpainted_brain(
            model=model,
            pseudo_healthy_path=args.inference_input,
            lesion_mask_path=args.inference_mask,
            output_path=args.inference_output,
            existing_lesion_path=args.existing_lesion_mask,
            output_combined_mask_path=args.output_combined_mask,
            device=device
        )


if __name__ == "__main__":
    main()
