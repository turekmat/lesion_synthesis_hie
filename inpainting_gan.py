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
    
    Tato verze pracuje s patchi zaměřenými na oblasti léze.
    """
    def __init__(
        self,
        pseudo_healthy_dir,
        adc_dir,
        lesion_mask_dir,
        transform=None,
        norm_range=(0, 1),
        crop_foreground=True,
        target_size=None,
        patch_size=(64, 64, 32),  # Velikost patchů pro trénink
        use_patches=True  # Příznak, zda používat patche nebo celé objemy
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
            patch_size (tuple): Velikost patchů pro trénink (D, H, W)
            use_patches (bool): Pokud True, bude používat patche místo celých objemů
        """
        self.pseudo_healthy_dir = Path(pseudo_healthy_dir)
        self.adc_dir = Path(adc_dir)
        self.lesion_mask_dir = Path(lesion_mask_dir)
        self.transform = transform
        self.norm_range = norm_range
        self.crop_foreground = crop_foreground
        self.target_size = target_size
        self.patch_size = patch_size
        self.use_patches = use_patches
        
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
        
        if self.use_patches:
            # Pokud používáme patche, připravíme si seznam všech patchů
            self.patches = []
            
            print("Připravuji seznam patchů zaměřených na oblasti lézí...")
            for triplet_idx, triplet in enumerate(self.file_triplets):
                # Načíst masku léze - potřebujeme najít oblasti s lézí
                lesion_mask_data, _ = self._load_and_preprocess(triplet['lesion_mask'])
                
                # Binarizovat masku léze
                lesion_mask_data = (lesion_mask_data > 0).astype(np.float32)
                
                # Zjistit, jestli maska obsahuje nějaké léze
                if np.sum(lesion_mask_data) > 0:
                    # Najít souřadnice lézí
                    lesion_coords = np.where(lesion_mask_data > 0)
                    
                    # Pro každý bod léze vytvoříme patch
                    # Omezíme počet patchů pro velké léze, aby nedošlo k duplikátům
                    # a příliš velkému datasetu
                    max_points = 10  # Maximální počet patchů pro jednu lézi
                    step = max(1, len(lesion_coords[0]) // max_points)
                    
                    for i in range(0, len(lesion_coords[0]), step):
                        z, y, x = lesion_coords[0][i], lesion_coords[1][i], lesion_coords[2][i]
                        
                        # Zaznamenat informace o patchích
                        self.patches.append({
                            'triplet_idx': triplet_idx,
                            'center': (z, y, x),
                            'patient_id': triplet['patient_id']
                        })
            
            print(f"Celkem vytvořeno {len(self.patches)} patchů zaměřených na léze")
    
    def __len__(self):
        if self.use_patches:
            return len(self.patches)
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
    
    def _extract_patch(self, volume, center, patch_size):
        """Extrahuje patch z objemu podle zadaného středu a velikosti"""
        z, y, x = center
        d, h, w = patch_size
        
        # Vypočítat hranice patche
        z_start = max(0, z - d // 2)
        y_start = max(0, y - h // 2)
        x_start = max(0, x - w // 2)
        
        z_end = min(volume.shape[0], z_start + d)
        y_end = min(volume.shape[1], y_start + h)
        x_end = min(volume.shape[2], x_start + w)
        
        # Extrahovat patch
        patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pokud je patch menší než požadovaná velikost, doplníme ho nulami
        if patch.shape != patch_size:
            # Vytvořit prázdný patch požadované velikosti
            padded_patch = np.zeros(patch_size, dtype=patch.dtype)
            
            # Vložit extrahovaný patch do prázdného
            padded_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            
            return padded_patch
        
        return patch
    
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
        if self.use_patches:
            # Patch-based přístup
            patch_info = self.patches[idx]
            triplet_idx = patch_info['triplet_idx']
            center = patch_info['center']
            patient_id = patch_info['patient_id']
            
            # Získat cestu k souborům
            triplet = self.file_triplets[triplet_idx]
            
            # Načíst a předzpracovat data
            pseudo_healthy_data, ph_img = self._load_and_preprocess(triplet['pseudo_healthy'])
            adc_data, _ = self._load_and_preprocess(triplet['adc'])
            lesion_mask_data, _ = self._load_and_preprocess(triplet['lesion_mask'])
            
            # Binarizovat masku léze
            lesion_mask_data = (lesion_mask_data > 0).astype(np.float32)
            
            # Normalizovat hodnoty
            pseudo_healthy_data = self._normalize(pseudo_healthy_data)
            adc_data = self._normalize(adc_data)
            
            # Extrahovat patche
            pseudo_healthy_patch = self._extract_patch(pseudo_healthy_data, center, self.patch_size)
            adc_patch = self._extract_patch(adc_data, center, self.patch_size)
            lesion_mask_patch = self._extract_patch(lesion_mask_data, center, self.patch_size)
            
            # Vytvořit sample slovník pro MONAI transformace
            sample = {
                'pseudo_healthy': pseudo_healthy_patch,
                'adc': adc_patch,
                'lesion_mask': lesion_mask_patch,
                'patient_id': patient_id,
                'volume_shape': pseudo_healthy_data.shape,  # Uložit původní rozměr pro pozdější použití
                'patch_center': center  # Uložit pozici patche v původním objemu
            }
        else:
            # Původní přístup s celými objemy
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
                'patient_id': triplet['patient_id'],
                'volume_shape': pseudo_healthy_data.shape  # Uložit původní rozměr pro pozdější použití
            }
        
        # Aplikovat transformace, pokud jsou definovány
        if self.transform:
            # Přidat debug informace o tvaru před transformací
            if idx == 0:  # Jen pro první vzorek pro omezení výpisů
                print(f"DEBUG: Tvary před transformací: pseudo_healthy={sample['pseudo_healthy'].shape}, "
                      f"adc={sample['adc'].shape}, lesion_mask={sample['lesion_mask'].shape}")
                print(f"DEBUG: Typy před transformací: pseudo_healthy={type(sample['pseudo_healthy'])}, "
                      f"adc={type(sample['adc'])}, lesion_mask={type(sample['lesion_mask'])}")
            
            sample = self.transform(sample)
            
            # Přidat debug informace o tvaru po transformaci
            if idx == 0:  # Jen pro první vzorek pro omezení výpisů
                print(f"DEBUG: Tvary po transformaci: pseudo_healthy={sample['pseudo_healthy'].shape}, "
                      f"adc={sample['adc'].shape}, lesion_mask={sample['lesion_mask'].shape}")
                print(f"DEBUG: Typy po transformaci: pseudo_healthy={type(sample['pseudo_healthy'])}, "
                      f"adc={type(sample['adc'])}, lesion_mask={type(sample['lesion_mask'])}")
        
        # Převést data na torch tenzory s přidáním kanálové dimenze
        for key in ['pseudo_healthy', 'adc', 'lesion_mask']:
            if isinstance(sample[key], np.ndarray):
                # Převést numpy array na torch tensor a přidat kanálovou dimenzi
                sample[key] = torch.from_numpy(sample[key]).unsqueeze(0)
            elif isinstance(sample[key], torch.Tensor) and sample[key].ndim == 3:
                # Pokud je to už tensor, ale bez kanálové dimenze, přidáme ji
                sample[key] = sample[key].unsqueeze(0)
        
        # Přidat debug informace o konečném tvaru
        if idx == 0:  # Jen pro první vzorek pro omezení výpisů
            print(f"DEBUG: Finální tvary: pseudo_healthy={sample['pseudo_healthy'].shape}, "
                  f"adc={sample['adc'].shape}, lesion_mask={sample['lesion_mask'].shape}")
        
        # Print each 50th sample shape to track diversity
        if idx % 50 == 0:
            print(f"Sample {idx} shapes: pseudo_healthy={sample['pseudo_healthy'].shape}, "
                  f"adc={sample['adc'].shape}, lesion_mask={sample['lesion_mask'].shape}")
        
        return sample

class Generator(nn.Module):
    """
    UNet-like generátor pro inpainting lézí
    Vstup: Pseudo-zdravý mozek a maska léze
    Výstup: ADC mapa léze
    """
    def __init__(
        self, 
        in_channels=2,  # 1 pro pseudo-zdravý + 1 pro masku léze
        out_channels=1,  # 1 pro ADC mapu
        base_filters=64,
        depth=4,  # Snížíme hloubku pro lepší stabilitu s různými vstupy
        min_size=32
    ):
        super().__init__()
        
        # Přidáme kontrolu minimální velikosti vstupu
        self.min_input_size = min_size
        
        # Enkodér - postupně snižuje rozlišení a zvyšuje počet kanálů
        self.enc_blocks = nn.ModuleList()
        self.enc_channels = []  # Pro uchování počtu kanálů v jednotlivých vrstvách

        # První blok enkodéru (ne-stride konvoluce)
        self.enc_blocks.append(
            nn.Sequential(
                nn.Conv3d(in_channels, base_filters, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(base_filters),
                nn.ReLU(inplace=True),
                nn.Conv3d(base_filters, base_filters, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(base_filters),
                nn.ReLU(inplace=True)
            )
        )
        self.enc_channels.append(base_filters)
        
        # Další bloky enkodéru (konvoluce se stride)
        current_size = min_size
        for i in range(1, depth):
            in_channels = base_filters * (2 ** (i - 1))
            out_channels = base_filters * (2 ** i)
            
            # Pokud by další downsampling vedl k příliš malé velikosti, přestaneme
            if current_size // 2 < 4:  # Minimální velikost 4x4x4
                break
                
            current_size = current_size // 2
            
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.enc_channels.append(out_channels)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(self.enc_channels[-1], self.enc_channels[-1]*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(self.enc_channels[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.enc_channels[-1]*2, self.enc_channels[-1]*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(self.enc_channels[-1]*2),
            nn.ReLU(inplace=True)
        )
        
        # Dekodér - postupně zvyšuje rozlišení a snižuje počet kanálů
        self.dec_blocks = nn.ModuleList()
        self.dec_channels = []
        
        # První decoder blok (zpracovává bottleneck)
        self.dec_blocks.append(
            nn.Sequential(
                nn.Conv3d(self.enc_channels[-1]*2, self.enc_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(self.enc_channels[-1]),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.enc_channels[-1], self.enc_channels[-1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(self.enc_channels[-1]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            )
        )
        self.dec_channels.append(self.enc_channels[-1])

        # Další bloky dekodéru se skip connections
        for i in range(len(self.enc_channels) - 1, 0, -1):
            if i == 1:  # Poslední blok dekodéru
                dec_in = self.dec_channels[-1] + self.enc_channels[i-1]
                
                # Menší počet kanálů v posledním bloku, neupsamplujeme
                self.dec_blocks.append(
                    nn.Sequential(
                        nn.Conv3d(dec_in, dec_in // 2, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm3d(dec_in // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(dec_in // 2, dec_in // 2, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm3d(dec_in // 2),
                        nn.ReLU(inplace=True)
                    )
                )
                self.dec_channels.append(dec_in // 2)
            else:
                dec_in = self.dec_channels[-1] + self.enc_channels[i-1]
                dec_out = self.enc_channels[i-1]
                
                self.dec_blocks.append(
                    nn.Sequential(
                        nn.Conv3d(dec_in, dec_out, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm3d(dec_out),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(dec_out, dec_out, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm3d(dec_out),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
                    )
                )
                self.dec_channels.append(dec_out)
        
        # Poslední úprava pro finální výstup - jeden blok bez upsample
        if len(self.dec_blocks) < len(self.enc_blocks):
            if self.dec_blocks:
                dec_in = self.dec_channels[-1] + self.enc_channels[0]
                dec_out = base_filters // 2  # Half of initial filters
                
                # Poslední decoder
                self.dec_blocks.append(
                    nn.Sequential(
                        # Zredukujeme počet kanálů po konkatenaci s prvním enkodérem
                        nn.Conv3d(dec_in, base_filters, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm3d(base_filters),
                        nn.ReLU(inplace=True)
                    )
                )
                self.dec_channels.append(base_filters)
                
        # Attention modul pro zaměření na léze
        self.attention = nn.Sequential(
            nn.Conv3d(self.dec_channels[-1] + 1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Finální konvoluce pro generování výstupu 
        self.final_conv = nn.Sequential(
            nn.Conv3d(self.dec_channels[-1], out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Normalizovaný výstup v rozmezí [-1, 1]
        )
    
    def forward(self, pseudo_healthy, lesion_mask):
        """
        Forward pass generátoru s vylepšeným přechodem na hraně masky
        
        Args:
            pseudo_healthy (torch.Tensor): Tensor pseudo-zdravého mozku [B, 1, D, H, W] nebo [B, D, H, W]
            lesion_mask (torch.Tensor): Tensor binární masky léze [B, 1, D, H, W] nebo [B, D, H, W]
        
        Returns:
            torch.Tensor: Generovaná ADC mapa léze [B, 1, D, H, W]
        """
        # Kontrola a úprava tvaru vstupů - zajistíme 5D tensory [B, C, D, H, W]
        if pseudo_healthy.ndim == 4:
            pseudo_healthy = pseudo_healthy.unsqueeze(1)  # Přidáme kanálovou dimenzi
            
        if lesion_mask.ndim == 4:
            lesion_mask = lesion_mask.unsqueeze(1)  # Přidáme kanálovou dimenzi
        
        # Kontrola minimální velikosti vstupu
        min_size = self.min_input_size
        d, h, w = pseudo_healthy.shape[2:5]
        
        if d < min_size or h < min_size or w < min_size:
            print(f"Warning: Input size {(d, h, w)} is smaller than minimum size {min_size}. Results may be unpredictable.")
        
        # Spojit vstupy pro generátor
        x = torch.cat([pseudo_healthy, lesion_mask], dim=1)
        
        # Encoder - uložíme všechny výstupy pro skip connections
        enc_features = []
        for block in self.enc_blocks:
            x = block(x)
            enc_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder s přeskočenými spojeními
        for i, block in enumerate(self.dec_blocks):
            # Pro všechny vrstvy kromě poslední provádíme konkatenaci
            if i > 0:  # První decoder nemá skip connection (zpracovává bottleneck)
                # Index feature mapy z enkodéru, kterou chceme použít pro skip connection
                # Jdeme od konce k začátku
                skip_idx = len(enc_features) - i - 1
                skip = enc_features[skip_idx]
                
                # Interpolace, pokud je potřeba
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
                
                # Konkatenace se skip connection
                x = torch.cat([x, skip], dim=1)
            
            # Aplikujeme decoder blok
            x = block(x)
        
        # Aplikace attention mechanismu na finální feature mapu
        # Konkatenujeme masku léze s feature mapou pro attention model
        attention_input = torch.cat([x, F.interpolate(lesion_mask, size=x.shape[2:], mode='nearest')], dim=1)
        attention_map = self.attention(attention_input)
        
        # Aplikujeme attention mapu na feature mapu - více se soustředíme na oblasti s lézí
        x = x * attention_map + x  # Reziduální spojení pro stabilitu
        
        # Finální konvoluce pro generování výstupu
        raw_output = self.final_conv(x)
        
        # === Vylepšení hranového přechodu ===
        # Vytvoříme plynulý přechod na hranici masky pro přirozenější inpainting
        
        # 1. Dilatace a eroze masky pro vytvoření hranového pásu
        kernel_size = 5  # Velikost jádra pro dilataci/erozi
        padding = kernel_size // 2
        
        # Dilatace masky
        dilated_mask = F.max_pool3d(lesion_mask, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Eroze masky (1 - max_pool3d(1 - mask))
        eroded_mask = 1 - F.max_pool3d(1 - lesion_mask, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Hranice masky je rozdíl mezi dilatovanou a erodovanou maskou
        edge_mask = dilated_mask - eroded_mask
        
        # 2. Vytvoření váhované přechodové masky pro plynulé prolínání
        # Použijeme Gaussovo rozmazání na masku léze pro vytvoření plynulého přechodu
        
        # Vytvoříme přechodovou masku s jemnějším přechodem
        # Nejprve vytvoříme základní přechodovou masku
        transition_mask = (dilated_mask - eroded_mask) * 0.5 + eroded_mask
        
        # Aplikujeme další vyhlazení - vytvoříme jemnější verzi pomocí 3D konvoluce
        # Simulujeme Gaussův filtr pomocí průměrové konvoluce
        smooth_kernel_size = 7
        smooth_padding = smooth_kernel_size // 2
        smooth_transition_mask = F.avg_pool3d(transition_mask, 
                                              kernel_size=smooth_kernel_size, 
                                              stride=1, 
                                              padding=smooth_padding)
        
        # 3. Finální kombinace s plynulým přechodem
        # - vnitřek léze: použijeme raw_output
        # - hranice léze: vážený průměr mezi raw_output a pseudo_healthy
        # - mimo lézi: použijeme pseudo_healthy (zajistí identity loss v oblastech mimo lézi)
        
        # Aplikujeme vyhlazený přechod s větším důrazem na zachování původního obrazu mimo lézi
        final_output = pseudo_healthy * (1 - smooth_transition_mask) + raw_output * smooth_transition_mask
        
        # Zajistíme, že výstup má správný počet kanálů [B, 1, D, H, W]
        if final_output.shape[1] != 1:
            print(f"Warning: Generator output has incorrect channel dimension: {final_output.shape}. Adjusting to [B, 1, D, H, W]")
            final_output = final_output[:, 0:1]
            
        return final_output


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
            pseudo_healthy (torch.Tensor): Tensor pseudo-zdravého mozku [B, 1, D, H, W] nebo [B, D, H, W]
            lesion_mask (torch.Tensor): Tensor binární masky léze [B, 1, D, H, W] nebo [B, D, H, W]
            adc_map (torch.Tensor): Tensor ADC mapy (reálné nebo generované) [B, 1, D, H, W] nebo [B, D, H, W]
        
        Returns:
            torch.Tensor: Mapa skóre pravděpodobnosti reálné/generované léze [B, 1, D', H', W']
        """
        # Kontrola a úprava tvarů vstupů - zajistíme 5D tensory [B, C, D, H, W]
        if pseudo_healthy.ndim == 4:
            pseudo_healthy = pseudo_healthy.unsqueeze(1)  # Přidáme kanálovou dimenzi
            
        if lesion_mask.ndim == 4:
            lesion_mask = lesion_mask.unsqueeze(1)  # Přidáme kanálovou dimenzi
            
        if adc_map.ndim == 4:
            adc_map = adc_map.unsqueeze(1)  # Přidáme kanálovou dimenzi
        
        # Kontrola minimální velikosti vstupu
        min_size = self.min_input_size
        d, h, w = pseudo_healthy.shape[2:5]
        
        if d < min_size or h < min_size or w < min_size:
            print(f"Warning: Input size {(d, h, w)} is smaller than minimum size {min_size}. Results may be unpredictable.")
        
        # Kontrola a úprava dimenzí vstupů
        # Zajistíme, že každý vstup má přesně 1 kanál
        if pseudo_healthy.shape[1] != 1:
            print(f"Warning: Unexpected pseudo_healthy channels: {pseudo_healthy.shape[1]}, using only first channel")
            pseudo_healthy = pseudo_healthy[:, 0:1]
            
        if lesion_mask.shape[1] != 1:
            print(f"Warning: Unexpected lesion_mask channels: {lesion_mask.shape[1]}, using only first channel")
            lesion_mask = lesion_mask[:, 0:1]
            
        if adc_map.shape[1] != 1:
            print(f"Warning: Unexpected adc_map channels: {adc_map.shape[1]}, using only first channel")
            adc_map = adc_map[:, 0:1]
        
        # Spojit všechny vstupy pro diskriminátor - nyní by měly mít správné dimenze
        x = torch.cat([pseudo_healthy, lesion_mask, adc_map], dim=1)
        
        # Debug informace o tvaru výsledného tenzoru
        # print(f"Discriminator input shape after concatenation: {x.shape}")
        
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
        lambda_edge=3.0  # Increased from 1.0 to 3.0 for stronger edge preservation
    ):
        super().__init__()
        
        # Generátor a diskriminátor
        self.generator = Generator(
            in_channels=2,  # pseudo-healthy + mask
            out_channels=1,  # ADC s lézí
            base_filters=gen_features,
            depth=4,  # Změněno z 5 na 4 pro konzistenci s výchozí hodnotou v Generator
            min_size=32
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
    
    def safe_ssim(self, pred, target):
        """
        Bezpečný výpočet SSIM, který ošetřuje různé tvary tensorů
        
        Args:
            pred (torch.Tensor): Predikovaný tensor
            target (torch.Tensor): Cílový tensor
            
        Returns:
            torch.Tensor: SSIM hodnota
        """
        # Zajistíme, že oba tensory mají správný tvar s jedním kanálem
        if pred.shape[1] != 1:
            pred = pred[:, 0:1]
        if target.shape[1] != 1:
            target = target[:, 0:1]
            
        try:
            return self.ssim_loss(pred, target)
        except ValueError as e:
            print(f"Chyba při výpočtu SSIM: {e}")
            print(f"Tvar pred: {pred.shape}, tvar target: {target.shape}")
            # Fallback na L1 loss
            return torch.tensor(0.0, device=pred.device)
    
    def compute_edge_loss(self, pred, target, mask, kernel_size=3):
        """
        Výpočet edge loss pro zajištění plynulého přechodu na hranici masky
        Vylepšená verze, která využívá detekci hranic a gradientů pro plynulejší přechody
        
        Args:
            pred (torch.Tensor): Predikovaná ADC mapa [B, 1, D, H, W]
            target (torch.Tensor): Cílová ADC mapa [B, 1, D, H, W]
            mask (torch.Tensor): Binární maska léze [B, 1, D, H, W]
            kernel_size (int): Velikost jádra pro dilataci a erozi hranic
        
        Returns:
            torch.Tensor: Skalární hodnota edge loss
        """
        # Kontrola a úprava tvaru tensorů
        if pred.shape[1] != 1:
            pred = pred[:, 0:1]
            
        if target.shape[1] != 1:
            target = target[:, 0:1]
            
        if mask.shape[1] != 1:
            mask = mask[:, 0:1]
                
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
        
        # 1. Detekce hranic masky pomocí dilatace a eroze
        # Vytvoříme jádro pro 3D dilataci/erozi
        padding = kernel_size // 2
        
        # Dilatace masky
        dilated_mask = F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Eroze masky (1 - max_pool3d(1 - mask))
        eroded_mask = 1 - F.max_pool3d(1 - mask, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Hranice masky je rozdíl mezi dilatovanou a erodovanou maskou
        edge_mask = dilated_mask - eroded_mask
        
        # 2. Výpočet gradientů obrazu pro zpřesnění hran
        # Sobel operátor pro detekci hran ve 3D - zjednodušená verze pro každou dimenzi
        # Definujeme konvoluční jádra pro detekci hran v různých směrech
        sobel_z = torch.tensor([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                               [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]], dtype=torch.float32, device=pred.device)
        
        sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                               [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                               [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32, device=pred.device)
        
        sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                               [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                               [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32, device=pred.device)
        
        # Rozšíříme na potřebné dimenze a přidáme kanálovou dimenzi
        sobel_z = sobel_z.view(1, 1, 3, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3, 3)
        sobel_x = sobel_x.view(1, 1, 3, 3, 3)
        
        # Aplikujeme Sobel operátory na predikované a cílové obrazy
        # Využijeme funkcionální konvoluci
        pred_grad_z = F.conv3d(pred, sobel_z, padding=1, groups=1)
        pred_grad_y = F.conv3d(pred, sobel_y, padding=1, groups=1)
        pred_grad_x = F.conv3d(pred, sobel_x, padding=1, groups=1)
        
        target_grad_z = F.conv3d(target, sobel_z, padding=1, groups=1)
        target_grad_y = F.conv3d(target, sobel_y, padding=1, groups=1)
        target_grad_x = F.conv3d(target, sobel_x, padding=1, groups=1)
        
        # Magnitudy gradientů
        pred_grad_magnitude = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + pred_grad_z**2)
        target_grad_magnitude = torch.sqrt(target_grad_x**2 + target_grad_y**2 + target_grad_z**2)
        
        # 3. Kombinovaný Edge Loss - soustředí se na hrany masky
        # L1 Loss na hranách (edge_mask) a váhovaný L1 Loss v oblasti masky
        edge_area_loss = F.l1_loss(pred * edge_mask, target * edge_mask, reduction='sum')
        mask_area_loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        
        # Gradient loss - porovnání magnitudy gradientů na hranách
        grad_edge_loss = F.l1_loss(pred_grad_magnitude * edge_mask, 
                                   target_grad_magnitude * edge_mask, 
                                   reduction='sum')
        
        # Normalizační faktory - prevence dělení nulou
        edge_sum = torch.sum(edge_mask) + 1e-8
        mask_sum = torch.sum(mask) + 1e-8
        
        # Kombinovaný loss s větší vahou na hrany a gradienty
        combined_loss = (
            (2.0 * edge_area_loss / edge_sum) +  # Větší váha na hrany
            (mask_area_loss / mask_sum) +
            (1.5 * grad_edge_loss / edge_sum)    # Přidání gradientní složky
        ) / 4.5  # Normalizace
        
        return combined_loss
    
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
        # Kontrola a úprava tvaru tensorů
        if pred.shape[1] != 1:
            print(f"Warning: Predikovaná ADC mapa má tvar {pred.shape}, očekáván je [B, 1, D, H, W]")
            pred = pred[:, 0:1]
            
        if target.shape[1] != 1:
            print(f"Warning: Cílová ADC mapa má tvar {target.shape}, očekáván je [B, 1, D, H, W]")
            target = target[:, 0:1]
            
        if mask.shape[1] != 1:
            print(f"Warning: Maska léze má tvar {mask.shape}, očekáván je [B, 1, D, H, W]")
            mask = mask[:, 0:1]
            
        # Výpočet SSIM loss pouze v oblasti léze
        pred_masked = pred * mask
        target_masked = target * mask
        ssim_loss_val = 1.0 - self.safe_ssim(pred_masked, target_masked)
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
        # Kontrola a zajištění správných tvarů tensorů
        if pseudo_healthy.shape[1] != 1:
            pseudo_healthy = pseudo_healthy[:, 0:1]
        if lesion_mask.shape[1] != 1:
            lesion_mask = lesion_mask[:, 0:1]
        if real_adc.shape[1] != 1:
            real_adc = real_adc[:, 0:1]
        if fake_adc.shape[1] != 1:
            fake_adc = fake_adc[:, 0:1]
            
        # Adversarial loss pro generátor - motivovat generátor k vytváření věrohodných lézí
        target_real = torch.ones_like(disc_fake_pred).to(disc_fake_pred.device)
        adv_loss = self.adversarial_loss(disc_fake_pred, target_real)
        
        # Rekonstrukční L1 loss - pouze v oblasti léze
        l1_loss_masked = self.l1_loss(fake_adc * lesion_mask, real_adc * lesion_mask)
        
        # Vylepšená identity loss - oblast mimo masku by měla zůstat naprosto nezměněná
        # Vytvoříme váhovanou masku pro oblasti mimo lézi
        outside_mask = 1 - lesion_mask
        
        # Dilatace masky pro zjištění přechodových oblastí
        kernel_size = 5
        padding = kernel_size // 2
        dilated_mask = F.max_pool3d(lesion_mask, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Přechodová oblast - oblasti v blízkosti léze, ale ne přímo v lézi
        transition_area = dilated_mask - lesion_mask
        
        # Oblasti daleko od léze - mimo dilatovanou masku
        far_outside = outside_mask - transition_area
        
        # Silnější váha pro oblasti daleko od léze (2x) a střední váha pro přechodové oblasti (1.5x)
        weighted_outside_mask = far_outside * 2.0 + transition_area * 1.5
        
        # Identity loss s váhovanou maskou
        identity_loss = self.l1_loss(fake_adc * weighted_outside_mask, pseudo_healthy * weighted_outside_mask)
        
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
        # Kontrola a zajištění správných tvarů tensorů
        if pseudo_healthy.shape[1] != 1:
            pseudo_healthy = pseudo_healthy[:, 0:1]
        if lesion_mask.shape[1] != 1:
            lesion_mask = lesion_mask[:, 0:1]
        if real_adc.shape[1] != 1:
            real_adc = real_adc[:, 0:1]
        if fake_adc.shape[1] != 1:
            fake_adc = fake_adc[:, 0:1]
            
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
            pseudo_healthy (torch.Tensor): Pseudo-zdravý mozek [B, 1, D, H, W] nebo [B, D, H, W]
            lesion_mask (torch.Tensor): Binární maska léze [B, 1, D, H, W] nebo [B, D, H, W]
        
        Returns:
            torch.Tensor: Generovaná ADC mapa s lézí [B, 1, D, H, W]
        """
        # Kontrola a úprava tvarů vstupů je už implementována v metodě Generator.forward
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
                fake_adc = self.model.generator(pseudo_healthy, lesion_mask)
                
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
        
        # Flag pro debug výpisy - nastavit na True pouze při diagnostice
        debug_first_batch = True
        
        for batch_idx, batch in enumerate(pbar):
            # Přesunout data na správné zařízení
            pseudo_healthy = batch['pseudo_healthy'].to(self.device)
            lesion_mask = batch['lesion_mask'].to(self.device)
            real_adc = batch['adc'].to(self.device)
            
            # Debug výpisy pro diagnostiku
            if batch_idx == 0 and debug_first_batch:
                print(f"\nDEBUG - Input shapes:")
                print(f"pseudo_healthy: {pseudo_healthy.shape}")
                print(f"lesion_mask: {lesion_mask.shape}")
                print(f"real_adc: {real_adc.shape}")
            
            # ===== Trénink diskriminátoru =====
            self.optimizer_d.zero_grad()
            
            # Generovat falešnou ADC mapu
            with torch.amp.autocast('cuda') if self.use_amp else nullcontext():
                fake_adc = self.model.generator(pseudo_healthy, lesion_mask)
                
                # Debug výpisy pro diagnostiku
                if batch_idx == 0 and debug_first_batch:
                    print(f"fake_adc shape after generator: {fake_adc.shape}")
                
                # Kontrola a zajištění správných tvarů tensorů před předáním diskriminátoru
                if pseudo_healthy.shape[1] != 1:
                    pseudo_healthy_d = pseudo_healthy[:, 0:1]
                else:
                    pseudo_healthy_d = pseudo_healthy
                    
                if lesion_mask.shape[1] != 1:
                    lesion_mask_d = lesion_mask[:, 0:1]
                else:
                    lesion_mask_d = lesion_mask
                    
                if real_adc.shape[1] != 1:
                    real_adc_d = real_adc[:, 0:1]
                else:
                    real_adc_d = real_adc
                    
                if fake_adc.shape[1] != 1:
                    fake_adc_d = fake_adc[:, 0:1]
                else:
                    fake_adc_d = fake_adc
                
                # Debug výpisy pro diagnostiku
                if batch_idx == 0 and debug_first_batch:
                    print(f"\nDEBUG - Processed tensor shapes for discriminator:")
                    print(f"pseudo_healthy_d: {pseudo_healthy_d.shape}")
                    print(f"lesion_mask_d: {lesion_mask_d.shape}")
                    print(f"real_adc_d: {real_adc_d.shape}")
                    print(f"fake_adc_d: {fake_adc_d.shape}")
                    debug_first_batch = False  # Pouze pro první batch
                
                # Predikce diskriminátoru pro reálná a falešná data
                disc_real_pred = self.model.discriminator(pseudo_healthy_d, lesion_mask_d, real_adc_d)
                disc_fake_pred = self.model.discriminator(pseudo_healthy_d, lesion_mask_d, fake_adc_d.detach())
                
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
                
                # Predikce diskriminátoru pro falešná data s kontrolou tvarů
                if fake_adc.shape[1] != 1:
                    fake_adc_d = fake_adc[:, 0:1]
                else:
                    fake_adc_d = fake_adc
                
                disc_fake_pred = self.model.discriminator(pseudo_healthy_d, lesion_mask_d, fake_adc_d)
                
                # Loss generátoru
                # Zajistíme správný tvar fake_adc před výpočtem loss funkce
                fake_adc_loss = fake_adc_d  # Použijeme již upravený tensor se správným tvarem [B, 1, D, H, W]
                
                g_loss, g_adv_loss, g_l1_loss, g_identity_loss, g_ssim_loss, g_edge_loss = \
                    self.model.generator_loss(
                        pseudo_healthy, lesion_mask, real_adc, fake_adc_loss, disc_fake_pred
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
        Validace modelu
        
        Args:
            val_loader: DataLoader s validačními daty
            epoch (int): Aktuální epocha
        
        Returns:
            dict: Validační metriky
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
        
        # Ukládat všechny batche, abychom mohli náhodně vybrat jeden pro vizualizaci
        all_batches = []
        mghnicu_405_batch = None
        
        with torch.no_grad():
            for batch in val_loader:
                # Přesunout data na zařízení
                pseudo_healthy = batch['pseudo_healthy'].to(self.device)
                lesion_mask = batch['lesion_mask'].to(self.device)
                real_adc = batch['adc'].to(self.device)
                patient_id = batch['patient_id'][0]  # Bereme první pacienta z batche
                
                # Kontrola, zda jde o pacienta MGHNICU_405
                if 'MGHNICU_405' in patient_id:
                    mghnicu_405_batch = batch
                    print(f"Nalezen pacient MGHNICU_405 ve validační sadě, bude prioritně použit pro vizualizaci")
                
                # Uložit batch pro pozdější vizualizaci
                all_batches.append(batch)
                
                # Forward pass
                with torch.amp.autocast('cuda') if self.use_amp else nullcontext():
                    # Generování falešných ADC map
                    fake_adc = self.model.generator(pseudo_healthy, lesion_mask)
                    
                    # Kontrola a zajištění správných tvarů tensorů před předáním diskriminátoru
                    if pseudo_healthy.shape[1] != 1:
                        pseudo_healthy_d = pseudo_healthy[:, 0:1]
                    else:
                        pseudo_healthy_d = pseudo_healthy
                        
                    if lesion_mask.shape[1] != 1:
                        lesion_mask_d = lesion_mask[:, 0:1]
                    else:
                        lesion_mask_d = lesion_mask
                        
                    if real_adc.shape[1] != 1:
                        real_adc_d = real_adc[:, 0:1]
                    else:
                        real_adc_d = real_adc
                        
                    if fake_adc.shape[1] != 1:
                        fake_adc_d = fake_adc[:, 0:1]
                    else:
                        fake_adc_d = fake_adc
                    
                    # Generator loss
                    disc_fake_pred = self.model.discriminator(pseudo_healthy_d, lesion_mask_d, fake_adc_d)
                    
                    # Výpočet loss pro generátor se správnými argumenty
                    g_loss, g_adv_loss, g_l1_loss, g_identity_loss, g_ssim_loss, g_edge_loss = \
                        self.model.generator_loss(
                        pseudo_healthy, lesion_mask, real_adc, fake_adc_d, disc_fake_pred
                    )
                    
                    # Výpočet loss pro diskriminátor
                    # Získat predikci pro reálná data
                    disc_real_pred = self.model.discriminator(pseudo_healthy_d, lesion_mask_d, real_adc_d)
                    d_loss = self.model.discriminator_loss(
                        pseudo_healthy, lesion_mask, real_adc, fake_adc_d, 
                        disc_real_pred, disc_fake_pred
                    )
                
                # Akumulovat metriky
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
            
        # Vizualizace na náhodně vybraném obrazu, prioritně MGHNICU_405 pokud je dostupný
        if all_batches:
            # Prioritně použít MGHNICU_405, pokud byl nalezen
            selected_batch = mghnicu_405_batch if mghnicu_405_batch is not None else random.choice(all_batches)
            
            # Vytvořit PDF vizualizaci pro vybraný batch
            pdf_path = self.create_full_volume_pdf_visualization(epoch, selected_batch)
            patient_id = selected_batch['patient_id'][0]
            print(f"Full volume PDF visualization created for patient {patient_id} at: {pdf_path}")
            
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
        
        # Kontrola a úprava tvaru tensorů
        if pseudo_healthy.shape[1] != 1:
            pseudo_healthy = pseudo_healthy[:, 0:1]
        if lesion_mask.shape[1] != 1:
            lesion_mask = lesion_mask[:, 0:1]
        if real_adc.shape[1] != 1:
            real_adc = real_adc[:, 0:1]
        if fake_adc.shape[1] != 1:
            fake_adc = fake_adc[:, 0:1]
            
        # Vytvořit kombinovaný výsledek (pseudo-zdravý + vygenerovaná léze)
        # Kombinujeme pseudo-zdravý obraz a generovanou ADC mapu v oblasti léze
        inpainted_adc = pseudo_healthy * (1 - lesion_mask) + fake_adc * lesion_mask
        
        # Vypočítat metriky mezi reálnou ADC mapou a inpainted výsledkem (celý objem)
        # Kontrola tvarů před výpočtem SSIM
        if inpainted_adc.shape[1] != 1:
            print(f"Upravuji tvar inpainted_adc z {inpainted_adc.shape} na [B, 1, D, H, W]")
            inpainted_adc_ssim = inpainted_adc[:, 0:1]  # Vezmi pouze první kanál
        else:
            inpainted_adc_ssim = inpainted_adc
            
        if real_adc.shape[1] != 1:
            print(f"Upravuji tvar real_adc z {real_adc.shape} na [B, 1, D, H, W]")
            real_adc_ssim = real_adc[:, 0:1]
        else:
            real_adc_ssim = real_adc
            
        # Počítáme SSIM pro celý objem s upravenými tvary
        try:
            print(f"SSIM výpočet - tvar inpainted_adc_ssim: {inpainted_adc_ssim.shape}, tvar real_adc_ssim: {real_adc_ssim.shape}")
            ssim_val = self.model.safe_ssim(inpainted_adc_ssim, real_adc_ssim).item()
        except ValueError as e:
            print(f"Chyba při výpočtu SSIM: {e}")
            print(f"Tvar inpainted_adc: {inpainted_adc.shape}, tvar real_adc: {real_adc.shape}")
            # Fallback na L1 loss pokud SSIM selže
            ssim_val = -1.0  # Neplatná hodnota jako indikátor problému
        
        # Počítáme MAE (mean absolute error) pro celý objem
        mae_val = F.l1_loss(inpainted_adc, real_adc).item()
        
        # Cesta k výstupnímu PDF souboru
        pdf_filename = f"epoch_{epoch}_patient_{patient_id}_full_volume.pdf"
        pdf_path = self.pdf_visualization_dir / pdf_filename
        
        # Konvertovat data na CPU a numpy pro vizualizaci
        # Důležité: převedeme z formátu [B, C, D, H, W] na [D, H, W] pro správné dělení na 2D řezy
        pseudo_healthy_np = pseudo_healthy.squeeze().cpu().numpy()  # [D, H, W]
        real_adc_np = real_adc.squeeze().cpu().numpy()  # [D, H, W]
        fake_adc_np = fake_adc.squeeze().cpu().numpy()  # [D, H, W]
        lesion_mask_np = lesion_mask.squeeze().cpu().numpy()  # [D, H, W]
        inpainted_adc_np = inpainted_adc.squeeze().cpu().numpy()  # [D, H, W]
        
        # Zkontrolujeme tvary dat pro debugování
        print(f"Tvar dat pro vizualizaci - pseudo_healthy: {pseudo_healthy_np.shape}")
        print(f"Tvar dat pro vizualizaci - real_adc: {real_adc_np.shape}")
        print(f"Tvar dat pro vizualizaci - fake_adc: {fake_adc_np.shape}")
        print(f"Tvar dat pro vizualizaci - lesion_mask: {lesion_mask_np.shape}")
        print(f"Tvar dat pro vizualizaci - inpainted_adc: {inpainted_adc_np.shape}")
        
        # Pokud jsou data 4D místo 3D (např. [C, D, H, W]), vezmeme první kanál
        if len(pseudo_healthy_np.shape) == 4:
            pseudo_healthy_np = pseudo_healthy_np[0]  # První kanál
        if len(real_adc_np.shape) == 4:
            real_adc_np = real_adc_np[0]
        if len(fake_adc_np.shape) == 4:
            fake_adc_np = fake_adc_np[0]
        if len(lesion_mask_np.shape) == 4:
            lesion_mask_np = lesion_mask_np[0]
        if len(inpainted_adc_np.shape) == 4:
            inpainted_adc_np = inpainted_adc_np[0]
        
        # Počet řezů ve směru D (hloubka)
        num_slices = pseudo_healthy_np.shape[0]
        max_slices_per_page = 5  # Maximální počet řezů na stránku
        
        with PdfPages(pdf_path) as pdf:
            # Pro každou stránku
            for page_start in range(0, num_slices, max_slices_per_page):
                page_end = min(page_start + max_slices_per_page, num_slices)
                slices_on_page = page_end - page_start
                
                # Vytvořit obrázek s řádky (řezy) a sloupci (typy obrazů)
                fig = plt.figure(figsize=(15, 3 * slices_on_page))  # Zvětšeno pro 5 sloupců
                
                # Přidat titulek s metrikami
                fig.suptitle(f'Patient {patient_id}, Epoch {epoch}, SSIM: {ssim_val:.4f}, MAE: {mae_val:.4f}', 
                            fontsize=14)
                
                # Vytvořit grid pro umístění subplotů - přidán jeden sloupec pro inpainted výsledek
                gs = gridspec.GridSpec(slices_on_page, 5, figure=fig)
                
                # Projít všechny řezy na aktuální stránce
                for i in range(slices_on_page):
                    slice_idx = page_start + i
                    
                    # Pro každý typ obrazu vytvořit subplot
                    ax1 = fig.add_subplot(gs[i, 0])
                    # Vybrat konkrétní 2D řez
                    ax1.imshow(pseudo_healthy_np[slice_idx], cmap='gray')
                    ax1.set_title('Pseudo-Healthy' if i == 0 else '')
                    ax1.axis('off')
                    
                    ax2 = fig.add_subplot(gs[i, 1])
                    ax2.imshow(real_adc_np[slice_idx], cmap='gray')
                    ax2.set_title('Real ADC' if i == 0 else '')
                    ax2.axis('off')
                    
                    ax3 = fig.add_subplot(gs[i, 2])
                    ax3.imshow(fake_adc_np[slice_idx], cmap='gray')
                    ax3.set_title('Generated ADC' if i == 0 else '')
                    ax3.axis('off')

                    # Nový sloupec pro kombinovaný výsledek (inpainted)
                    ax5 = fig.add_subplot(gs[i, 3])
                    ax5.imshow(inpainted_adc_np[slice_idx], cmap='gray')
                    ax5.set_title('Inpainted Result' if i == 0 else '')
                    ax5.axis('off')
                    
                    ax4 = fig.add_subplot(gs[i, 4])
                    ax4.imshow(lesion_mask_np[slice_idx], cmap='gray')
                    ax4.set_title('Lesion Mask' if i == 0 else '')
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
    
    # Nastavení pro patch-based trénink
    use_patches = args.use_patches if hasattr(args, 'use_patches') else True
    
    if use_patches:
        # Nastavení velikosti patchů
        patch_size = (64, 64, 32)  # Výchozí velikost patche
        if hasattr(args, 'patch_size') and args.patch_size:
            patch_size = tuple(map(int, args.patch_size.split(',')))
        print(f"Použito trénování na patchích o velikosti: {patch_size}")
        # Pro patch-based trénink použijeme patch jako cílovou velikost pro model
        model_input_size = patch_size
    else:
        print(f"Použito trénování na celých objemech, dataset bude dynamicky zpracovávat každý vzorek individuálně.")
        # Pro trénink na celých objemech použijeme původní cílovou velikost
        model_input_size = target_size
    
    if args.target_size:
        # Použít zadanou cílovou velikost, pokud je specifikována
        model_input_size = tuple(map(int, args.target_size.split(',')))
        print(f"Použita ručně specifikovaná cílová velikost pro model: {model_input_size}")
    
    # Definovat transformace pro augmentaci dat (pouze pro tréninkovou část)
    train_transforms = None
    if args.use_augmentations:
        # Pro testování patch-based tréninku je možné vypnout augmentace, pokud způsobují problémy
        # Nastavte následující příznak na True pro vypnutí augmentací
        disable_augmentation_for_testing = False
        
        if disable_augmentation_for_testing:
            print("Augmentace dočasně vypnuty pro testování")
        else:
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
                # DŮLEŽITÉ: Odstraňujeme RandRotate90d, která způsobuje problémy s různými tvary
                # RandRotate90d(
                #     keys=keys,
                #     prob=args.aug_rotate_prob,
                #     max_k=3,  # maximálně 3 rotace (0, 90, 180, 270 stupňů)
                #     spatial_axes=(0, 1)  # Opraveno - používáme (0, 1) pro první dvě dimenze
                # ),
                
                # Náhodné překlápění (zrcadlení)
                RandFlipd(
                    keys=keys,
                    prob=args.aug_flip_prob,
                    spatial_axis=None  # náhodný výběr os - toto je v pořádku, protože None znamená náhodný výběr z dostupných os
                ),
                
                # Affinní transformace (rotace, škálování, posuny)
                RandAffined(
                    keys=keys,
                    prob=args.aug_affine_prob,
                    rotate_range=(np.pi/36, np.pi/36, np.pi/36),  # max +/- 5 stupňů ve všech osách - správně pro 3D data
                    scale_range=(0.05, 0.05, 0.05),  # škálování o +/- 5% - správně pro 3D data
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
    patch_size = None
    if args.use_patches:
        # Převeďte string na tuple celých čísel
        patch_size = tuple(map(int, args.patch_size.split(',')))
        print(f"Používám patch-based training s velikostí patche: {patch_size}")
    
    train_dataset = LesionInpaintingDataset(
        pseudo_healthy_dir=args.pseudo_healthy_dir,
        adc_dir=args.adc_dir,
        lesion_mask_dir=args.lesion_mask_dir,
        norm_range=(0, 1),
        crop_foreground=args.crop_foreground,
        transform=train_transforms,
        patch_size=patch_size,
        use_patches=args.use_patches
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
    
    # Najděme index pacienta MGHNICU_405, aby byl vždy ve validační sadě
    mghnicu_405_index = None
    for i, triplet in enumerate(train_dataset.file_triplets):
        if 'MGHNICU_405' in triplet['patient_id']:
            mghnicu_405_index = i
            print(f"Nalezen pacient MGHNICU_405 na indexu {i}, bude vždy použit pro validaci")
            break
    
    # Pro validační dataset nepoužíváme augmentace, musíme proto vytvořit nový dataset pro validaci
    if args.use_augmentations and val_size > 0:
        # Vytvoříme indexy pro rozdělení datasetu
        indices = list(range(dataset_size))
        
        # Zamíchat indexy a rozdělit je na trénovací a validační
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        
        # Pokud jsme našli MGHNICU_405, zajistit, že je ve validační sadě
        if mghnicu_405_index is not None:
            # Odstranit MGHNICU_405 z indexů, pokud tam je
            if mghnicu_405_index in indices:
                indices.remove(mghnicu_405_index)
            
            # Rozdělit zbytek indexů
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size - 1]  # Rezervovat místo pro MGHNICU_405
            
            # Přidat MGHNICU_405 do validační sady
            val_indices.append(mghnicu_405_index)
        else:
            # Standardní rozdělení, pokud MGHNICU_405 není nalezen
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
        
        # Vytvoříme zvlášť validační dataset bez augmentací
        val_dataset = LesionInpaintingDataset(
            pseudo_healthy_dir=args.pseudo_healthy_dir,
            adc_dir=args.adc_dir,
            lesion_mask_dir=args.lesion_mask_dir,
            norm_range=(0, 1),
            crop_foreground=args.crop_foreground,
            transform=None,  # bez augmentací pro validaci
            patch_size=patch_size,
            use_patches=args.use_patches
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
        # s úpravou pro vždy zařazení MGHNICU_405 do validační sady
        if mghnicu_405_index is not None:
            # Vytvoříme vlastní rozdělení s MGHNICU_405 vždy ve validační sadě
            from torch.utils.data import Subset
            
            # Vytvoříme indexy pro rozdělení datasetu
            indices = list(range(dataset_size))
            
            # Odstranit MGHNICU_405 z indexů, pokud tam je
            if mghnicu_405_index in indices:
                indices.remove(mghnicu_405_index)
                
            # Zamíchat zbývající indexy
            np.random.seed(args.seed)
            np.random.shuffle(indices)
            
            # Rozdělit zbytek indexů
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size - 1]
            
            # Přidat MGHNICU_405 do validační sady
            val_indices.append(mghnicu_405_index)
            
            # Vytvořit Subset datasety
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(train_dataset, val_indices)
        else:
            # Standardní rozdělení pomocí random_split
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
    
    # Definovat vstupní velikost pro model (závisí na patch-based nebo full-volume tréninku)
    if args.use_patches:
        # Pro patch-based trénink použijeme velikost patche
        model_input_size = patch_size
    else:
        # Pro full-volume trénink zkusíme najít velikost dat
        # Podíváme se na první vzorek v datasetu
        sample = train_dataset[0]
        if isinstance(train_dataset, Subset):
            # Pokud používáme Subset, musíme získat původní dataset
            sample = train_dataset.dataset[train_dataset.indices[0]]
        
        # Zjistit velikost z prvního vzorku
        if 'volume_shape' in sample:
            model_input_size = sample['volume_shape']
        else:
            # Výchozí velikost, pokud nelze zjistit z datasetu
            model_input_size = (64, 64, 64)
    
    print(f"Vstupní velikost pro model: {model_input_size}")
    
    # Inicializovat model s novými rozměry
    model = LesionInpaintingGAN(
        img_size=model_input_size,  # Použít správnou velikost podle typu tréninku
        gen_features=args.gen_features,
        disc_base_filters=args.disc_base_filters,
        disc_n_layers=args.disc_n_layers,
        lambda_l1=args.lambda_l1,
        lambda_identity=args.lambda_identity,
        lambda_ssim=args.lambda_ssim,
        lambda_edge=args.lambda_edge
    ).to(device)
    
    # Inicializovat optimizer pro generátor a diskriminátor
    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    
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
    
    # Vytvořit výstupní adresář
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializovat trainer
    trainer = LesionInpaintingTrainer(
        model=model,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=device,
        output_dir=output_dir,
        config=args,
        use_amp=args.use_amp,
        visualize_interval=args.visualize_interval
    )
    
    # Načíst checkpoint, pokud je zadán
    start_epoch = 0
    if args.checkpoint:
        print(f"Načítám checkpoint: {args.checkpoint}")
        start_epoch = trainer.load_model(args.checkpoint)
        print(f"Načten checkpoint z epochy {start_epoch}")
    
    # Spustit trénink
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        start_epoch=start_epoch,
        checkpoint_interval=args.checkpoint_interval
    )


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
    
    # Nové parametry pro patch-based trénink
    parser.add_argument('--use_patches', action='store_true', default=True,
                        help='Použít patch-based trénink místo celých objemů')
    parser.add_argument('--patch_size', type=str, default='64,64,32',
                        help='Velikost patchů pro trénink, např. "64,64,32"')
    
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
    parser.add_argument('--lambda_edge', type=float, default=3.0,
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
