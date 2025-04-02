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
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime
import torchvision.transforms as transforms

# Nastavení seedu pro reprodukovatelnost
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Vlastní 3D transformace pro data augmentaci
class RandomRotation3D:
    def __init__(self, angles=(3, 3, 3)):
        self.angles = angles
        
    def __call__(self, tensor):
        x_angle = random.uniform(-self.angles[0], self.angles[0])
        y_angle = random.uniform(-self.angles[1], self.angles[1])
        z_angle = random.uniform(-self.angles[2], self.angles[2])
        
        # Příprava matic rotace ve 3D
        # Implementace rotací pomocí interpolace
        
        # Použijeme affine_grid a grid_sample pro 3D rotaci
        # Toto je zjednodušená implementace, která provádí pouze lehké rotace
        
        # Pro jednoduchost vrátíme tensor bez změny
        # Plná implementace rotací ve 3D vyžaduje složitější transformace
        return tensor

class RandomFlip3D:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, tensor):
        if random.random() < self.p:
            # Získání maximální dimenze tenzoru
            max_dim = len(tensor.shape) - 1  # Pro 4D tenzor [C,D,H,W] je max_dim=3
            
            # Vybíráme náhodnou prostorovou dimenzi (ignorujeme batch a channel)
            # Začínáme od indexu 2 (první prostorová dimenze)
            spatial_dims = list(range(2, max_dim + 1))
            
            # Pokud existují prostorové dimenze k flipování
            if spatial_dims:
                axis = random.choice(spatial_dims)
                return torch.flip(tensor, [axis])
        return tensor

# Dataset pro LabelGAN
class LabelGANDataset(Dataset):
    """Dataset pro LabelGAN, který generuje LABEL mapy (léze)"""
    
    def __init__(self, label_dir, lesion_atlas_path, transform=None, augment=True):
        """
        Args:
            label_dir: Adresář s registrovanými LABEL mapami
            lesion_atlas_path: Cesta k atlasu frekvence lézí
            transform: Volitelné transformace (např. normalizace)
            augment: Použití datových augmentací (True/False)
        """
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                                  if f.endswith('.mha') or f.endswith('.nii.gz') or f.endswith('.nii')])
        
        self.lesion_atlas_path = lesion_atlas_path
        self.transform = transform
        self.augment = augment
        
        # Načtení atlasu frekvence lézí
        if self.lesion_atlas_path.endswith('.nii.gz') or self.lesion_atlas_path.endswith('.nii'):
            self.lesion_atlas = nib.load(self.lesion_atlas_path).get_fdata()
        else:
            self.lesion_atlas = sitk.GetArrayFromImage(sitk.ReadImage(self.lesion_atlas_path))
        
        # Normalizace do rozsahu [0, 1]
        self.lesion_atlas = (self.lesion_atlas - self.lesion_atlas.min()) / (self.lesion_atlas.max() - self.lesion_atlas.min())
        
        # Definování augmentací
        if self.augment:
            self.augmentation = [
                RandomRotation3D(angles=(3, 3, 3)),  # Malé rotace ve všech osách
                RandomFlip3D(p=0.5),  # Náhodný flip s 50% pravděpodobností
            ]
        else:
            self.augmentation = None
        
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
        label = torch.FloatTensor(label).unsqueeze(0)  # Přidání kanálové dimenze
        lesion_atlas = torch.FloatTensor(self.lesion_atlas).unsqueeze(0)  # Přidání kanálové dimenze
        
        # Aplikace augmentací, pokud jsou povoleny
        if self.augment and self.augmentation:
            # Vytvoříme seed, aby se stejné augmentace aplikovaly na oba tensory
            seed = random.randint(0, 2**32 - 1)
            
            # Aplikace stejných augmentací na oba tensory
            torch.manual_seed(seed)
            random.seed(seed)
            for aug in self.augmentation:
                label = aug(label)
                
            torch.manual_seed(seed)
            random.seed(seed)
            for aug in self.augmentation:
                lesion_atlas = aug(lesion_atlas)
        
        # Aplikace dalších transformací, pokud jsou definovány
        if self.transform:
            label = self.transform(label)
            lesion_atlas = self.transform(lesion_atlas)
        
        # Vytvoření náhodného šumu pro generátor
        noise = torch.randn_like(lesion_atlas)
        
        return {
            'label': label,  # Skutečná LABEL mapa
            'lesion_atlas': lesion_atlas,  # Atlas frekvence lézí
            'noise': noise  # Náhodný šum pro variabilitu při generování
        }

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

# Generator architektury
class LesionGenerator(nn.Module):
    """Generator pro vytváření realistických lézí"""
    
    def __init__(self, in_channels=2, features=64, depth=4, dropout_rate=0.3, use_attention=True):
        """
        Args:
            in_channels: Počet vstupních kanálů (atlas lézí + šum)
            features: Počet základních feature map
            depth: Hloubka U-Net architektury
            dropout_rate: Míra dropout v generátoru
            use_attention: Použít attention mechanismus pro lepší detaily
        """
        super(LesionGenerator, self).__init__()
        
        self.use_attention = use_attention
        self.depth = depth
        self.features = features
        
        # Encoder část - stejná jako u U-Net
        self.encoder = nn.ModuleList()
        
        # První konvoluční vrstva
        self.encoder.append(nn.Conv3d(in_channels, features, kernel_size=4, stride=2, padding=1))
        
        # Další vrstvy encoderu
        for i in range(1, depth):
            in_ch = features * (2**(i-1))
            out_ch = features * (2**i)
            
            self.encoder.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(out_ch)
                )
            )
        
        # Self-attention vrstva po poslední vrstvě encoderu
        if use_attention:
            self.attention = SelfAttention3D(features * (2**(depth-1)))
        
        # Decoder část
        self.decoder = nn.ModuleList()
        
        # Vrstvy decoderu s skip connections
        for i in range(depth):
            if i == 0:
                # První decode vrstva - bez skip connection
                in_ch = features * (2**(depth-1))
                out_ch = features * (2**(depth-2))
            else:
                # Další decode vrstvy se skip connections - OPRAVENO
                # Pro každou vrstvu správně vypočítáme počet výstupních kanálů
                in_ch = features * (2**(depth-i-1)) + features * (2**(depth-i-2))  # Skip connection + předchozí vrstva
                if i == depth - 1:  # Pro poslední vrstvu
                    out_ch = features
                else:
                    out_ch = features * (2**(depth-i-2))
            
            # Decoder vrstvy s batch normalizací a dropoutem
            self.decoder.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.Dropout3d(dropout_rate if i < depth-1 else 0)  # Dropout kromě poslední vrstvy
                )
            )
        
        # Výstupní vrstva pro binární LABEL mapu
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features + features, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid pro binární mapu (0-1)
        )
        
    def forward(self, lesion_atlas, noise):
        """
        Args:
            lesion_atlas: Atlas frekvence lézí (B, 1, D, H, W)
            noise: Náhodný šum (B, 1, D, H, W)
        Returns:
            Generovaná binární LABEL mapa (B, 1, D, H, W)
        """
        # Zapamatujeme si původní rozměry pro zajištění stejného výstupu
        orig_shape = lesion_atlas.shape
        
        # Spojení atlasu lézí a šumu jako vstup
        x = torch.cat([lesion_atlas, noise], dim=1)
        
        if hasattr(torch, '_C') and hasattr(torch._C, '_log_api_usage_once'):
            if hasattr(torch._C, '_log_api_usage_once'):
                torch._C._log_api_usage_once(f"Input shape: {x.shape}")
        
        # Uložíme si výstupy z encoderu pro skip connections
        encoder_outputs = []
        
        # Encoder forward pass
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if hasattr(torch, '_C') and hasattr(torch._C, '_log_api_usage_once'):
                if hasattr(torch._C, '_log_api_usage_once'):
                    torch._C._log_api_usage_once(f"Encoder {i} output shape: {x.shape}")
            if i < self.depth - 1:  # Uložíme všechny kromě poslední vrstvy
                encoder_outputs.append(x)
        
        # Aplikace self-attention na bottleneck
        if self.use_attention:
            x = self.attention(x)
        
        # Decoder forward pass s skip connections
        for i, layer in enumerate(self.decoder):
            # Pro první vrstvu dekodéru nepřidáváme skip connection
            if i > 0:
                skip_idx = self.depth - i - 1
                
                # Bezpečné spojení s kontrolou rozměrů
                try:
                    skip_x = encoder_outputs[skip_idx]
                    # Před spojením se ujistíme, že rozměry odpovídají
                    if x.shape[2:] != skip_x.shape[2:]:
                        skip_x = F.interpolate(skip_x, size=x.shape[2:], mode='trilinear', align_corners=False)
                    x = torch.cat([x, skip_x], dim=1)
                except Exception as e:
                    print(f"Chyba při skip connection {i}: {e}")
                    print(f"Dekodér {i} - x.shape: {x.shape}, skip_x.shape: {skip_x.shape}")
                    # Pokud spojení selže, pokračujeme bez skip connection
                    pass
            
            x = layer(x)
            if hasattr(torch, '_C') and hasattr(torch._C, '_log_api_usage_once'):
                if hasattr(torch._C, '_log_api_usage_once'):
                    torch._C._log_api_usage_once(f"Decoder {i} output shape: {x.shape}")
        
        # Poslední skip connection a výstupní vrstva
        try:
            skip_x = encoder_outputs[0]
            # Před spojením se ujistíme, že rozměry odpovídají
            if x.shape[2:] != skip_x.shape[2:]:
                skip_x = F.interpolate(skip_x, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip_x], dim=1)
        except Exception as e:
            print(f"Chyba při finální skip connection: {e}")
            # Pokud spojení selže, pokračujeme bez skip connection
            pass
        
        lesion_map = self.output_layer(x)
        
        # Ujistíme se, že výstup má stejné prostorové rozměry jako vstup
        if lesion_map.shape[2:] != orig_shape[2:]:
            lesion_map = F.interpolate(lesion_map, size=orig_shape[2:], mode='trilinear', align_corners=False)
        
        return lesion_map

# Discriminator architektury
class LesionDiscriminator(nn.Module):
    """Discriminator pro rozlišení reálných a syntetických LABEL map"""
    
    def __init__(self, in_channels=2, features=64, use_spectral_norm=True, depth=4):
        """
        Args:
            in_channels: Počet vstupních kanálů (atlas lézí + LABEL mapa)
            features: Počet základních feature map
            use_spectral_norm: Použít spektrální normalizaci pro stabilnější trénink
            depth: Hloubka diskriminátoru
        """
        super(LesionDiscriminator, self).__init__()
        
        # Spektrální normalizace pomáhá stabilizovat trénink GAN
        norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # První vrstva bez normalizace
        self.initial = norm_layer(nn.Conv3d(in_channels, features, kernel_size=4, stride=2, padding=1))
        
        # Střední vrstvy s normalizací
        self.conv_layers = nn.ModuleList()
        for i in range(1, depth):
            in_ch = features * (2**(i-1))
            out_ch = features * (2**i)
            
            self.conv_layers.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    norm_layer(nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)),
                    nn.BatchNorm3d(out_ch)
                )
            )
        
        # Výstupní vrstva pro patch-based diskriminátor
        self.output_layer = nn.Conv3d(features * (2**(depth-1)), 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, lesion_atlas, label):
        """
        Args:
            lesion_atlas: Atlas frekvence lézí (B, 1, D, H, W)
            label: LABEL mapa (B, 1, D, H, W)
        Returns:
            Pravděpodobnost, že LABEL mapa je reálná
        """
        # Kontrola a úprava rozměrů, pokud je to potřeba
        if lesion_atlas.shape != label.shape:
            # Vypíšeme varování, pokud se rozměry neshodují
            print(f"Varování: Rozměry lesion_atlas {lesion_atlas.shape} a label {label.shape} se neshodují")
            
            # Použijeme interpolaci, abychom změnili rozměr label na stejný jako lesion_atlas
            label = F.interpolate(label, size=lesion_atlas.shape[2:], mode='trilinear', align_corners=False)
        
        # Spojení atlasu lézí a LABEL mapy
        x = torch.cat([lesion_atlas, label], dim=1)
        
        # Průchod první vrstvou
        x = self.initial(x)
        
        # Průchod středními vrstvami
        for layer in self.conv_layers:
            x = layer(x)
        
        # Poslední vrstva
        x = self.output_layer(x)
        
        return x

# Pomocné funkce
def save_sample(data, path, slice_idx=None):
    """
    Ukládá vygenerované vzorky
    
    Args:
        data: Dict obsahující data ke zobrazení
        path: Cesta, kam ukládat snímky
        slice_idx: Seznam indexů řezů pro zobrazení (pokud None, vyberou se automaticky)
    """
    # Vytvoření adresáře, pokud neexistuje
    os.makedirs(path, exist_ok=True)
    
    # Získání dat z dictionaries
    real_label = data['label'][0, 0].cpu().numpy()  # První vzorek z batche, první kanál
    lesion_atlas = data['lesion_atlas'][0, 0].cpu().numpy()
    if 'fake_label' in data:
        fake_label = data['fake_label'][0, 0].cpu().numpy()
    
    # Pokud není definován řez, najdeme středy v každé ose
    if slice_idx is None:
        D, H, W = real_label.shape
        slice_idx = [D // 2, H // 2, W // 2]  # Střední řezy
    
    # Axiální řez (D)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(real_label[slice_idx[0], :, :], cmap='binary')
    plt.title('Real Label - Axial')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(lesion_atlas[slice_idx[0], :, :], cmap='hot')
    plt.title('Lesion Atlas - Axial')
    plt.colorbar()
    
    if 'fake_label' in data:
        plt.subplot(1, 3, 3)
        plt.imshow(fake_label[slice_idx[0], :, :], cmap='binary')
        plt.title('Generated Label - Axial')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'axial.png'), dpi=200)
    plt.close()
    
    # Koronální řez (H)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(real_label[:, slice_idx[1], :], cmap='binary')
    plt.title('Real Label - Coronal')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(lesion_atlas[:, slice_idx[1], :], cmap='hot')
    plt.title('Lesion Atlas - Coronal')
    plt.colorbar()
    
    if 'fake_label' in data:
        plt.subplot(1, 3, 3)
        plt.imshow(fake_label[:, slice_idx[1], :], cmap='binary')
        plt.title('Generated Label - Coronal')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'coronal.png'), dpi=200)
    plt.close()
    
    # Sagitální řez (W)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(real_label[:, :, slice_idx[2]], cmap='binary')
    plt.title('Real Label - Sagittal')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(lesion_atlas[:, :, slice_idx[2]], cmap='hot')
    plt.title('Lesion Atlas - Sagittal')
    plt.colorbar()
    
    if 'fake_label' in data:
        plt.subplot(1, 3, 3)
        plt.imshow(fake_label[:, :, slice_idx[2]], cmap='binary')
        plt.title('Generated Label - Sagittal')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'sagittal.png'), dpi=200)
    plt.close()
    
    # Uložení .nii.gz souborů pro pozdější použití
    if 'fake_label' in data:
        # Vytvoříme nifti z numpy array
        fake_nifti = nib.Nifti1Image(fake_label, np.eye(4))
        nib.save(fake_nifti, os.path.join(path, 'generated_label.nii.gz'))

def train_lesion_gan(args):
    """
    Hlavní trénovací funkce pro LabelGAN
    
    Args:
        args: Argumenty tréninku, včetně:
            - label_dir: Adresář s registrovanými LABEL mapami
            - lesion_atlas_path: Cesta k atlasu frekvence lézí
            - output_dir: Adresář pro ukládání výstupů
            - batch_size: Velikost dávky
            - epochs: Počet epoch
            - lr: Learning rate
            - b1, b2: Beta parametry pro Adam optimalizér
            - save_interval: Interval pro ukládání checkpointů
            - device: Zařízení pro trénink (cpu/cuda)
            - seed: Seed pro reprodukovatelnost
            - debug: Režim pro diagnostiku a ladění
    """
    # Nastavení seedu pro reprodukovatelnost
    set_seed(args.seed)
    
    # Aktivace debug režimu
    debug_mode = hasattr(args, 'debug') and args.debug
    
    # Kontrola a vytvoření výstupního adresáře
    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Inicializace dataset a dataloader
    print("Inicializace datasetu...")
    dataset = LabelGANDataset(
        label_dir=args.label_dir,
        lesion_atlas_path=args.lesion_atlas_path,
        augment=True  # Použití augmentací
    )
    
    # Kontrola rozměrů dat a ověření vhodné hloubky sítě
    if len(dataset) > 0:
        sample = dataset[0]
        label_shape = sample['label'].shape
        atlas_shape = sample['lesion_atlas'].shape
        
        if debug_mode:
            print(f"Rozměry dat: LABEL mapa {label_shape}, Atlas lézí {atlas_shape}")
            
            # Zjištění minimálního rozměru a doporučené hloubky
            min_dim = min(label_shape[1:])
            max_depth = 0
            while min_dim > 1:
                min_dim = min_dim // 2
                max_depth += 1
            
            recommended_depth = min(max_depth - 1, 5)  # Omezení max. hloubky na 5
            if args.gen_depth > recommended_depth:
                print(f"VAROVÁNÍ: Zadaná hloubka sítě (gen_depth={args.gen_depth}) může být příliš velká pro dané rozměry dat!")
                print(f"Doporučená hloubka pro tyto rozměry je: {recommended_depth}")
                print(f"Malý rozměr (Z=64) může způsobit problémy při downsamplingu ve větší hloubce.")
                
                # V debug režimu můžeme automaticky upravit hloubku
                if debug_mode:
                    print(f"V debug režimu automaticky upravuji hloubku na doporučenou hodnotu {recommended_depth}")
                    args.gen_depth = recommended_depth
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == "cuda" else False
    )
    
    # Inicializace modelů
    print("Inicializace modelů...")
    generator = LesionGenerator(
        in_channels=2,  # atlas lézí + šum
        features=args.gen_features,
        depth=args.gen_depth,
        dropout_rate=args.dropout_rate,
        use_attention=args.use_attention
    ).to(args.device)
    
    discriminator = LesionDiscriminator(
        in_channels=2,  # atlas lézí + LABEL mapa
        features=args.disc_features,
        use_spectral_norm=args.use_spectral_norm,
        depth=args.disc_depth
    ).to(args.device)
    
    # Registrace debug hooks, pokud je debug mód aktivován
    if debug_mode:
        print("Registruji debug hooks pro sledování rozměrů tenzorů...")
        
        def hook_fn(module, input, output):
            module_name = module.__class__.__name__
            input_shapes = [x.shape if isinstance(x, torch.Tensor) else "None" for x in input]
            if isinstance(output, torch.Tensor):
                output_shape = output.shape
            else:
                output_shape = "Not a tensor"
            print(f"Module {module_name}: Input shapes {input_shapes}, Output shape {output_shape}")
        
        # Registrace hooks na vybrané vrstvy
        for name, module in generator.named_modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
                module.register_forward_hook(hook_fn)
                
        # Výpis architektury modelu
        print("\nArchitektura generátoru:")
        print(generator)
        print("\nArchitektura diskriminátoru:")
        print(discriminator)
        
        # Testovací průchod generátorem pro ověření rozměrů
        print("\nTestovací průchod generátorem pro ověření rozměrů...")
        with torch.no_grad():
            test_input_atlas = torch.randn(1, 1, 128, 128, 64).to(args.device)
            test_input_noise = torch.randn(1, 1, 128, 128, 64).to(args.device)
            try:
                test_output = generator(test_input_atlas, test_input_noise)
                print(f"Testovací průchod úspěšný! Vstup: {test_input_atlas.shape}, Výstup: {test_output.shape}")
                
                # Kontrola, zda výstup má stejné rozměry jako vstup
                if test_output.shape[2:] != test_input_atlas.shape[2:]:
                    print(f"VAROVÁNÍ: Výstup generátoru má jiné rozměry ({test_output.shape[2:]}) než vstup ({test_input_atlas.shape[2:]})!")
                    print("Pro trénink je důležité, aby výstup měl stejné rozměry jako vstup.")
            except Exception as e:
                print(f"Chyba při testovacím průchodu: {e}")
                print("Zkontrolujte architekturu sítě a hloubku.")
                if hasattr(e, '__traceback__'):
                    import traceback
                    traceback.print_tb(e.__traceback__)
    
    # Inicializace optimizérů
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    
    # Schedulery pro postupné snižování learning rate
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='min', factor=0.5, patience=5, verbose=True
    )
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Kritéria pro loss funkce
    # Pro adversarial loss použijeme BCE s logits
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    # Weighted BCE pro řešení imbalance mezi pozitivními a negativními voxely v LABEL mapách
    # Tato funkce bude přikládat větší váhu menšinovým třídám (léze)
    def weighted_binary_cross_entropy(output, target):
        # Vypočítáme váhy založené na procentu pozitivních voxelů v každém vzorku
        # Tím zajistíme, že dáme větší váhu pozitivním voxelům, které jsou méně časté
        batch_size = target.size(0)
        weights = []
        
        for i in range(batch_size):
            pos = torch.sum(target[i])
            neg = torch.numel(target[i]) - pos
            pos_weight = neg / (pos + 1e-8)  # Prevence dělení nulou
            
            # Omezení maximální váhy, aby nebyla příliš vysoká
            pos_weight = torch.clamp(pos_weight, 1.0, 50.0)
            
            # Vytvoření váhové mapy pro vzorek
            weight_map = torch.ones_like(target[i])
            weight_map[target[i] > 0.5] = pos_weight
            
            weights.append(weight_map)
        
        weight_tensor = torch.stack(weights)
        
        # BCE loss s váhami
        loss = F.binary_cross_entropy_with_logits(output, target, weight=weight_tensor)
        return loss
    
    # Dice loss pro spatial overlap lézí
    def dice_loss(pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)  # Pro použití s logits
        
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Výpočet Dice koeficientu
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        
        # Dice loss
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    # Focal loss pro řešení imbalance
    def focal_loss(pred, target, alpha=0.8, gamma=2.0):
        # Sigmoid
        pred = torch.sigmoid(pred)
        
        # Focal Loss
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** gamma
        
        # Váha pro pozitivní a negativní třídy
        alpha_weight = target * alpha + (1 - target) * (1 - alpha)
        
        # Kombinace
        focal_loss = -alpha_weight * focal_weight * torch.log(pt + 1e-8)
        return focal_loss.mean()
    
    # Gradient penalty pro WGAN-GP
    def compute_gradient_penalty(D, real_samples, fake_samples, lesion_atlas):
        """WGAN-GP gradient penalty"""
        # Náhodný koeficient pro interpolaci mezi reálnými a fake vzorky
        alpha = torch.rand((real_samples.size(0), 1, 1, 1, 1), device=args.device)
        
        # Interpolované vzorky
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        # Diskriminační skóre pro interpolované vzorky
        d_interpolates = D(lesion_atlas, interpolates)
        
        # Vytvoření plných jedniček
        fake = torch.ones((real_samples.size(0), 1, d_interpolates.size(2), 
                          d_interpolates.size(3), d_interpolates.size(4)), 
                         device=args.device, requires_grad=False)
        
        # Výpočet gradientů
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Výpočet normy gradientů pro každý vzorek
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    # Kombinovaná loss funkce pro generator
    def generator_loss(fake_pred, fake_label, real_label, use_dice=True, use_focal=True):
        # Adversarial loss
        adv_loss = adversarial_loss(fake_pred, torch.ones_like(fake_pred))
        
        # Váhovaná BCE loss
        wbce_loss = weighted_binary_cross_entropy(fake_label, real_label)
        
        # Dice loss pro lepší spatial overlap
        d_loss = dice_loss(fake_label, real_label) if use_dice else 0.0
        
        # Focal loss pro řešení class imbalance
        f_loss = focal_loss(fake_label, real_label) if use_focal else 0.0
        
        # Kombinovaná loss
        g_loss = adv_loss + wbce_loss + d_loss + f_loss
        
        return g_loss, {
            'adv_loss': adv_loss.item(),
            'wbce_loss': wbce_loss.item(),
            'dice_loss': d_loss if use_dice else 0.0,
            'focal_loss': f_loss if use_focal else 0.0,
            'total': g_loss.item()
        }
    
    # Začátek tréninku
    print("Začínám trénink...")
    global_step = 0
    best_g_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_g_losses = []
        epoch_d_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            # Přesun dat na zařízení
            real_labels = batch['label'].to(args.device)
            lesion_atlas = batch['lesion_atlas'].to(args.device)
            noise = batch['noise'].to(args.device)
            
            batch_size = real_labels.size(0)
            
            # Výpis rozměrů v debug režimu
            if debug_mode and global_step == 0:
                print(f"Rozměry vstupních dat - real_labels: {real_labels.shape}, lesion_atlas: {lesion_atlas.shape}, noise: {noise.shape}")
            
            # ----------
            # Trénink diskriminátoru
            # ----------
            optimizer_D.zero_grad()
            
            # Generování fake LABEL map
            with torch.no_grad():
                fake_labels = generator(lesion_atlas, noise)
                
                # Kontrola rozměrů v debug režimu
                if debug_mode and global_step == 0:
                    print(f"Rozměry generovaných dat - fake_labels: {fake_labels.shape}")
                    
                    # Kontrola shodnosti rozměrů
                    if fake_labels.shape != real_labels.shape:
                        print(f"VAROVÁNÍ: Rozměry generovaných dat ({fake_labels.shape}) se neshodují s reálnými daty ({real_labels.shape})!")
                        
                        # Interpolace na správnou velikost
                        fake_labels = F.interpolate(fake_labels, size=real_labels.shape[2:], mode='trilinear', align_corners=False)
                        print(f"Provedena interpolace - nové rozměry: {fake_labels.shape}")
            
            # Diskriminační skóre pro reálné a fake vzorky
            real_pred = discriminator(lesion_atlas, real_labels)
            fake_pred = discriminator(lesion_atlas, fake_labels.detach())
            
            # Trénovací postupy pro diskriminátor:
            if args.use_wgan:
                # WGAN-GP
                gradient_penalty = compute_gradient_penalty(
                    discriminator, real_labels, fake_labels.detach(), lesion_atlas
                )
                
                # Wasserstein distance
                d_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + args.lambda_gp * gradient_penalty
            else:
                # Standardní GAN loss
                real_loss = adversarial_loss(real_pred, torch.ones_like(real_pred))
                fake_loss = adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
                d_loss = (real_loss + fake_loss) / 2
            
            # Backpropagation
            d_loss.backward()
            optimizer_D.step()
            
            # Omezení D tréninku pro lepší vyváženost
            if global_step % args.n_critic == 0:
                # ----------
                # Trénink generátoru
                # ----------
                optimizer_G.zero_grad()
                
                # Generování fake LABEL map
                fake_labels = generator(lesion_atlas, noise)
                
                # Kontrola a případná úprava rozměrů
                if fake_labels.shape != real_labels.shape:
                    if debug_mode:
                        print(f"Úprava rozměrů generovaných dat: {fake_labels.shape} -> {real_labels.shape}")
                    fake_labels = F.interpolate(fake_labels, size=real_labels.shape[2:], mode='trilinear', align_corners=False)
                
                # Diskriminační skóre pro fake vzorky (pohled G)
                fake_pred = discriminator(lesion_atlas, fake_labels)
                
                # Výpočet loss pro generator
                g_loss, g_loss_components = generator_loss(
                    fake_pred, fake_labels, real_labels,
                    use_dice=args.use_dice_loss,
                    use_focal=args.use_focal_loss
                )
                
                # Backpropagation
                g_loss.backward()
                optimizer_G.step()
                
                # Sledování ztrát
                epoch_g_losses.append(g_loss.item())
            
            # Sledování ztrát diskriminátoru
            epoch_d_losses.append(d_loss.item())
            
            # Aktualizace progress baru
            progress_bar.set_postfix({
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item() if global_step % args.n_critic == 0 else "N/A"
            })
            
            # Ukládání vzorků v pravidelných intervalech
            if global_step % args.save_interval == 0:
                with torch.no_grad():
                    sample_data = {
                        'label': real_labels,
                        'lesion_atlas': lesion_atlas,
                        'fake_label': generator(lesion_atlas, noise)
                    }
                    # Kontrola a případná úprava rozměrů
                    if sample_data['fake_label'].shape != real_labels.shape:
                        sample_data['fake_label'] = F.interpolate(
                            sample_data['fake_label'], 
                            size=real_labels.shape[2:], 
                            mode='trilinear', 
                            align_corners=False
                        )
                    save_sample(
                        sample_data,
                        os.path.join(samples_dir, f"step_{global_step}")
                    )
            
            global_step += 1
        
        # Průměrné ztráty za epochu
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses) if epoch_g_losses else float('inf')
        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        
        print(f"Epoch {epoch+1}/{args.epochs} | G_loss: {avg_g_loss:.4f} | D_loss: {avg_d_loss:.4f}")
        
        # Aktualizace scheduler
        scheduler_G.step(avg_g_loss)
        scheduler_D.step(avg_d_loss)
        
        # Ukládání modelů
        if (epoch + 1) % 5 == 0 or avg_g_loss < best_g_loss:
            state = {
                'epoch': epoch,
                'global_step': global_step,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
            }
            
            # Ukládání aktuálního checkpointu
            torch.save(
                state, 
                os.path.join(models_dir, f"checkpoint_e{epoch+1}.pt")
            )
            
            # Ukládání nejlepšího modelu podle G loss
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                torch.save(
                    state, 
                    os.path.join(models_dir, "best_model.pt")
                )
                print(f"Uložen nový nejlepší model s G_loss: {best_g_loss:.4f}")
    
    print("Trénink dokončen!")
    return generator, discriminator

def generate_lesion_samples(args):
    """
    Generuje vzorky lézí pomocí natrénovaného modelu
    
    Args:
        args: Argumenty pro generování vzorků, včetně:
            - model_path: Cesta k natrénovanému modelu
            - lesion_atlas_path: Cesta k atlasu frekvence lézí
            - output_dir: Adresář pro ukládání výstupů
            - n_samples: Počet vzorků k vygenerování
            - device: Zařízení pro generování (cpu/cuda)
            - seed: Seed pro reprodukovatelnost
            - use_thresholding: Použít thresholding pro binarizaci výstupů
            - threshold: Práh pro binarizaci (0-1)
    """
    # Nastavení seedu pro reprodukovatelnost
    set_seed(args.seed)
    
    # Kontrola a vytvoření výstupního adresáře
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Načtení atlasu lézí
    print("Načítání atlasu lézí...")
    if args.lesion_atlas_path.endswith('.nii.gz') or args.lesion_atlas_path.endswith('.nii'):
        lesion_atlas = nib.load(args.lesion_atlas_path)
        lesion_atlas_data = lesion_atlas.get_fdata()
        affine = lesion_atlas.affine
    else:
        lesion_atlas_sitk = sitk.ReadImage(args.lesion_atlas_path)
        lesion_atlas_data = sitk.GetArrayFromImage(lesion_atlas_sitk)
        affine = np.eye(4)  # Defaultní affine, pokud není k dispozici
    
    # Normalizace do rozsahu [0, 1]
    lesion_atlas_data = (lesion_atlas_data - lesion_atlas_data.min()) / (lesion_atlas_data.max() - lesion_atlas_data.min())
    
    # Výpis informací o rozměrech
    print(f"Rozměry atlasu lézí: {lesion_atlas_data.shape}")
    
    # Převod na tensor
    lesion_atlas_tensor = torch.FloatTensor(lesion_atlas_data).unsqueeze(0).unsqueeze(0).to(args.device)
    
    # Načtení modelu
    print("Načítání modelu...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # Inicializace generátoru
    generator = LesionGenerator(
        in_channels=2,
        features=args.gen_features,
        depth=args.gen_depth,
        dropout_rate=args.dropout_rate,
        use_attention=args.use_attention
    ).to(args.device)
    
    try:
        # Načtení vah
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print("Model úspěšně načten.")
    except Exception as e:
        print(f"Chyba při načítání modelu: {e}")
        print("Zkouším alternativní klíče...")
        # Zkusit jiné možné klíče v checkpointu
        for key in checkpoint.keys():
            print(f"Dostupný klíč: {key}")
        
        if 'state_dict' in checkpoint:
            print("Zkouším načíst 'state_dict'...")
            try:
                generator.load_state_dict(checkpoint['state_dict'])
                print("Model úspěšně načten z 'state_dict'.")
            except Exception as e2:
                print(f"Chyba při alternativním načítání: {e2}")
                return False
    
    # Registrace debug hooks, pokud je debug mód aktivován
    if hasattr(args, 'debug') and args.debug:
        print("Registruji debug hooks pro sledování rozměrů tenzorů...")
        
        def hook_fn(module, input, output):
            module_name = module.__class__.__name__
            input_shapes = [x.shape if isinstance(x, torch.Tensor) else "None" for x in input]
            if isinstance(output, torch.Tensor):
                output_shape = output.shape
            else:
                output_shape = "Not a tensor"
            print(f"Module {module_name}: Input shapes {input_shapes}, Output shape {output_shape}")
        
        # Registrace hooks na vybrané vrstvy
        for name, module in generator.named_modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
                module.register_forward_hook(hook_fn)
        
        # Výpis architektury modelu
        print("\nArchitektura generátoru:")
        print(generator)
    
    generator.eval()
    
    print(f"Generuji {args.n_samples} vzorků lézí...")
    
    # Generování vzorků
    with torch.no_grad():
        for i in tqdm(range(args.n_samples)):
            # Vytvoření šumu
            noise = torch.randn_like(lesion_atlas_tensor).to(args.device)
            
            # Generování léze
            fake_label = generator(lesion_atlas_tensor, noise)
            
            # Aplikace thresholdingu, pokud je vyžadován
            if args.use_thresholding:
                fake_label = (fake_label > args.threshold).float()
            
            # Převod na numpy array
            fake_label_np = fake_label.squeeze().cpu().numpy()
            
            # Uložení jako NIfTI soubor
            fake_nifti = nib.Nifti1Image(fake_label_np, affine)
            output_path = os.path.join(args.output_dir, f"generated_lesion_{i+1}.nii.gz")
            nib.save(fake_nifti, output_path)
            
            # Vytvoření vizualizací
            if i < 10:  # Vizualizujeme pouze prvních 10 vzorků
                # Najdeme středové řezy nebo řezy s lézemi
                D, H, W = fake_label_np.shape
                
                # Hledání řezů s lézemi
                d_slices = [d for d in range(D) if fake_label_np[d, :, :].max() > 0]
                h_slices = [h for h in range(H) if fake_label_np[:, h, :].max() > 0]
                w_slices = [w for w in range(W) if fake_label_np[:, :, w].max() > 0]
                
                # Pokud nejsou nalezeny řezy s lézemi, použijeme středové řezy
                slice_idx = [
                    d_slices[len(d_slices)//2] if d_slices else D//2,
                    h_slices[len(h_slices)//2] if h_slices else H//2,
                    w_slices[len(w_slices)//2] if w_slices else W//2
                ]
                
                # Vytvoření vizualizací - axiální, koronální a sagitální řez
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                
                # Axiální řez
                axs[0].imshow(fake_label_np[slice_idx[0], :, :], cmap='binary')
                axs[0].set_title(f'Axial (z={slice_idx[0]})')
                axs[0].axis('off')
                
                # Koronální řez
                axs[1].imshow(fake_label_np[:, slice_idx[1], :], cmap='binary')
                axs[1].set_title(f'Coronal (y={slice_idx[1]})')
                axs[1].axis('off')
                
                # Sagitální řez
                axs[2].imshow(fake_label_np[:, :, slice_idx[2]], cmap='binary')
                axs[2].set_title(f'Sagittal (x={slice_idx[2]})')
                axs[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"visualization_{i+1}.png"), dpi=200)
                plt.close()
    
    print(f"Vygenerováno {args.n_samples} vzorků lézí do {args.output_dir}")
    return True

def main():
    """Hlavní funkce programu"""
    parser = argparse.ArgumentParser(description="LabelGAN - generování realistických lézí pro HIE")
    
    # Obecné argumenty
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"],
                        help="Režim běhu: 'train' nebo 'generate'")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed pro reprodukovatelnost")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Zařízení pro trénink/generování (cuda/cpu)")
    parser.add_argument("--debug", action="store_true",
                       help="Zapnout debug režim s výpisem rozměrů tenzorů")
    
    # Argumenty pro dataset
    parser.add_argument("--label_dir", type=str, default="data/BONBID2023_Train/3LABEL",
                        help="Adresář s registrovanými LABEL mapami")
    parser.add_argument("--lesion_atlas_path", type=str, default="data/archive/lesion_atlases/lesion_atlas.nii",
                        help="Cesta k atlasu frekvence lézí")
    parser.add_argument("--output_dir", type=str, default="output/labelgan",
                        help="Adresář pro ukládání výstupů")
    
    # Argumenty pro model
    parser.add_argument("--gen_features", type=int, default=64,
                        help="Počet základních feature map v generátoru")
    parser.add_argument("--gen_depth", type=int, default=4,
                        help="Hloubka U-Net architektury v generátoru")
    parser.add_argument("--disc_features", type=int, default=64,
                        help="Počet základních feature map v diskriminátoru")
    parser.add_argument("--disc_depth", type=int, default=4,
                        help="Hloubka architektury v diskriminátoru")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Míra dropout v generátoru")
    parser.add_argument("--use_attention", action="store_true", 
                        help="Použít attention mechanismus pro lepší detaily")
    parser.add_argument("--use_spectral_norm", action="store_true",
                        help="Použít spektrální normalizaci pro stabilnější trénink")
    
    # Argumenty pro trénink
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Velikost dávky")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Počet epoch")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="Beta1 parametr pro Adam optimalizér")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="Beta2 parametr pro Adam optimalizér")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Interval pro ukládání vzorků (v krocích)")
    parser.add_argument("--use_wgan", action="store_true",
                        help="Použít WGAN-GP místo standardního GAN")
    parser.add_argument("--lambda_gp", type=float, default=10.0,
                        help="Váha gradient penalty pro WGAN-GP")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Počet aktualizací diskriminátoru na jednu aktualizaci generátoru")
    parser.add_argument("--use_dice_loss", action="store_true",
                        help="Použít Dice loss pro lepší prostorovou shodu lézí")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Použít Focal loss pro řešení class imbalance")
    
    # Argumenty pro generování
    parser.add_argument("--model_path", type=str, default="output/labelgan/models/best_model.pt",
                        help="Cesta k natrénovanému modelu pro generování")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Počet vzorků k vygenerování")
    parser.add_argument("--use_thresholding", action="store_true",
                        help="Použít thresholding pro binarizaci výstupů")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Práh pro binarizaci (0-1)")
    
    args = parser.parse_args()
    
    # Logování argumentů
    print("Spouštím LabelGAN s následujícími argumenty:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Zapnutí debugovacího režimu
    if args.debug:
        def hook_fn(module, input, output):
            module_name = module.__class__.__name__
            print(f"Shape for {module_name}: input {[x.shape if x is not None else None for x in input]}, output {output.shape}")
        
        def register_hooks(model):
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
                    module.register_forward_hook(hook_fn)
        
        print("Debug režim je zapnutý - budou vypisovány informace o rozměrech tenzorů")
    
    # Spuštění příslušného režimu
    if args.mode == "train":
        print("Kontrola rozměrů vstupních dat...")
        # Načtení vzorku dat pro kontrolu rozměrů
        dataset = LabelGANDataset(
            label_dir=args.label_dir,
            lesion_atlas_path=args.lesion_atlas_path,
            augment=False
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            label_shape = sample['label'].shape
            atlas_shape = sample['lesion_atlas'].shape
            print(f"Rozměry dat: LABEL mapa {label_shape}, Atlas lézí {atlas_shape}")
            
            # Doporučení vhodné hloubky sítě
            min_dim = min(label_shape[1:])
            max_depth = 0
            while min_dim > 1:
                min_dim = min_dim // 2
                max_depth += 1
            
            recommended_depth = min(max_depth - 1, 5)  # Omezení max. hloubky na 5
            if args.gen_depth > recommended_depth:
                print(f"VAROVÁNÍ: Zadaná hloubka sítě (gen_depth={args.gen_depth}) může být příliš velká pro dané rozměry dat.")
                print(f"Doporučená hloubka: {recommended_depth}")
                
                # Automatická korekce hloubky, pokud je zapnutý debug režim
                if args.debug:
                    args.gen_depth = recommended_depth
                    print(f"V debug režimu automaticky upravuji hloubku na doporučenou hodnotu {recommended_depth}")
        
        # Spuštění tréninku
        train_lesion_gan(args)
    elif args.mode == "generate":
        generate_lesion_samples(args)
    else:
        print(f"Neznámý režim: {args.mode}")

if __name__ == "__main__":
    main()
