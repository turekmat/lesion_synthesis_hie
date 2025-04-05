import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from tqdm import tqdm
import random
import argparse
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    CropForegroundd,
    ResizeWithPadOrCropd
)

# Nastavení seed pro reprodukovatelnost
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset třída pro léze s atlasem
class LesionDataset(Dataset):
    def __init__(self, labels_dir, lesion_atlas_path, transform=None, image_size=(96, 96, 96)):
        self.labels_dir = labels_dir
        
        # Kontrola, zda adresář existuje
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Adresář s daty neexistuje: {labels_dir}")
        
        # Načtení všech souborů .nii a .nii.gz
        all_files = []
        nii_files = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.nii')]
        nii_gz_files = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.nii.gz')]
        all_files = nii_files + nii_gz_files
        
        if not all_files:
            raise ValueError(f"Adresář {labels_dir} neobsahuje žádné .nii nebo .nii.gz soubory")
        
        print(f"\n--- Inicializace datasetu ---")
        print(f"Adresář: {labels_dir}")
        print(f"Počet nalezených souborů: {len(all_files)} (.nii: {len(nii_files)}, .nii.gz: {len(nii_gz_files)})")
        
        # Filtrování souborů, které neobsahují žádné léze
        self.labels_files = []
        skipped_count = 0
        empty_count = 0
        error_count = 0
        
        for file_path in sorted(all_files):
            try:
                nii_img = nib.load(file_path)
                data = nii_img.get_fdata()
                
                # Kontrola, zda soubor obsahuje nějaké léze (nenulové hodnoty)
                if np.any(data > 0):
                    self.labels_files.append(file_path)
                else:
                    empty_count += 1
                    skipped_count += 1
            except Exception as e:
                print(f"Chyba při kontrole souboru {file_path}: {str(e)}")
                error_count += 1
                skipped_count += 1
        
        # Výpis statistik o načtených souborech
        print(f"\n--- Výsledky zpracování souborů ---")
        print(f"Počet souborů s lézemi: {len(self.labels_files)}")
        print(f"Počet souborů bez lézí (prázdné): {empty_count}")
        print(f"Počet souborů s chybou při načtení: {error_count}")
        print(f"Celkem přeskočeno: {skipped_count}")
        
        if self.labels_files:
            # Ukázkový soubor pro informace o rozměrech
            sample_img = nib.load(self.labels_files[0])
            sample_data = sample_img.get_fdata()
            print(f"\n--- Ukázkový soubor ---")
            print(f"Rozměry: {sample_data.shape}")
            print(f"Datový typ: {sample_data.dtype}")
            print(f"Rozsah hodnot: {np.min(sample_data)} - {np.max(sample_data)}")
        
        if not self.labels_files:
            raise ValueError(f"Žádný z nalezených souborů neobsahuje léze. Dataset je prázdný.")
        
        self.lesion_atlas_path = lesion_atlas_path
        self.transform = transform
        self.image_size = image_size
        
        # Načtení atlasu lézí
        atlas_nii = nib.load(lesion_atlas_path)
        atlas_data = atlas_nii.get_fdata()
        self.atlas_affine = atlas_nii.affine
        
        # Informace o atlasu lézí
        print(f"\n--- Atlas lézí ---")
        print(f"Cesta: {lesion_atlas_path}")
        print(f"Rozměry: {atlas_data.shape}")
        print(f"Rozsah hodnot: {np.min(atlas_data)} - {np.max(atlas_data)}")
        
        # Normalizace atlasu
        if np.max(atlas_data) > 0:
            atlas_data = atlas_data / np.max(atlas_data)
        
        self.atlas_data = atlas_data
        print(f"\nDataset úspěšně inicializován s {len(self.labels_files)} vzorky.")
        print("-" * 50)
    
    def __len__(self):
        return len(self.labels_files)
    
    def load_nifti(self, path):
        try:
            nii_img = nib.load(path)
            data = nii_img.get_fdata()
            
            # Odstraněny podrobné výpisy pro každý soubor
            
            # Převedení na binární masku
            data = (data > 0).astype(np.float32)
            
            # Normalizace dat
            if np.max(data) > 0:
                data = data / np.max(data)
                
            return data, nii_img.affine
        except Exception as e:
            print(f"Chyba při načítání souboru {path}: {str(e)}")
            return np.zeros(self.image_size, dtype=np.float32), np.eye(4)
    
    def __getitem__(self, idx):
        label_path = self.labels_files[idx]
        
        # Načtení a příprava labeled dat
        label_data, affine = self.load_nifti(label_path)
        
        # Resize na požadovanou velikost (zjednodušeno - předpokládáme, že všechny jsou stejné velikosti)
        # V produkčním prostředí by bylo lepší použít MONAI transformace
        
        # Příprava dat jako tensor
        label_tensor = torch.tensor(label_data).unsqueeze(0).float()  # Přidáme kanálovou dimenzi
        atlas_tensor = torch.tensor(self.atlas_data).unsqueeze(0).float()  # Přidáme kanálovou dimenzi
        
        # Generování náhodného šumu
        noise = torch.randn(100, 1, 1, 1)
        
        sample = {
            'label': label_tensor,
            'atlas': atlas_tensor,
            'noise': noise,
            'path': label_path
        }
        
        return sample

# SwinUNETR Generator
class SwinUNETRGenerator(nn.Module):
    def __init__(self, img_size=(96, 96, 96), in_channels=101, feature_size=48):
        super(SwinUNETRGenerator, self).__init__()
        
        # SwinUNETR hlavní komponenta
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,  # Šum (100) + atlas (1)
            out_channels=16,  # Mezivýstup
            feature_size=feature_size,
        )
        
        # Finální konvoluční vrstvy pro generování lézí
        self.final = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Decoder pro multi-pattern generování
        self.pattern_decoder = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Konverze vzorů na léze
        self.pattern_to_lesion = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, noise, atlas):
        batch_size = atlas.size(0)
        
        # Příprava šumu - rozšíření na prostorové dimenze
        noise_expanded = noise.expand(batch_size, 100, 
                                     atlas.size(2), atlas.size(3), atlas.size(4))
        
        # Kombinace šumu a atlasu
        x = torch.cat([noise_expanded, atlas], dim=1)
        
        # Průchod SwinUNETR
        features = self.swin_unetr(x)
        
        # Základní výstup
        base_output = self.final(features)
        
        # Multi-pattern přístup
        lesion_patterns = self.pattern_decoder(features)
        outputs = [base_output]
        
        # Generování různých vzorů lézí
        for i in range(4):  # Menší počet vzorů pro jednoduchost
            pattern = lesion_patterns[:, i:i+1]
            sub_output = self.pattern_to_lesion(pattern)
            outputs.append(sub_output)
        
        # Kombinace všech vzorů pomocí max operace
        combined_output = outputs[0]
        for i in range(1, len(outputs)):
            combined_output = torch.max(combined_output, outputs[i])
        
        # Maskování atlasem lézí - generovat léze pouze v povolených oblastech
        masked_output = combined_output * (atlas > 0.01).float()
        
        return masked_output

# Patch-based Discriminator
class LesionDiscriminator(nn.Module):
    def __init__(self):
        super(LesionDiscriminator, self).__init__()
        
        # PatchGAN diskriminátor se spektrální normalizací
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(2, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Výstupní vrstva - patch klasifikace
        self.output = nn.Conv3d(512, 1, kernel_size=3, padding=1)
    
    def forward(self, x, atlas):
        # Kombinace vstupu s atlasem
        x = torch.cat([x, atlas], dim=1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return self.output(x)

# Ztrátové funkce a pomocné funkce
def compute_gradient_penalty(discriminator, real_samples, fake_samples, atlas, device):
    """Výpočet penalizace gradientu pro Wasserstein GAN s gradient penalty"""
    batch_size = real_samples.size(0)
    
    # Náhodné váhy pro interpolaci
    alpha = torch.rand(batch_size, 1, 1, 1, 1).to(device)
    
    # Interpolace mezi reálnými a generovanými vzorky
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)
    
    # Průchod diskriminátorem
    d_interpolates = discriminator(interpolates, atlas)
    
    # Výpočet gradientů
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Výpočet penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def compute_lesion_size_loss(generated, atlas):
    """Penalizace pro příliš malé nebo příliš velké léze"""
    batch_size = generated.size(0)
    
    # Binární maska lézí
    binary = (generated > 0.5).float()
    
    loss = 0.0
    for i in range(batch_size):
        bin_mask = binary[i, 0].cpu().numpy()
        atlas_mask = atlas[i, 0].cpu().numpy() > 0
        
        # Výpočet počtu komponent
        labeled, num_components = ndimage.label(bin_mask)
        
        if num_components == 0:
            # Penalizace za žádné léze
            loss += torch.tensor(5.0, device=generated.device)
        else:
            # Výpočet velikostí komponent
            component_sizes = ndimage.sum(bin_mask, labeled, range(1, num_components+1))
            
            # Preferujeme menší počet větších lézí
            avg_size = np.mean(component_sizes) if len(component_sizes) > 0 else 0
            size_factor = 1.0 / (1.0 + avg_size)  # Klesající s velikostí
            count_factor = np.log1p(num_components) / 3.0  # Rostoucí s počtem, ale pomaleji
            
            combined_factor = size_factor * count_factor
            loss += torch.tensor(combined_factor, device=generated.device)
    
    return loss / batch_size

def compute_anatomical_consistency_loss(generated, atlas):
    """Zajištění anatomické konzistence - léze by měly být převážně v oblastech definovaných atlasem"""
    # Binární maska lézí
    binary = (generated > 0.5).float()
    
    # Maska atlasu
    atlas_mask = (atlas > 0.01).float()
    
    # Léze mimo atlas
    outside_atlas = binary * (1 - atlas_mask)
    
    # Penalizace za léze mimo povolené oblasti
    loss = torch.sum(outside_atlas) / (torch.sum(binary) + 1e-8)
    
    return loss

def compute_coverage_loss(generated, atlas, min_coverage=0.01, max_coverage=65.5):
    """
    Penalizuje léze s celkovým pokrytím mimo požadovaný rozsah (0.01% - 65.5%)
    
    Args:
        generated: Generovaný výstup modelu
        atlas: Atlas definující povolené oblasti pro léze (funguje jako maska mozku)
        min_coverage: Minimální akceptované pokrytí (0.01%)
        max_coverage: Maximální akceptované pokrytí (65.5% v rámci masky mozku)
                     (Pro celý objem by to odpovídalo přibližně 3.5%)
    
    Returns:
        Penalizační loss pro pokrytí mimo požadovaný rozsah
    """
    # Binarizace generovaného výstupu
    binary = (generated > 0.5).float()
    
    # Maska mozku - atlas slouží jako maska mozku
    brain_mask = (atlas > 0.01).float()
    
    batch_size = generated.size(0)
    loss = 0.0
    
    for i in range(batch_size):
        # Výpočet celkového pokrytí v rámci mozku
        total_brain_voxels = torch.sum(brain_mask[i])
        lesion_voxels = torch.sum(binary[i] * brain_mask[i])  # Počítáme pouze léze uvnitř mozku
        
        if total_brain_voxels > 0:
            coverage_ratio = (lesion_voxels / total_brain_voxels) * 100  # Převedeno na procenta
            
            # Penalizace pokud je pokrytí mimo požadovaný rozsah
            if coverage_ratio < min_coverage:
                # Kvadratická penalizace pro příliš malé pokrytí
                loss += 10.0 * (min_coverage - coverage_ratio) ** 2
            elif coverage_ratio > max_coverage:
                # Kvadratická penalizace pro příliš velké pokrytí
                loss += 10.0 * (coverage_ratio - max_coverage) ** 2
            # Pokud je v rozsahu, žádná penalizace
    
    return loss / batch_size

def compute_dice_score(pred, target):
    """Výpočet Dice skóre mezi predikcí a cílem"""
    smooth = 1e-5
    pred_binary = (pred > 0.5).float()
    intersection = torch.sum(pred_binary * target)
    return (2. * intersection + smooth) / (torch.sum(pred_binary) + torch.sum(target) + smooth)

def generate_diverse_noise(batch_size=1, z_dim=100, device=None):
    """Generování diverzního šumu pro vstup generátoru"""
    if device is None:
        device = torch.device("cpu")
    
    # Základní gaussovský šum
    noise = torch.randn(batch_size, z_dim, 1, 1, 1, device=device)
    
    # Zvýšení variability pomocí škálování a perturbací
    scaling_factor = torch.rand(batch_size, 1, 1, 1, 1, device=device) * 0.5 + 0.75
    noise = noise * scaling_factor
    
    # Přidání lokálních perturbací pro lepší variabilitu lézí
    perturbation_mask = (torch.rand(batch_size, z_dim, 1, 1, 1, device=device) > 0.8).float()
    perturbation = torch.randn(batch_size, z_dim, 1, 1, 1, device=device) * 0.3
    noise = noise + perturbation * perturbation_mask
    
    return noise

# Vylepšení: Přidání funkce pro výpočet pokrytí lézí pro monitorování
def calculate_lesion_coverage(generated, atlas):
    """
    Vypočítá pokrytí lézí v rámci mozku
    
    Args:
        generated: Generovaný výstup modelu
        atlas: Atlas definující povolené oblasti pro léze (funguje jako maska mozku)
    
    Returns:
        Dictionary s informacemi o pokrytí a počtu lézí
    """
    # Binarizace generovaného výstupu
    binary = (generated > 0.5).float()
    
    # Maska mozku - atlas slouží jako maska mozku
    brain_mask = (atlas > 0.001).float()
    
    batch_size = generated.size(0)
    results = []
    
    for i in range(batch_size):
        # Výpočet celkového pokrytí v rámci mozku
        bin_mask = binary[i, 0].cpu().numpy()
        brain_np = brain_mask[i, 0].cpu().numpy()
        
        total_brain_voxels = np.sum(brain_np)
        lesion_voxels = np.sum(bin_mask * brain_np)  # Počítáme pouze léze uvnitř mozku
        
        if total_brain_voxels > 0:
            coverage_percentage = (lesion_voxels / total_brain_voxels) * 100  # Převedeno na procenta
        else:
            coverage_percentage = 0.0
        
        # Výpočet počtu komponent a jejich velikostí
        bin_mask_in_brain = bin_mask * brain_np  # Léze pouze uvnitř mozku
        labeled, num_components = ndimage.label(bin_mask_in_brain)
        if num_components > 0:
            component_sizes = ndimage.sum(bin_mask_in_brain, labeled, range(1, num_components+1))
            avg_size = np.mean(component_sizes) if len(component_sizes) > 0 else 0
        else:
            avg_size = 0
        
        results.append({
            'coverage_percentage': coverage_percentage,
            'num_components': num_components,
            'avg_lesion_size': avg_size
        })
    
    return results

# Přidání nové ztrátové funkce pro atlas-guided generování
def compute_atlas_guidance_loss(generated, atlas):
    """
    Jemně navádí generátor k preferování oblastí s vyšší pravděpodobností v atlasu.
    Implementována jako lehká korekce, nikoliv striktní omezení.
    
    Hodnoty v atlasu se pohybují v rozmezí:
    - Min (nenulová): 6.12e-07
    - Max: 0.326
    - Průměr: 0.093
    
    Args:
        generated: Generovaný výstup modelu
        atlas: Atlas definující pravděpodobnost výskytu lézí v různých oblastech
    
    Returns:
        Lehký naváděcí signál pro generátor
    """
    # Binarizace generovaného výstupu
    binary = (generated > 0.5).float()
    
    batch_size = generated.size(0)
    loss = 0.0
    
    for i in range(batch_size):
        # Oblasti, kde jsou generovány léze
        lesion_areas = binary[i]
        
        if torch.sum(lesion_areas) > 0:
            # Průměrná pravděpodobnost v atlasu v místech, kde jsou generovány léze
            avg_atlas_prob = torch.sum(atlas[i] * lesion_areas) / torch.sum(lesion_areas)
            
            # Průměrná pravděpodobnost v celém atlasu (pro normalizaci)
            # Bereme v úvahu pouze nenulové hodnoty atlasu
            atlas_mask = (atlas[i] > 0.001).float()
            avg_atlas_overall = torch.sum(atlas[i] * atlas_mask) / (torch.sum(atlas_mask) + 1e-8)
            
            # Relativní skóre - jak moc se generované léze vyhýbají pravděpodobným oblastem
            # Čím nižší skóre, tím více jsou léze v méně pravděpodobných oblastech
            relative_score = avg_atlas_prob / (avg_atlas_overall + 1e-8)
            
            # Vzhledem k tomu, že průměrná hodnota atlasu je 0.093 a max je 0.326,
            # upravíme práh pro relativní skóre na 0.5 místo 0.8
            # To znamená, že penalizujeme pouze když je průměrná pravděpodobnost
            # v lézích méně než polovina průměrné pravděpodobnosti v atlasu
            if relative_score < 0.5:
                # Použijeme mírnější lineární penalizaci s malým kvadratickým prvkem
                correction = 0.2 * (0.5 - relative_score) + 0.1 * ((0.5 - relative_score) ** 2)
                loss += correction
    
    # Vrátíme jako PyTorch tensor místo float hodnoty
    if isinstance(loss, torch.Tensor):
        return loss / batch_size
    else:
        return torch.tensor(loss, device=generated.device) / batch_size

# Trénovací funkce
def train(generator, discriminator, dataloader, num_epochs, device, output_dir):
    # Vytvoření výstupního adresáře
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'generator_models'), exist_ok=True)  # Nový adresář pro modely generátoru
    
    # Vytvoření logovacího souboru pro sledování pokrytí lézí
    coverage_log_path = os.path.join(output_dir, 'lesion_coverage_log.csv')
    with open(coverage_log_path, 'w') as f:
        f.write("epoch,batch,sample,coverage_percentage,num_components,avg_lesion_size\n")
    
    # Optimizéry
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Kritéria ztráty
    bce_loss = nn.BCELoss()
    
    # Koeficienty pro vážení ztrát
    lambda_gp = 10.0
    lambda_size = 2.0
    lambda_anatomical = 5.0
    lambda_coverage = 3.0  # Snížit z 5.0
    lambda_atlas_guidance = 6.0  # Zvýšit z 1.0
    
    # Statistiky pro vykreslení
    g_losses = []
    d_losses = []
    coverage_stats = []  # Pro sledování pokrytí lézí
    atlas_guidance_losses = []  # Pro sledování efektivity navádění atlasem
    
    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_coverage_sum = 0.0
        epoch_atlas_guidance_loss = 0.0
        epoch_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            real_labels = batch['label'].to(device)
            atlas = batch['atlas'].to(device)
            
            # Generujeme diverzní šum
            noise = generate_diverse_noise(batch_size=real_labels.size(0), device=device)
            
            # Generování falešných vzorků (potřebujeme je pro oba módy tréninku)
            fake_labels = generator(noise, atlas)
            
            # ---------------------
            # Trénování diskriminátoru - pouze každou třetí iteraci
            # ---------------------
            if batch_idx % 3 == 0:
                optimizer_D.zero_grad()
                
                # Wasserstein ztráta
                real_validity = discriminator(real_labels, atlas)
                fake_validity = discriminator(fake_labels.detach(), atlas)
                
                # Wasserstein loss s gradient penalty
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                gradient_penalty = compute_gradient_penalty(discriminator, real_labels, fake_labels.detach(), atlas, device)
                d_loss += lambda_gp * gradient_penalty
                
                d_loss.backward()
                optimizer_D.step()
                
                epoch_d_loss += d_loss.item()
            
            # ---------------------
            # Trénování generátoru - častěji než diskriminátor (každou iteraci)
            # ---------------------
            optimizer_G.zero_grad()
            
            # Generujeme znovu vzorky pro trénink generátoru (pokud je trénink diskriminátoru vypnutý v této iteraci)
            if batch_idx % 3 != 0:
                fake_labels = generator(noise, atlas)
                
            fake_validity = discriminator(fake_labels, atlas)
            
            # Adversarial loss - chceme, aby diskriminátor označil generované vzorky jako reálné
            g_adv_loss = -torch.mean(fake_validity)
            
            # Velikost lézí loss
            size_loss = compute_lesion_size_loss(fake_labels, atlas)
            
            # Anatomická konzistence
            anatomical_loss = compute_anatomical_consistency_loss(fake_labels, atlas)
            
            # Pokrytí lézí v požadovaném rozsahu
            coverage_loss = compute_coverage_loss(fake_labels, atlas)
            
            # NOVĚ: Atlas-guided loss pro jemné navádění k pravděpodobnějším oblastem
            atlas_guidance_loss = compute_atlas_guidance_loss(fake_labels, atlas)
            epoch_atlas_guidance_loss += atlas_guidance_loss.item()
            
            # Celková ztráta generátoru
            g_loss = g_adv_loss + lambda_size * size_loss + lambda_anatomical * anatomical_loss + \
                     lambda_coverage * coverage_loss + lambda_atlas_guidance * atlas_guidance_loss
            
            g_loss.backward()
            optimizer_G.step()
            
            epoch_g_loss += g_loss.item()
            
            # Monitorování pokrytí lézí
            coverage_info = calculate_lesion_coverage(fake_labels.detach(), atlas)
            
            # Zápis do logu
            with open(coverage_log_path, 'a') as f:
                for i, info in enumerate(coverage_info):
                    f.write(f"{epoch+1},{batch_idx},{i},{info['coverage_percentage']:.4f},{info['num_components']},{info['avg_lesion_size']:.4f}\n")
                    epoch_coverage_sum += info['coverage_percentage']
                    epoch_samples += 1
        
        # Průměrné ztráty za epochu
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / max(1, len(dataloader) // 3)  # Upraveno pro méně časté trénování diskriminátoru
        avg_coverage = epoch_coverage_sum / max(1, epoch_samples)
        avg_atlas_guidance_loss = epoch_atlas_guidance_loss / len(dataloader)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        coverage_stats.append(avg_coverage)
        atlas_guidance_losses.append(avg_atlas_guidance_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}, Avg Coverage: {avg_coverage:.4f}%, Atlas Guidance: {avg_atlas_guidance_loss:.4f}")
        
        # Uložení checkpointu
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'epoch': epoch
            }, os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pt'))
        
        # NOVĚ: Ukládání generátoru každé 4 epochy ve formátu vhodném pro generování
        if (epoch + 1) % 4 == 0 or epoch == num_epochs - 1:
            generator_model_path = os.path.join(output_dir, 'generator_models', f'generator_epoch_{epoch+1}.pt')
            torch.save(generator.state_dict(), generator_model_path)
            print(f"Uložen model generátoru pro epoch {epoch+1} do {generator_model_path}")
            
            # Vytvoříme log soubor s parametry modelu
            with open(os.path.join(output_dir, 'generator_models', f'generator_epoch_{epoch+1}_info.txt'), 'w') as f:
                f.write(f"Epoch: {epoch+1}\n")
                f.write(f"Generator Loss: {avg_g_loss:.6f}\n")
                f.write(f"Average Coverage: {avg_coverage:.4f}%\n")
                f.write(f"Atlas Guidance Score: {avg_atlas_guidance_loss:.6f}\n")
                f.write(f"\nPokud chcete použít tento model pro generování vzorků, použijte příkaz:\n")
                f.write(f"python SwinUNETR_lesion_GAN.py --generate --model_path={generator_model_path} --lesion_atlas_path=<cesta_k_atlasu> --output_dir=<vystupni_adresar> --num_samples=<pocet_vzorku>")
        
        # Generování vzorků pro vizualizaci
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            save_samples(generator, dataloader, device, epoch, os.path.join(output_dir, 'samples'))
    
    # Vykreslení průběhu ztrát
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Ztráty
    plt.subplot(2, 2, 1)
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    
    # Plot 2: Pokrytí lézí
    plt.subplot(2, 2, 2)
    plt.plot(coverage_stats, label='Average Lesion Coverage (%)', color='green')
    plt.axhline(y=0.01, color='r', linestyle='--', label='Min Target (0.01%)')
    plt.axhline(y=65.5, color='r', linestyle='--', label='Max Target (65.5%)')
    plt.xlabel('Epoch')
    plt.ylabel('Lesion Coverage (% of Brain Volume)')
    plt.legend()
    plt.title('Average Lesion Coverage (within Brain Mask)')
    
    # Plot 3: Atlas Guidance Loss
    plt.subplot(2, 2, 3)
    plt.plot(atlas_guidance_losses, label='Atlas Guidance Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Atlas Guidance Loss (Lower is Better)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_stats.png'))
    plt.close()
    
    # Uložení finálního modelu
    torch.save(generator.state_dict(), os.path.join(output_dir, 'final_generator.pt'))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, 'final_discriminator.pt'))
    
    return generator, discriminator

# Funkce pro vizualizaci vzorků během tréninku
def save_samples(generator, dataloader, device, epoch, output_dir):
    generator.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        real_labels = batch['label'].to(device)
        atlas = batch['atlas'].to(device)
        noise = generate_diverse_noise(batch_size=real_labels.size(0), device=device)
        
        # Generování falešných vzorků
        fake_labels = generator(noise, atlas)
        
        # Analýza pokrytí lézí
        coverage_info = calculate_lesion_coverage(fake_labels, atlas)
        
        # Výběr jednoho vzorku pro vizualizaci
        real_np = real_labels[0, 0].cpu().numpy()
        fake_np = fake_labels[0, 0].cpu().numpy()
        atlas_np = atlas[0, 0].cpu().numpy()
        
        # Výpočet skóre atlas guidance pro vizualizaci
        binary = (fake_labels > 0.5).float()
        lesion_areas = binary[0]
        
        atlas_guidance_score = 0.0
        if torch.sum(lesion_areas) > 0:
            avg_atlas_prob = torch.sum(atlas[0] * lesion_areas) / torch.sum(lesion_areas)
            atlas_mask = (atlas[0] > 0.01).float()
            avg_atlas_overall = torch.sum(atlas[0] * atlas_mask) / (torch.sum(atlas_mask) + 1e-8)
            atlas_guidance_score = (avg_atlas_prob / (avg_atlas_overall + 1e-8)).item()
        
        # Najdeme střední řez nebo řez s největším obsahem léze
        mid_z = fake_np.shape[2] // 2
        if np.sum(fake_np[:, :, mid_z]) < 1:
            # Hledáme řez s největším obsahem léze
            z_sums = [np.sum(fake_np[:, :, z]) for z in range(fake_np.shape[2])]
            if max(z_sums) > 0:
                mid_z = np.argmax(z_sums)
        
        # Vizualizace
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Atlas s překryvem generovaných lézí
        axes[0].imshow(atlas_np[:, :, mid_z], cmap='viridis', alpha=0.7)
        binary_np = (fake_np > 0.5).astype(float)
        axes[0].imshow(binary_np[:, :, mid_z], cmap='gray', alpha=0.5)
        axes[0].set_title(f'Atlas s lézemi\nAtlas Guidance Score: {atlas_guidance_score:.2f}')
        axes[0].axis('off')
        
        axes[1].imshow(real_np[:, :, mid_z], cmap='gray')
        axes[1].set_title('Real Label')
        axes[1].axis('off')
        
        axes[2].imshow(fake_np[:, :, mid_z], cmap='gray')
        sample_info = coverage_info[0]
        is_in_range = 0.01 <= sample_info["coverage_percentage"] <= 65.5
        range_status = "✓" if is_in_range else "✗"
        axes[2].set_title(f'Generated Label\nCoverage: {sample_info["coverage_percentage"]:.2f}% {range_status}, Lesions: {sample_info["num_components"]}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_epoch_{epoch+1}.png'))
        plt.close()
        
        # Vytvoření log souboru s detaily
        with open(os.path.join(output_dir, f'sample_epoch_{epoch+1}_details.txt'), 'w') as f:
            for i, info in enumerate(coverage_info):
                is_in_range = 0.01 <= info["coverage_percentage"] <= 65.5
                f.write(f"Sample {i+1}:\n")
                f.write(f"  Coverage: {info['coverage_percentage']:.4f}% ({'v požadovaném rozmezí' if is_in_range else 'mimo požadované rozmezí'})\n")
                f.write(f"  Number of lesions: {info['num_components']}\n")
                f.write(f"  Average lesion size: {info['avg_lesion_size']:.4f} voxels\n")
                f.write(f"  Atlas Guidance Score: {atlas_guidance_score:.4f} (>1.0 znamená léze v pravděpodobnějších oblastech)\n\n")
    
    generator.train()

# Funkce pro generování vzorků z natrénovaného modelu
def generate_samples(model_path, lesion_atlas_path, output_dir, num_samples=10, min_threshold=0.3, feature_size=24):
    # Nastavení
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Načtení atlasu (který slouží i jako maska mozku)
    atlas_nii = nib.load(lesion_atlas_path)
    atlas_data = atlas_nii.get_fdata()
    atlas_affine = atlas_nii.affine
    
    # Normalizace atlasu
    if np.max(atlas_data) > 0:
        atlas_data = atlas_data / np.max(atlas_data)
    
    # Příprava atlasu jako tensor
    atlas_tensor = torch.tensor(atlas_data, device=device).unsqueeze(0).unsqueeze(0).float()
    
    # Načtení modelu
    img_size = atlas_data.shape
    print(f"Inicializuji generátor s feature_size={feature_size}")
    generator = SwinUNETRGenerator(img_size=img_size, feature_size=feature_size)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()
    
    # Vytvoření výstupního adresáře
    os.makedirs(output_dir, exist_ok=True)
    
    # Vytvoření souboru pro sledování statistik
    stats_file = os.path.join(output_dir, 'lesion_statistics.csv')
    with open(stats_file, 'w') as f:
        f.write("sample,coverage_percentage,num_components,avg_lesion_size,is_in_target_range,atlas_guidance_score\n")
    
    # Generování vzorků
    all_coverages = []
    all_atlas_scores = []
    
    for i in range(num_samples):
        with torch.no_grad():
            # Různé varianty šumu pro diverzitu
            noise = generate_diverse_noise(batch_size=1, device=device)
            
            # Generování vzorku
            fake_label = generator(noise, atlas_tensor)
            
            # Analýza pokrytí
            coverage_info = calculate_lesion_coverage(fake_label, atlas_tensor)[0]
            coverage_percentage = coverage_info['coverage_percentage']
            num_components = coverage_info['num_components']
            avg_size = coverage_info['avg_lesion_size']
            
            # Je pokrytí v cílovém rozsahu?
            is_in_range = 0.01 <= coverage_percentage <= 65.5
            all_coverages.append(coverage_percentage)
            
            # Výpočet skóre atlas guidance pro statistiku
            binary = (fake_label > 0.5).float()
            lesion_areas = binary[0]
            
            atlas_guidance_score = 0.0
            if torch.sum(lesion_areas) > 0:
                avg_atlas_prob = torch.sum(atlas_tensor[0] * lesion_areas) / torch.sum(lesion_areas)
                atlas_mask = (atlas_tensor[0] > 0.01).float()
                avg_atlas_overall = torch.sum(atlas_tensor[0] * atlas_mask) / (torch.sum(atlas_mask) + 1e-8)
                atlas_guidance_score = (avg_atlas_prob / (avg_atlas_overall + 1e-8)).item()
            
            all_atlas_scores.append(atlas_guidance_score)
            
            # Zápis do statistického souboru
            with open(stats_file, 'a') as f:
                f.write(f"{i+1},{coverage_percentage:.4f},{num_components},{avg_size:.4f},{is_in_range},{atlas_guidance_score:.4f}\n")
            
            # Převod na numpy
            fake_np_raw = fake_label[0, 0].cpu().numpy()
            
            # Aplikace thresholdu pro získání binární masky
            fake_np = (fake_np_raw > min_threshold).astype(np.float32)
            
            # Výpis informací o vzorku - upravený popisek
            print(f"Vzorek {i+1}: {num_components} lézí, průměrná velikost: {avg_size:.2f}, pokrytí: {coverage_percentage:.4f}%")
            print(f"  Pokrytí v cílovém rozsahu (0.01% - 65.5% v rámci mozku): {'Ano' if is_in_range else 'Ne'}")
            print(f"  Atlas Guidance Score: {atlas_guidance_score:.4f} (>1.0 znamená léze v pravděpodobnějších oblastech)")
            
            # Uložení jako NIfTI
            fake_nii = nib.Nifti1Image(fake_np, atlas_affine)
            nib.save(fake_nii, os.path.join(output_dir, f'sample_{i+1}.nii.gz'))
            
            # Vizualizace prostředního řezu
            mid_z = fake_np.shape[2] // 2
            
            # Hledání řezu s nejvíce lézemi, pokud střední řez nemá žádné
            if np.sum(fake_np[:, :, mid_z]) == 0:
                z_sums = [np.sum(fake_np[:, :, z]) for z in range(fake_np.shape[2])]
                if max(z_sums) > 0:
                    mid_z = np.argmax(z_sums)
            
            # Uložení obrázku pro rychlou kontrolu
            plt.figure(figsize=(15, 5))
            
            # Plot 1: Generované léze
            plt.subplot(1, 3, 1)
            plt.imshow(fake_np[:, :, mid_z], cmap='gray')
            plt.title(f'Sample {i+1} - Axial Slice {mid_z}\nCoverage: {coverage_percentage:.2f}%, Lesions: {num_components}')
            plt.axis('off')
            
            # Plot 2: Atlas s překryvem generovaných lézí
            plt.subplot(1, 3, 2)
            atlas_slice = atlas_data[:, :, mid_z]
            plt.imshow(atlas_slice, cmap='viridis', alpha=0.7)
            plt.imshow(fake_np[:, :, mid_z], cmap='gray', alpha=0.5)
            plt.title(f'Atlas s lézemi\nAtlas Guidance Score: {atlas_guidance_score:.2f}')
            plt.axis('off')
            
            # Plot 3: Sagitální řez
            mid_y = fake_np.shape[1] // 2
            plt.subplot(1, 3, 3)
            plt.imshow(fake_np[:, mid_y, :], cmap='gray')
            plt.title(f'Sagittal Slice {mid_y}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'))
            plt.close()
    
    # Souhrnné statistiky - upravení rozsahu
    in_range_count = sum(1 for c in all_coverages if 0.01 <= c <= 65.5)
    avg_atlas_score = sum(all_atlas_scores) / len(all_atlas_scores) if all_atlas_scores else 0
    
    # Vytvoření histogramu pokrytí
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Histogram pokrytí
    plt.subplot(2, 1, 1)
    plt.hist(all_coverages, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=0.01, color='r', linestyle='--', label='Min Target (0.01%)')
    plt.axvline(x=65.5, color='r', linestyle='--', label='Max Target (65.5%)')
    plt.xlabel('Lesion Coverage (% of Brain Volume)')
    plt.ylabel('Number of Samples')
    plt.title(f'Lesion Coverage Distribution\n{in_range_count}/{num_samples} samples in target range (0.01% - 65.5%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Histogram atlas guidance skóre
    plt.subplot(2, 1, 2)
    plt.hist(all_atlas_scores, bins=20, color='lightgreen', edgecolor='black')
    plt.axvline(x=1.0, color='r', linestyle='--', label='Neutral (1.0)')
    plt.xlabel('Atlas Guidance Score')
    plt.ylabel('Number of Samples')
    plt.title(f'Atlas Guidance Score Distribution\nAverage Score: {avg_atlas_score:.4f} (>1.0 znamená léze v pravděpodobnějších oblastech)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistics.png'))
    plt.close()
    
    print(f"Vygenerováno {num_samples} vzorků v adresáři {output_dir}")
    print(f"Vzorky v cílovém rozsahu pokrytí (0.01% - 65.5% v rámci mozku): {in_range_count}/{num_samples} ({in_range_count/num_samples*100:.1f}%)")
    print(f"Průměrné Atlas Guidance Score: {avg_atlas_score:.4f} (>1.0 znamená léze v pravděpodobnějších oblastech)")

# Hlavní funkce
def main(args):
    try:
        # Nastavení reprodukovatelnosti
        set_seed(args.seed)
        
        # Nastavení zařízení
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Používám zařízení: {device}")
        
        if args.train:
            # Kontrola vstupních argumentů
            if not args.data_dir:
                raise ValueError("Pro trénink je nutné zadat --data_dir")
            if not args.lesion_atlas_path:
                raise ValueError("Pro trénink je nutné zadat --lesion_atlas_path")
            
            # Kontrola existence souborů
            if not os.path.exists(args.data_dir):
                raise FileNotFoundError(f"Adresář s trénovacími daty neexistuje: {args.data_dir}")
            
            # Kontrola existence atlasu lézí (bez ohledu na příponu)
            atlas_path = args.lesion_atlas_path
            if not os.path.exists(atlas_path):
                # Zkusíme alternativní přípony, pokud soubor neexistuje
                if atlas_path.endswith('.nii') and os.path.exists(atlas_path + '.gz'):
                    atlas_path = atlas_path + '.gz'
                elif atlas_path.endswith('.nii.gz') and os.path.exists(atlas_path[:-3]):
                    atlas_path = atlas_path[:-3]
                else:
                    raise FileNotFoundError(f"Soubor s atlasem lézí neexistuje: {args.lesion_atlas_path}")
            
            print(f"Načítám trénovací data z adresáře: {args.data_dir}")
            print(f"Načítám atlas lézí z: {atlas_path}")
            
            try:
                # Vytvoření datasetu a datového loaderu
                dataset = LesionDataset(
                    labels_dir=args.data_dir,
                    lesion_atlas_path=atlas_path,  # Použití ověřené cesty
                    image_size=(96, 96, 96)  # Nastavte podle velikosti vašich dat
                )
                
                if len(dataset) == 0:
                    raise ValueError("Dataset je prázdný. Ujistěte se, že adresář obsahuje validní .nii/.nii.gz soubory s lézemi.")
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=min(args.batch_size, len(dataset)),  # Zajištění, že batch_size není větší než velikost datasetu
                    shuffle=True,
                    num_workers=4,
                    drop_last=True
                )
                
                print(f"Načteno {len(dataset)} vzorků pro trénink.")
                
                # Vytvoření generátoru a diskriminátoru
                sample_data = next(iter(dataloader))
                img_size = tuple(sample_data['atlas'].shape[2:])
                print(f"Velikost dat: {img_size}")
                
                generator = SwinUNETRGenerator(img_size=img_size, feature_size=args.feature_size)
                discriminator = LesionDiscriminator()
                
                generator.to(device)
                discriminator.to(device)
                
                # Trénování modelu
                train(
                    generator=generator,
                    discriminator=discriminator,
                    dataloader=dataloader,
                    num_epochs=args.epochs,
                    device=device,
                    output_dir=args.output_dir
                )
            except Exception as e:
                print(f"Chyba při inicializaci datasetu nebo tréninku: {str(e)}")
                raise
        
        elif args.generate:
            # Kontrola vstupních argumentů pro generování
            if not args.model_path:
                raise ValueError("Pro generování vzorků je nutné zadat --model_path")
            if not args.lesion_atlas_path:
                raise ValueError("Pro generování vzorků je nutné zadat --lesion_atlas_path")
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Soubor s modelem neexistuje: {args.model_path}")
            
            # Kontrola existence atlasu lézí (bez ohledu na příponu)
            atlas_path = args.lesion_atlas_path
            if not os.path.exists(atlas_path):
                # Zkusíme alternativní přípony, pokud soubor neexistuje
                if atlas_path.endswith('.nii') and os.path.exists(atlas_path + '.gz'):
                    atlas_path = atlas_path + '.gz'
                elif atlas_path.endswith('.nii.gz') and os.path.exists(atlas_path[:-3]):
                    atlas_path = atlas_path[:-3]
                else:
                    raise FileNotFoundError(f"Soubor s atlasem lézí neexistuje: {args.lesion_atlas_path}")
            
            print(f"Načítám atlas lézí z: {atlas_path}")
            print(f"Používám feature_size: {args.feature_size}")
            
            # Generování vzorků z natrénovaného modelu
            generate_samples(
                model_path=args.model_path,
                lesion_atlas_path=atlas_path,  # Použití ověřené cesty
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                min_threshold=args.min_threshold,
                feature_size=args.feature_size  # Předání hodnoty z argumentů
            )
        
        else:
            print("Musíte zadat buď --train nebo --generate")
    
    except Exception as e:
        print(f"\nKritická chyba: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTipy pro řešení problémů:")
        print("1. Zkontrolujte, že adresář s daty existuje a obsahuje soubory .nii nebo .nii.gz")
        print("2. Ujistěte se, že soubory obsahují nějaké léze (nenulové hodnoty)")
        print("3. Zkontrolujte formát a integritu souborů")
        print("4. Vyzkoušejte menší batch_size, pokud máte problémy s pamětí")
        return 1
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinUNETR Lesion GAN")
    parser.add_argument("--train", action="store_true", help="Trénování modelu")
    parser.add_argument("--generate", action="store_true", help="Generování vzorků")
    parser.add_argument("--data_dir", type=str, help="Adresář s trénovacími daty (.nii nebo .nii.gz soubory)")
    parser.add_argument("--lesion_atlas_path", type=str, help="Cesta k atlasu lézí (.nii nebo .nii.gz)")
    parser.add_argument("--output_dir", type=str, default="output", help="Výstupní adresář")
    parser.add_argument("--model_path", type=str, help="Cesta k uloženému modelu (pro generování)")
    parser.add_argument("--epochs", type=int, default=100, help="Počet trénovacích epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="Velikost batche")
    parser.add_argument("--num_samples", type=int, default=10, help="Počet vzorků k vygenerování")
    parser.add_argument("--min_threshold", type=float, default=0.3, help="Minimální threshold pro binarizaci výstupu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed pro reprodukovatelnost")
    parser.add_argument("--feature_size", type=int, default=24, help="Základní velikost příznakových map v SwinUNETR")
    
    args = parser.parse_args()
    exit_code = main(args)
    import sys
    sys.exit(exit_code) 