import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import nibabel as nib
import os
import gc
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage

# Konstanty
IMAGE_SIZE = (128, 128, 64)
LATENT_DIM = 100
BATCH_SIZE = 1
DEFAULT_EPOCHS = 200  # Výchozí počet epoch
LAMBDA_SPARSITY = 10.0  # Váha pro sparsity loss
LAMBDA_ATLAS = 5.0      # Váha pro atlas guidance loss
LAMBDA_STRUCTURAL = 4.0  # Snížená váha pro strukturální loss - menší důraz na přesné struktury
HETEROGENEITY_LEVEL = 0.5  # Zvýšená konstanta pro úroveň heterogenity (0-1)
MULTI_SCALE_NOISE = True  # Použití víceúrovňového šumu pro realističtější heterogenitu
SHAPE_VARIATION = 0.6     # Míra variability tvaru (0-1) při generování různých vzorků
LESION_PATTERN_TYPES = 5  # Počet různých "vzorů" lézí pro generování
CONNECTIVITY_VARIATION = 0.5  # Míra variability spojitosti lézí (0-1)
COVERAGE_VARIATION = True     # Variabilita procentuálního pokrytí lézemi
DEFAULT_TARGET_COVERAGE = 0.01  # Výchozí cílové pokrytí lézemi (1%)
SAVE_INTERVAL = 10      # Jak často ukládat modely a vizualizace
GRADIENT_ACCUMULATION_STEPS = 4  # Akumulace gradientů pro simulaci větších batchů

# Dataset
class HIELesionDataset(Dataset):
    def __init__(self, labels_dir, atlas_path, transform=None):
        """
        Dataset pro HIE léze
        
        Args:
            labels_dir: Adresář s .nii soubory labelů
            atlas_path: Cesta k frekvenčnímu atlasu
            transform: Transformace pro augmentaci dat
        """
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Načtení frekvenčního atlasu - efektivnější načítání s typem float32
        atlas_nii = nib.load(atlas_path)
        self.atlas = torch.tensor(atlas_nii.get_fdata(dtype=np.float32), dtype=torch.float32)
        
        # Seznam labelů a filtrování prázdných (celočerných)
        self.label_files = []
        for file in os.listdir(labels_dir):
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(labels_dir, file)
                # Efektivnější načítání dat s explicitním typem
                label_data = nib.load(file_path).get_fdata(dtype=np.float32)
                # Kontrola, zda label není celočerný
                if np.sum(label_data) > 0:
                    self.label_files.append(file)
                # Explicitní uvolnění paměti
                del label_data
                gc.collect()
        
        print(f"Načteno {len(self.label_files)} labelů po odfiltrování celočerných")
    
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        # Efektivnější načítání
        label_nii = nib.load(label_path)
        label = torch.tensor(label_nii.get_fdata(dtype=np.float32), dtype=torch.float32)
        
        # Normalizace dat
        if self.transform:
            label = self.transform(label)
            
        return {'label': label, 'atlas': self.atlas}

# Původní verze generátoru pro zpětnou kompatibilitu
class LegacyGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(LegacyGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # Počáteční velikost (4x4x2)
        self.init_size = (4, 4, 2)
        init_channels = 512
        
        # Fully connected vrstva pro převod latentního vektoru
        self.fc = nn.Linear(latent_dim + np.prod(IMAGE_SIZE), np.prod(self.init_size) * init_channels)
        
        # 3D Transposed convolutions pro zvětšování dimenzí
        self.deconv_blocks = nn.Sequential(
            nn.BatchNorm3d(init_channels),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(init_channels, 256, 4, stride=2, padding=1),  # 8x8x4
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),  # 16x16x8
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),  # 32x32x16
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),  # 64x64x32
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),  # 128x128x64
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            
            nn.Conv3d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z, atlas):
        # Atlas jako podmínka pro generování
        atlas_flat = atlas.view(atlas.size(0), -1)
        z = torch.cat([z, atlas_flat], dim=1)
        
        out = self.fc(z)
        out = out.view(out.shape[0], 512, *self.init_size)
        out = self.deconv_blocks(out)
        
        # Aplikace frekvenčního atlasu jako masky - léze mohou být jen v nenulových oblastech
        atlas_expanded = atlas.unsqueeze(1)  # Přidání kanálového rozměru
        out = out * (atlas_expanded > 0).float()
        
        return out

# Optimalizovaný Generátor s menší spotřebou paměti a lepší strukturální schopností
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Počáteční velikost (4x4x2)
        self.init_size = (4, 4, 2)
        init_channels = 384  # Místo 512 - snížena šířka sítě
        
        # Rozdělená FC vrstva pro snížení spotřeby paměti
        # Místo jedné velké FC vrstvy použijeme dvě menší
        fc_intermediate_dim = 2048
        self.fc1 = nn.Linear(latent_dim, fc_intermediate_dim)
        self.fc2 = nn.Linear(fc_intermediate_dim, np.prod(self.init_size) * init_channels)
        
        # Atlas zpracován separátně, místo konkatenace - úspora paměti
        self.atlas_encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16 * 8, fc_intermediate_dim)
        )
        
        # 3D Transposed convolutions pro zvětšování dimenzí
        # Přidáváme residuální bloky pro lepší učení struktur
        self.deconv1 = nn.Sequential(
            nn.BatchNorm3d(init_channels),
            nn.ReLU(True),
            nn.ConvTranspose3d(init_channels, 192, 4, stride=2, padding=1)  # 8x8x4
        )
        
        self.res_block1 = ResidualBlock3D(192)
        
        self.deconv2 = nn.Sequential(
            nn.BatchNorm3d(192),
            nn.ReLU(True),
            nn.ConvTranspose3d(192, 96, 4, stride=2, padding=1)  # 16x16x8
        )
        
        self.res_block2 = ResidualBlock3D(96)
        
        self.deconv3 = nn.Sequential(
            nn.BatchNorm3d(96),
            nn.ReLU(True),
            nn.ConvTranspose3d(96, 48, 4, stride=2, padding=1)  # 32x32x16
        )
        
        self.res_block3 = ResidualBlock3D(48)
        
        self.deconv4 = nn.Sequential(
            nn.BatchNorm3d(48),
            nn.ReLU(True),
            nn.ConvTranspose3d(48, 24, 4, stride=2, padding=1)  # 64x64x32
        )
        
        self.res_block4 = ResidualBlock3D(24)
        
        self.deconv5 = nn.Sequential(
            nn.BatchNorm3d(24),
            nn.ReLU(True),
            nn.ConvTranspose3d(24, 12, 4, stride=2, padding=1)  # 128x128x64
        )
        
        self.res_block5 = ResidualBlock3D(12)
        
        # Finální konvoluce pro výstup
        self.final_conv = nn.Sequential(
            nn.Conv3d(12, 12, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv3d(12, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # Inicializace vah pro lepší start trénování
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z, atlas):
        # Efektivnější zpracování atlasu - jako 3D konvoluce místo flattening
        atlas_expanded = atlas.unsqueeze(1)  # Přidání kanálového rozměru
        atlas_features = self.atlas_encoder(atlas_expanded)
        
        # Zpracování latent vektoru
        x = F.relu(self.fc1(z))
        # Kombinace s atlas features
        x = x + atlas_features
        x = self.fc2(x)
        
        # Reshape pro dekonvoluci
        x = x.view(x.shape[0], 384, *self.init_size)
        
        # Postupné dekonvoluce s residuálními bloky
        x = self.deconv1(x)
        x = self.res_block1(x)
        
        x = self.deconv2(x)
        x = self.res_block2(x)
        
        x = self.deconv3(x)
        x = self.res_block3(x)
        
        x = self.deconv4(x)
        x = self.res_block4(x)
        
        x = self.deconv5(x)
        x = self.res_block5(x)
        
        # Finální konvoluce
        out = self.final_conv(x)
        
        # Aplikace frekvenčního atlasu jako masky - léze mohou být jen v nenulových oblastech
        out = out * (atlas_expanded > 0).float()
        
        # Přidání heterogenity během forward průchodu - pouze pro inference
        if not self.training and HETEROGENEITY_LEVEL > 0:
            out = torch.tensor(add_controlled_heterogeneity(out.cpu().numpy(), HETEROGENEITY_LEVEL), 
                               device=out.device, dtype=out.dtype)
        
        return out

# Residuální blok pro lepší učení strukturálních vlastností
class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out

# Optimalizovaný Diskriminátor s menší spotřebou paměti
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Menší model - postupné snižování počtu filtrů
        self.conv1 = nn.Conv3d(2, 12, 4, stride=2, padding=1)  # 64x64x32
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv3d(12, 24, 4, stride=2, padding=1)  # 32x32x16
        self.bn2 = nn.BatchNorm3d(24)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv3d(24, 48, 4, stride=2, padding=1)  # 16x16x8
        self.bn3 = nn.BatchNorm3d(48)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv3d(48, 96, 4, stride=2, padding=1)  # 8x8x4
        self.bn4 = nn.BatchNorm3d(96)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv3d(96, 192, 4, stride=2, padding=1)  # 4x4x2
        self.bn5 = nn.BatchNorm3d(192)
        self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)
        
        # Přidáváme průchod pro strukturální analýzu
        self.structure_conv = nn.Conv3d(192, 192, 3, stride=1, padding=1)
        self.structure_bn = nn.BatchNorm3d(192)
        self.structure_lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        # Výstupní vrstva bez sigmoidu (používáme BCEWithLogitsLoss)
        self.output_layer = nn.Linear(192 * 4 * 4 * 2, 1)
        
        # Inicializace vah pro lepší start trénování
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, img, atlas):
        # Spojení vstupu a atlasu jako kanály
        atlas_expanded = atlas.unsqueeze(1)  # Přidání kanálového rozměru
        x = torch.cat([img, atlas_expanded], dim=1)
        
        # Konvoluční vrstvy s extrakcí feature map pro strukturální loss
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        x = self.lrelu5(self.bn5(self.conv5(x)))
        
        # Strukturální features
        structure_features = self.structure_lrelu(self.structure_bn(self.structure_conv(x)))
        
        # Výstup
        features = x.view(x.size(0), -1)
        validity = self.output_layer(features)
        
        return validity, structure_features

# Custom loss funkce - optimalizované
def atlas_guided_loss(generated_images, atlas):
    """
    Loss funkce, která povzbuzuje generování v souladu s frekvenčním atlasem
    """
    # Přizpůsobení generování podle frekvencí v atlasu
    atlas_expanded = atlas.unsqueeze(1)
    # Penalizace generování v oblastech s nízkou frekvencí
    loss = -torch.mean(torch.log(1e-8 + generated_images) * atlas_expanded)
    return loss

def sparsity_loss(generated_images, target_sparsity=0.01):
    """
    Loss funkce pro řídkost lézí (0.01% - 2.5%)
    """
    # Výpočet aktuální sparsity
    current_sparsity = torch.mean(generated_images)
    # Penalizace když je sparsity mimo požadovaný rozsah (0.0001 - 0.025)
    loss = torch.abs(current_sparsity - target_sparsity)
    return loss

def structural_similarity_loss(generated, real):
    """
    Loss funkce pro strukturální podobnost generovaných a skutečných lézí
    Používá feature mapy z diskriminátoru pro hodnocení strukturální podobnosti
    """
    # L1 distance mezi feature mapami
    return F.l1_loss(generated, real)

def gradient_penalty_loss(real_images, fake_images, discriminator, atlas, device):
    """
    Gradient penalty pro Wasserstein GAN s gradient penalty (WGAN-GP)
    Pomáhá stabilizovat trénink
    """
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, 1).to(device)
    
    # Generování interpolovaných vzorků
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated = interpolated.requires_grad_(True)
    
    # Forward pass diskriminátoru
    d_interpolated, _ = discriminator(interpolated, atlas)
    
    # Výpočet gradientů
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    
    # Výpočet gradient penalty
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

# Přidání nové funkce pro vytvoření heterogenity v existujících lézích
def add_controlled_heterogeneity(input_volume, heterogeneity_level=HETEROGENEITY_LEVEL, multi_scale=MULTI_SCALE_NOISE):
    """
    Přidá kontrolovanou heterogenitu do vstupního objemu (během generování)
    s možností víceúrovňového šumu pro realističtější textury
    
    Args:
        input_volume: Tensor s generovanými lézemi
        heterogeneity_level: Úroveň heterogenity (0-1)
        multi_scale: Použít víceúrovňový šum pro realističtější textury
        
    Returns:
        Tensor s heterogenními lézemi
    """
    if heterogeneity_level <= 0:
        return input_volume
    
    # Převedení na numpy pro snazší manipulaci
    is_tensor = torch.is_tensor(input_volume)
    if is_tensor:
        device = input_volume.device
        input_np = input_volume.cpu().numpy()
    else:
        input_np = input_volume
    
    # Aplikace pouze na hodnoty > 0 (oblasti léze)
    binary_mask = input_np > 0.1
    
    if multi_scale:
        # Vytvoření víceúrovňového šumu pro realističtější textury
        # Kombinace různých prostorových frekvencí
        noise_fine = np.random.normal(0, heterogeneity_level, input_np.shape)
        noise_medium = np.random.normal(0, heterogeneity_level, input_np.shape)
        noise_coarse = np.random.normal(0, heterogeneity_level, input_np.shape)
        
        # Rozdílné vyhlazení pro různé frekvence
        noise_fine = ndimage.gaussian_filter(noise_fine, sigma=0.5)
        noise_medium = ndimage.gaussian_filter(noise_medium, sigma=1.5)
        noise_coarse = ndimage.gaussian_filter(noise_coarse, sigma=3.0)
        
        # Kombinace šumů s různými váhami pro různé textury
        smooth_noise = noise_fine * 0.6 + noise_medium * 0.3 + noise_coarse * 0.1
    else:
        # Původní implementace s jednou úrovní šumu
        noise = np.random.normal(0, heterogeneity_level, input_np.shape)
        smooth_noise = ndimage.gaussian_filter(noise, sigma=0.7)
    
    # Aplikace šumu pouze na oblasti léze, posílení vnitřní heterogenity
    heterogeneous = input_np.copy()
    # Zvýšení faktoru vlivu šumu na hodnoty pro větší vnitřní kontrast
    heterogeneous[binary_mask] = input_np[binary_mask] * (1.0 + smooth_noise[binary_mask] * 0.7)
    
    # Přidání "fluktuací hustoty" uvnitř léze pro realističtější vzhled
    if np.sum(binary_mask) > 100:  # Jen pro dostatečně velké léze
        # Vytvoření náhodných "prasklin" nebo kanálků uvnitř léze
        fracture_mask = np.random.random(input_np.shape) < 0.05
        fracture_mask = fracture_mask & binary_mask
        if np.sum(fracture_mask) > 0:
            # Snížení hodnot v místech "prasklin"
            heterogeneous[fracture_mask] *= 0.3
    
    # Normalizace hodnot zpět do rozsahu 0-1
    heterogeneous = np.clip(heterogeneous, 0, 1)
    
    # Vrácení zpět jako tensor, pokud byl vstup tensor
    if is_tensor:
        return torch.tensor(heterogeneous, device=device, dtype=input_volume.dtype)
    return heterogeneous

# Kontrola a hlášení dostupné CUDA paměti
def report_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        free_memory = total_memory - reserved_memory
        print(f"GPU Memory: Total={total_memory:.2f}GB, Reserved={reserved_memory:.2f}GB, "
              f"Allocated={allocated_memory:.2f}GB, Free={free_memory:.2f}GB")
        return free_memory
    return 0

# Hlavní trénovací funkce
def train(labels_dir, atlas_path, output_dir, epochs=DEFAULT_EPOCHS, 
          save_interval=SAVE_INTERVAL, device='cuda'):
    # Vytvoření output adresáře
    os.makedirs(output_dir, exist_ok=True)
    
    # Reportování dostupné paměti
    if device == 'cuda':
        free_memory = report_gpu_memory()
        print(f"Trénink začíná s {free_memory:.2f}GB volné GPU paměti")
    
    # Inicializace datasetů a dataloaderů - omezený počet workers pro menší spotřebu paměti
    dataset = HIELesionDataset(labels_dir, atlas_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                           pin_memory=(device=='cuda'), persistent_workers=True)
    
    # Inicializace modelů
    generator = Generator(LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizery s nižší learning rate pro stabilnější trénink
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # Loss funkce - změna na BCEWithLogitsLoss, který je bezpečný pro autocast
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    # Zapnutí automatického mixed precision (FP16) pro úsporu paměti na podporovaných GPU
    scaler = torch.amp.GradScaler() if device == 'cuda' else None
    
    # Trénovací smyčka
    for epoch in range(epochs):
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
            
            # Resetování akumulovaných gradientů
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            
            for i, batch in enumerate(tepoch):
                # Přesun dat na device s explicitním uvolněním paměti
                labels = batch['label'].to(device, non_blocking=True)
                atlas = batch['atlas'].to(device, non_blocking=True)
                
                # Reálné a falešné labely pro discriminator
                real_labels = torch.ones(labels.size(0), 1).to(device)
                fake_labels = torch.zeros(labels.size(0), 1).to(device)
                
                # -----------------
                # Trénink discriminatoru
                # -----------------
                if scaler:
                    with torch.amp.autocast(device_type='cuda'):
                        # Diskriminátor na reálných datech
                        labels_unsqueeze = labels.unsqueeze(1)  # Přidání kanálového rozměru
                        real_validity, real_features = discriminator(labels_unsqueeze, atlas)
                        d_real_loss = adversarial_loss(real_validity, real_labels)
                        
                        # Diskriminátor na generovaných datech
                        z = torch.randn(labels.size(0), LATENT_DIM).to(device)
                        fake_images = generator(z, atlas).detach()
                        fake_validity, fake_features = discriminator(fake_images, atlas)
                        d_fake_loss = adversarial_loss(fake_validity, fake_labels)
                        
                        # Celková loss discriminatoru
                        d_loss = (d_real_loss + d_fake_loss) / 2
                        d_loss = d_loss / GRADIENT_ACCUMULATION_STEPS  # Normalizace pro accumulation
                    
                    # Škálování gradientu a zpětná propagace
                    scaler.scale(d_loss).backward()
                    
                    # Aplikace akumulovaných gradientů po několika krocích
                    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.step(optimizer_D)
                        optimizer_D.zero_grad()
                else:
                    # Standardní trénink bez mixed precision
                    labels_unsqueeze = labels.unsqueeze(1)
                    real_validity, real_features = discriminator(labels_unsqueeze, atlas)
                    d_real_loss = adversarial_loss(real_validity, real_labels)
                    
                    z = torch.randn(labels.size(0), LATENT_DIM).to(device)
                    fake_images = generator(z, atlas).detach()
                    fake_validity, fake_features = discriminator(fake_images, atlas)
                    d_fake_loss = adversarial_loss(fake_validity, fake_labels)
                    
                    d_loss = (d_real_loss + d_fake_loss) / 2
                    d_loss = d_loss / GRADIENT_ACCUMULATION_STEPS  # Normalizace pro accumulation
                    d_loss.backward()
                    
                    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        optimizer_D.step()
                        optimizer_D.zero_grad()
                
                # -----------------
                # Trénink generátoru
                # -----------------
                if scaler:
                    with torch.amp.autocast(device_type='cuda'):
                        # Generátor se snaží oklamat diskriminátor
                        z = torch.randn(labels.size(0), LATENT_DIM).to(device)
                        fake_images = generator(z, atlas)
                        fake_validity, fake_features = discriminator(fake_images, atlas)
                        g_adversarial_loss = adversarial_loss(fake_validity, real_labels)
                        
                        # Atlas-guided loss
                        g_atlas_loss = atlas_guided_loss(fake_images, atlas)
                        
                        # Sparsity loss
                        g_sparsity_loss = sparsity_loss(fake_images, target_sparsity=0.01)
                        
                        # Strukturální loss - porovnání feature map z diskriminátoru
                        _, real_features = discriminator(labels_unsqueeze, atlas)
                        g_structural_loss = structural_similarity_loss(fake_features, real_features)
                        
                        # Celková loss generátoru
                        g_loss = (g_adversarial_loss 
                                 + LAMBDA_ATLAS * g_atlas_loss 
                                 + LAMBDA_SPARSITY * g_sparsity_loss
                                 + LAMBDA_STRUCTURAL * g_structural_loss)
                        g_loss = g_loss / GRADIENT_ACCUMULATION_STEPS  # Normalizace pro accumulation
                    
                    # Škálování gradientu a zpětná propagace
                    scaler.scale(g_loss).backward()
                    
                    # Aplikace akumulovaných gradientů po několika krocích
                    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.step(optimizer_G)
                        scaler.update()
                        optimizer_G.zero_grad()
                else:
                    # Standardní trénink bez mixed precision
                    z = torch.randn(labels.size(0), LATENT_DIM).to(device)
                    fake_images = generator(z, atlas)
                    fake_validity, fake_features = discriminator(fake_images, atlas)
                    g_adversarial_loss = adversarial_loss(fake_validity, real_labels)
                    
                    g_atlas_loss = atlas_guided_loss(fake_images, atlas)
                    g_sparsity_loss = sparsity_loss(fake_images, target_sparsity=0.01)
                    
                    # Strukturální loss
                    _, real_features = discriminator(labels_unsqueeze, atlas)
                    g_structural_loss = structural_similarity_loss(fake_features, real_features)
                    
                    # Celková loss
                    g_loss = (g_adversarial_loss 
                             + LAMBDA_ATLAS * g_atlas_loss 
                             + LAMBDA_SPARSITY * g_sparsity_loss
                             + LAMBDA_STRUCTURAL * g_structural_loss)
                    g_loss = g_loss / GRADIENT_ACCUMULATION_STEPS
                    g_loss.backward()
                    
                    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        optimizer_G.step()
                        optimizer_G.zero_grad()
                
                # Aktualizace progress baru
                sparsity_value = torch.mean(fake_images).item()
                tepoch.set_postfix(D_loss=d_loss.item() * GRADIENT_ACCUMULATION_STEPS, 
                                  G_loss=g_loss.item() * GRADIENT_ACCUMULATION_STEPS, 
                                  Sparsity=sparsity_value,
                                  Struct_loss=g_structural_loss.item())
                
                # Explicitní uvolnění paměti pro proměnné, které už nepotřebujeme
                del labels, atlas, fake_images, real_validity, fake_validity, real_features, fake_features
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Uložení modelů podle intervalu ukládání
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
            }, os.path.join(output_dir, f'generator_{epoch+1}.pt'))
            
            torch.save({
                'epoch': epoch + 1,
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, os.path.join(output_dir, f'discriminator_{epoch+1}.pt'))
            
            # Vygenerování vzorků pro vizualizaci
            with torch.no_grad():
                z = torch.randn(1, LATENT_DIM).to(device)
                atlas_sample = dataset[0]['atlas'].unsqueeze(0).to(device)
                with torch.amp.autocast(device_type='cuda', enabled=(device=='cuda')):
                    sample = generator(z, atlas_sample).cpu().numpy()
                
                # Post-processing pro zlepšení struktury
                processed_sample = post_process_lesions(sample[0, 0])
                
                # Uložení vzorků jako nii file
                sample_img = nib.Nifti1Image(processed_sample, np.eye(4))
                nib.save(sample_img, os.path.join(output_dir, f'sample_epoch_{epoch+1}.nii.gz'))
                
                # Vizualizace středového řezu
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(processed_sample[:, :, processed_sample.shape[2]//2], cmap='gray')
                plt.title(f'Generated - Epoch {epoch+1}')
                plt.subplot(1, 2, 2)
                plt.imshow(atlas_sample[0, :, :, atlas_sample.shape[3]//2].cpu(), cmap='hot')
                plt.title('Atlas')
                plt.savefig(os.path.join(output_dir, f'sample_visual_epoch_{epoch+1}.png'))
                plt.close()
                
                # Explicitní uvolnění paměti
                del z, atlas_sample, sample, processed_sample
                if device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Reportování dostupné paměti po uložení
            if device == 'cuda':
                report_gpu_memory()

# Nová funkce pro analýzu jednoho konkrétního trénovacího souboru
def analyze_single_file(file_path):
    """
    Analyzuje jeden trénovací soubor pro extrakci morfologických vlastností
    
    Args:
        file_path: Cesta k souboru s lézemi
        
    Returns:
        Dictionary s morfologickými vlastnostmi
    """
    try:
        # Načtení dat
        label_data = nib.load(file_path).get_fdata(dtype=np.float32)
        
        # Přeskočíme prázdné soubory
        if np.sum(label_data) == 0:
            return None
        
        # Morfologické vlastnosti
        properties = {
            "volumes": [],
            "compactness": [],
            "connectivities": [],
            "elongations": [],
            "coverage": 0.0
        }
        
        # Výpočet celkového pokrytí
        total_voxels = np.prod(label_data.shape)
        lesion_voxels = np.sum(label_data > 0.5)
        properties["coverage"] = lesion_voxels / total_voxels
        
        # Binarizace dat
        binary = (label_data > 0.5).astype(np.int32)
        
        # Extrakce jednotlivých komponent
        labeled, num_features = ndimage.label(binary)
        
        # Analýza každé komponenty (léze)
        for i in range(1, num_features + 1):
            component = (labeled == i)
            
            # Objem
            volume = np.sum(component)
            properties["volumes"].append(volume)
            
            # Pouze pokračovat pro dostatečně velké léze
            if volume < 20:
                continue
                
            # Kompaktnost (kulatost)
            eroded = ndimage.binary_erosion(component)
            surface = np.sum(component) - np.sum(eroded)
            if surface > 0:
                comp = (6 * np.sqrt(np.pi) * volume) / (surface ** 1.5)
                properties["compactness"].append(min(comp, 1.0))
            
            # Spojitost
            eroded_multiple = ndimage.binary_erosion(component, iterations=2)
            erosion_ratio = np.sum(eroded_multiple) / volume if volume > 0 else 0
            properties["connectivities"].append(erosion_ratio)
            
            # Elongace
            if volume > 50:
                try:
                    coords = np.array(np.where(component)).T
                    if len(coords) > 3:
                        centered_coords = coords - np.mean(coords, axis=0)
                        cov = np.cov(centered_coords, rowvar=False)
                        eigenvalues = np.linalg.eigvalsh(cov)
                        if eigenvalues[0] > 0:
                            elong = np.sqrt(eigenvalues[2] / eigenvalues[0])
                            properties["elongations"].append(min(elong, 10.0))
                except:
                    pass
        
        # Průměrné hodnoty
        properties["avg_connectivity"] = np.mean(properties["connectivities"]) if properties["connectivities"] else 0.5
        properties["avg_compactness"] = np.mean(properties["compactness"]) if properties["compactness"] else 0.5
        properties["avg_elongation"] = np.mean(properties["elongations"]) if properties["elongations"] else 1.0
        
        # Počet lézí
        properties["num_lesions"] = num_features
        
        return properties
        
    except Exception as e:
        print(f"Chyba při analýze souboru {file_path}: {str(e)}")
        return None

# Funkce pro generování nových lézí pomocí natrénovaného modelu - upravená verze
def generate_samples(generator_path, atlas_path, output_dir, num_samples=10, 
                   heterogeneity=HETEROGENEITY_LEVEL, shape_variation=SHAPE_VARIATION,
                   connectivity_variation=CONNECTIVITY_VARIATION, labels_dir=None, device='cuda'):
    # Vytvoření output adresáře
    os.makedirs(output_dir, exist_ok=True)
    
    # Reportování dostupné paměti
    if device == 'cuda':
        free_memory = report_gpu_memory()
        print(f"Generování začíná s {free_memory:.2f}GB volné GPU paměti")
    
    # Získání seznamu trénovacích souborů, pokud je cesta poskytnuta
    training_files = []
    file_properties = []
    
    if labels_dir and os.path.exists(labels_dir):
        print("Hledání vhodných trénovacích souborů...")
        for file in os.listdir(labels_dir):
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(labels_dir, file)
                
                # Rychlá kontrola, zda soubor není prázdný
                try:
                    img = nib.load(file_path)
                    if np.sum(img.get_fdata(dtype=np.float32)) > 0:
                        training_files.append(file_path)
                except:
                    pass
        
        print(f"Nalezeno {len(training_files)} vhodných trénovacích souborů")
    
    # Načtení atlasu - efektivnější načítání
    atlas_nii = nib.load(atlas_path)
    atlas = torch.tensor(atlas_nii.get_fdata(dtype=np.float32), dtype=torch.float32).unsqueeze(0).to(device)
    
    # Kontrola typu checkpointu
    try:
        # Načítáme s weights_only=True pro bezpečnost
        checkpoint = torch.load(generator_path, map_location=device, weights_only=True)
        print(f"Checkpoint načten z: {generator_path}")
        
        # Zjistíme, jestli je to nový nebo starý formát checkpointu
        is_new_format = False
        has_discriminator = False
        
        if isinstance(checkpoint, dict):
            # Zjistíme, jestli je to dict se state_dict nebo celým modelem
            if 'generator_state_dict' in checkpoint:
                is_new_format = True
                print("Detekován nový formát checkpointu (s generator_state_dict)")
            elif 'discriminator_state_dict' in checkpoint:
                has_discriminator = True
                print("Detekován checkpoint discriminatoru - nelze použít pro generátor")
            else:
                # Pokusíme se zjistit formát z klíčů
                keys = list(checkpoint.keys())
                if keys and keys[0].startswith('deconv_blocks'):
                    print("Detekován starý formát generátoru")
                elif keys and (keys[0].startswith('fc1') or keys[0].startswith('deconv1')):
                    is_new_format = True
                    print("Detekován nový formát generátoru")
        
        # Rozhodneme, jaký generátor vytvořit na základě detekce formátu
        if has_discriminator:
            print("CHYBA: Načten checkpoint discriminatoru místo generátoru.")
            print("Potřebujete zadat správnou cestu k checkpointu generátoru.")
            return
        
        # Vybereme správný typ generátoru
        if is_new_format:
            print("Použití nového generátoru s residuálními bloky")
            generator = Generator(LATENT_DIM).to(device)
        else:
            print("Použití kompatibilního generátoru pro starý formát")
            generator = LegacyGenerator(LATENT_DIM).to(device)
        
        # Načtení vah s ošetřením všech možných formátů
        try:
            if is_new_format and 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
                print("Úspěšně načteny váhy z 'generator_state_dict'")
            elif is_new_format:
                generator.load_state_dict(checkpoint)
                print("Úspěšně načteny váhy z nového formátu")
            else:
                # Pro starý formát
                generator.load_state_dict(checkpoint)
                print("Úspěšně načteny váhy ze starého formátu")
        except Exception as e:
            print(f"Chyba při načítání vah: {str(e)}")
            print("\nPokud jste právě aktualizovali kód a používáte starý model, je potřeba:")
            print("1. Buď natrénovat nový model s aktualizovanou architekturou")
            print("2. Nebo použít starší verzi kódu pro generování\n")
            return
        
        generator.eval()
        
        # Generování vzorků
        all_samples = []
        
        print(f"Generování {num_samples} vzorků...")
        for i in range(num_samples):
            # Pro každý vzorek vybereme náhodný trénovací soubor
            target_properties = None
            reference_file = None
            
            if training_files:
                # Náhodný výběr trénovacího souboru
                reference_file = training_files[np.random.randint(0, len(training_files))]
                print(f"Vzorek {i+1}: Používám referenční soubor: {os.path.basename(reference_file)}")
                
                # Analýza vlastností vybraného souboru
                target_properties = analyze_single_file(reference_file)
                
                if target_properties:
                    print(f"  - Pokrytí: {target_properties['coverage']*100:.3f}%")
                    print(f"  - Počet lézí: {target_properties['num_lesions']}")
                    if 'avg_connectivity' in target_properties:
                        print(f"  - Průměrná spojitost: {target_properties['avg_connectivity']:.2f}")
            
            # Pokud nebyl poskytnut trénovací adresář nebo analýza selhala, použijeme výchozí vlastnosti
            if target_properties is None:
                target_properties = {
                    "coverage": DEFAULT_TARGET_COVERAGE,
                    "avg_connectivity": connectivity_variation
                }
            
            # Generování vzorku
            # Náhodný latentní vektor
            z = torch.randn(1, LATENT_DIM).to(device)
            
            # Přidání náhodné perturbace pro jedinečnost
            pattern_weights = np.zeros(LESION_PATTERN_TYPES)
            dominant_pattern = np.random.randint(0, LESION_PATTERN_TYPES)
            pattern_weights[dominant_pattern] = 0.6
            
            for j in range(LESION_PATTERN_TYPES):
                if j != dominant_pattern:
                    pattern_weights[j] = np.random.random() * 0.4
            
            pattern_weights /= np.sum(pattern_weights)
            
            # Modulace latentního vektoru
            section_size = LATENT_DIM // LESION_PATTERN_TYPES
            for j in range(LESION_PATTERN_TYPES):
                start_idx = j * section_size
                end_idx = start_idx + section_size if j < LESION_PATTERN_TYPES - 1 else LATENT_DIM
                z[0, start_idx:end_idx] *= (1.0 + pattern_weights[j])
            
            # Generování
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', enabled=(device=='cuda')):
                    sample = generator(z, atlas)
                sample_cpu = sample.cpu().numpy()
                
                # Nastavení parametrů na základě cílových vlastností
                target_coverage = target_properties["coverage"]
                target_connectivity = target_properties.get("avg_connectivity", connectivity_variation)
                
                # Modulace heterogenity a variability tvaru
                use_heterogeneity = heterogeneity * (0.8 + 0.4 * np.random.random())  # ±20% variace
                use_shape_var = shape_variation * (0.8 + 0.4 * np.random.random())    # ±20% variace
                
                # Post-processing
                processed_sample = post_process_lesions(
                    sample_cpu[0, 0],
                    heterogeneity=use_heterogeneity,
                    shape_variation=use_shape_var,
                    connectivity_variation=target_connectivity,
                    target_coverage=target_coverage,
                    multi_scale=MULTI_SCALE_NOISE
                )
                
                # Výpočet aktuálního pokrytí
                actual_coverage = np.sum(processed_sample > 0) / np.prod(processed_sample.shape)
                
                # Přidání vzorku do výsledků
                all_samples.append({
                    'data': processed_sample,
                    'heterogeneity': use_heterogeneity,
                    'shape_variation': use_shape_var,
                    'connectivity': target_connectivity,
                    'target_coverage': target_coverage,
                    'actual_coverage': actual_coverage,
                    'reference_file': os.path.basename(reference_file) if reference_file else None
                })
        
        # Uložení a zobrazení vzorků
        for i, sample_info in enumerate(all_samples):
            processed_sample = sample_info['data']
            het_level = sample_info['heterogeneity']
            shape_var = sample_info['shape_variation']
            connect = sample_info['connectivity']
            coverage = sample_info['actual_coverage'] * 100  # Pro lepší čitelnost v procentech
            ref_file = sample_info['reference_file']
            
            # Uložení výsledku
            sample_img = nib.Nifti1Image(processed_sample, atlas_nii.affine)
            nib.save(sample_img, os.path.join(output_dir, f'generated_sample_{i+1}.nii.gz'))
            
            # Vizualizace středového řezu
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(processed_sample[:, :, processed_sample.shape[2]//2], cmap='gray')
            plt.title(f'Sample {i+1}: Het={het_level:.2f}, Connect={connect:.2f}, Coverage={coverage:.3f}%')
            plt.subplot(1, 2, 2)
            plt.imshow(atlas[0, :, :, atlas.shape[3]//2].cpu(), cmap='hot')
            plt.title('Atlas')
            plt.savefig(os.path.join(output_dir, f'generated_sample_{i+1}.png'))
            plt.close()
            
            print(f"Vygenerován vzorek {i+1}: Sparsity = {np.mean(processed_sample):.6f}, " +
                 f"Het = {het_level:.2f}, Connect = {connect:.2f}, Coverage = {coverage:.3f}%, " +
                 f"Vzor podle: {ref_file if ref_file else 'výchozí'}")
            
            # Explicitní uvolnění paměti
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Chyba při generování vzorků: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTipy pro řešení problémů:")
        print("1. Zkontrolujte cestu k souboru generátoru")
        print("2. Ujistěte se, že používáte správný formát checkpointu") 
        print("3. Zkuste natrénovat nový model s aktuální verzí kódu")

# Přidání funkce post_process_lesions, která byla odstraněna
def post_process_lesions(generated_sample, threshold=0.5, min_size=10, heterogeneity=HETEROGENEITY_LEVEL, 
                        shape_variation=SHAPE_VARIATION, connectivity_variation=CONNECTIVITY_VARIATION,
                        target_coverage=DEFAULT_TARGET_COVERAGE, multi_scale=MULTI_SCALE_NOISE):
    """
    Post-processing pro vylepšení struktury generovaných lézí s kontrolovanou heterogenitou,
    variabilitou tvarů mezi vzorky a nastavitelnou spojitostí/fragmentací lézí
    
    Args:
        generated_sample: Generovaný vzorek (numpy array)
        threshold: Hodnota pro binarizaci
        min_size: Minimální velikost komponenty pro zachování
        heterogeneity: Úroveň požadované heterogenity (0-1), kde 0 je hladká a 1 je velmi heterogenní
        shape_variation: Míra variability tvaru (0-1) při generování různých vzorků
        connectivity_variation: Míra spojitosti lézí (0-1), kde 0 je více fragmentované, 1 je více spojité
        target_coverage: Cílové pokrytí lézemi (procento objemu)
        multi_scale: Použít víceúrovňový šum
        
    Returns:
        Vylepšený binární obraz s heterogenní strukturou
    """
    # Binarizace s nižším prahem pro zachování více detailů
    # Adaptivně přizpůsobit práh podle variace tvaru
    adaptive_threshold = threshold - (shape_variation * 0.2)
    binary = (generated_sample > adaptive_threshold).astype(np.int32)
    
    # Odstranění malých komponent
    labeled, num_features = ndimage.label(binary)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes > min_size
    mask_sizes[0] = 0  # Ignorování pozadí
    binary = mask_sizes[labeled]
    
    # Aplikace morfologických operací podle požadované spojitosti
    # Pro vysokou spojitost použijeme closing pro vyplnění děr a spojení blízkých lézí
    # Pro nízkou spojitost použijeme opening pro fragmentaci
    if connectivity_variation > 0.7:  # Vysoce spojité léze
        # Silnější closing pro spojování blízkých oblastí
        struct_size = 3
        struct_element = np.ones((struct_size, struct_size, struct_size))
        binary = ndimage.binary_closing(binary, structure=struct_element, iterations=2).astype(np.float32)
        # Případně přidáme dilataci pro další spojení
        if connectivity_variation > 0.9:
            binary = ndimage.binary_dilation(binary, structure=np.ones((2,2,2))).astype(np.float32)
            binary = ndimage.binary_closing(binary, structure=struct_element).astype(np.float32)
    elif connectivity_variation > 0.4:  # Střední spojitost
        # Standardní closing
        struct_size = max(1, int(3 - shape_variation * 2))
        struct_element = np.ones((struct_size, struct_size, struct_size))
        binary = ndimage.binary_closing(binary, structure=struct_element).astype(np.float32)
    else:  # Nízká spojitost - více fragmentů
        # Použijeme opening pro vytvoření mezer
        binary = ndimage.binary_closing(binary, structure=np.ones((2,2,2))).astype(np.float32)
        if connectivity_variation < 0.2:
            # Pro velmi nízkou spojitost použijeme erozi následovanou dilatací s menším kernelem
            binary = ndimage.binary_erosion(binary, structure=np.ones((2,2,2))).astype(np.float32)
            binary = ndimage.binary_dilation(binary, structure=np.ones((1,1,1))).astype(np.float32)
    
    # Přidání kontrolované heterogenity uvnitř lézí
    if heterogeneity > 0:
        # Pokud je povolena heterogenita
        
        # Vytvoříme víceúrovňový šum pro realističtější textury
        if multi_scale:
            # Jemnější šum
            noise_fine = np.random.normal(0, heterogeneity, binary.shape)
            noise_fine = ndimage.gaussian_filter(noise_fine, sigma=0.5)
            
            # Středně hrubý šum
            noise_medium = np.random.normal(0, heterogeneity, binary.shape)
            noise_medium = ndimage.gaussian_filter(noise_medium, sigma=1.5)
            
            # Hrubší šum pro větší struktury
            noise_coarse = np.random.normal(0, heterogeneity, binary.shape)
            noise_coarse = ndimage.gaussian_filter(noise_coarse, sigma=3.0)
            
            # Kombinace šumů s různými váhami
            # Vyšší váha pro jemnější šum vytvoří více detailů
            noise = noise_fine * 0.6 + noise_medium * 0.3 + noise_coarse * 0.1
        else:
            # Jednoduchý náhodný šum
            noise = np.random.normal(0, heterogeneity, binary.shape)
            noise = ndimage.gaussian_filter(noise, sigma=0.7) 
        
        # Aplikujeme šum pouze na oblasti s lézemi
        binary_float = binary.astype(float)
        
        # Kombinace původního obrazu a šumu pouze v oblastech léze
        # Vyšší váha šumu pro výraznější heterogenitu
        heterogeneous = binary_float + (noise * binary_float * 0.8)
        
        # Normalizace hodnot mezi 0.5-1 pro léze (místo 0-1) pro zachování celkové struktury
        heterogeneous = np.clip(heterogeneous, 0, 1)
        
        # V oblastech léze zajistíme minimální hodnotu 0.4
        heterogeneous = np.where(binary > 0, 0.4 + heterogeneous * 0.6, 0)
        
        # Přidání náhodných "prasklin" nebo nižších hodnot v některých oblastech
        # Upraveno podle connectivity_variation - fragmentovanější léze mají více prasklin
        crack_probability = 0.08 * (1.0 - connectivity_variation)
        if np.random.random() < 0.7:  # 70% šance na přidání prasklin
            # Vytvoření masky pro oblasti nižší hustoty
            crack_mask = np.random.random(binary.shape) < crack_probability
            crack_mask = crack_mask & (binary > 0)
            heterogeneous[crack_mask] *= 0.4  # Snížení hodnoty v těchto oblastech
            
        # Vyhladíme vznikající oblasti, ale méně agresivně
        smoothing_sigma = 0.3 + (connectivity_variation * 0.2)  # Více spojité léze jsou více vyhlazené
        heterogeneous = ndimage.gaussian_filter(heterogeneous, sigma=smoothing_sigma)
        
        # Opětovná binarizace s adaptivním prahem, abychom zachovali přibližně stejnou velikost
        vol_before = np.sum(binary)
        
        # Variabilní práh dle požadované heterogenity
        binary_het_threshold = 0.3 - (heterogeneity * 0.1)
        binary_het = heterogeneous > binary_het_threshold
        
        # Úprava pokrytí, pokud je cílová hodnota specifikována
        current_coverage = np.sum(binary_het) / np.prod(binary_het.shape)
        
        # Pokud se aktuální pokrytí výrazně liší od cílového, upravíme práh
        if abs(current_coverage - target_coverage) > 0.002:  # Povolená 0.2% odchylka
            # Zkusíme najít práh, který přiblíží pokrytí k cílovému
            best_threshold = binary_het_threshold
            best_diff = abs(current_coverage - target_coverage)
            
            for test_thresh in np.linspace(0.1, 0.5, 20):
                test_binary = heterogeneous > test_thresh
                test_coverage = np.sum(test_binary) / np.prod(test_binary.shape)
                diff = abs(test_coverage - target_coverage)
                
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = test_thresh
            
            # Aplikace nejlepšího prahu
            binary_het = heterogeneous > best_threshold
        
        # Opětovné odstranění malých izolovaných komponent, které mohly vzniknout
        labeled, _ = ndimage.label(binary_het)
        sizes = np.bincount(labeled.ravel())
        mask_sizes = sizes > min_size
        mask_sizes[0] = 0
        binary_het = mask_sizes[labeled]
        
        # Vrácení heterogenní léze
        if np.random.random() < 0.3:  # 30% šance na zachování reálných hodnot místo binární masky
            # Vrácení přímo hodnot heterogenity pro realističtější vzhled
            # Ale jen tam, kde je binární maska
            result = np.zeros_like(heterogeneous)
            result[binary_het] = heterogeneous[binary_het]
            return result
        else:
            # Vrácení binární masky
            return binary_het.astype(np.float32)
    else:
        # Při nulové heterogenitě použijeme původní postup
        binary = ndimage.gaussian_filter(binary.astype(float), sigma=0.5)
        binary = (binary > 0.5).astype(np.float32)
        return binary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GAN model pro generování HIE lézí')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'], 
                        help='režim operace: trénink nebo generování')
    parser.add_argument('--labels_dir', type=str, help='adresář s labelovými daty')
    parser.add_argument('--atlas_path', type=str, help='cesta k frekvenčnímu atlasu')
    parser.add_argument('--output_dir', type=str, help='adresář pro výstupy')
    parser.add_argument('--generator_path', type=str, help='cesta k natrénovanému generátoru (pro generování)')
    parser.add_argument('--num_samples', type=int, default=10, help='počet vzorků k vygenerování')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='počet trénovacích epoch')
    parser.add_argument('--save_interval', type=int, default=SAVE_INTERVAL, help='interval pro ukládání modelů')
    parser.add_argument('--gpu', action='store_true', help='použít GPU')
    parser.add_argument('--heterogeneity', type=float, default=HETEROGENEITY_LEVEL, 
                        help='úroveň heterogenity generovaných lézí (0-1)')
    parser.add_argument('--shape_variation', type=float, default=SHAPE_VARIATION,
                        help='variabilita tvaru mezi generovanými vzorky (0-1)')
    parser.add_argument('--connectivity', type=float, default=CONNECTIVITY_VARIATION,
                        help='míra spojitosti lézí (0-1), kde 0 je více fragmentované, 1 je více spojité')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    print(f"Použití zařízení: {device}")
    
    if args.mode == 'train':
        train(args.labels_dir, args.atlas_path, args.output_dir, 
              args.epochs, args.save_interval, device)
    else:
        generate_samples(args.generator_path, args.atlas_path, args.output_dir, 
                        args.num_samples, args.heterogeneity, args.shape_variation,
                        args.connectivity, args.labels_dir, device)
