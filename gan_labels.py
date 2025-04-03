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
LAMBDA_STRUCTURAL = 8.0  # Váha pro strukturální loss
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

# Post-processing funkce pro vylepšení výsledků
def post_process_lesions(generated_sample, threshold=0.5, min_size=10):
    """
    Post-processing pro vylepšení struktury generovaných lézí
    
    Args:
        generated_sample: Generovaný vzorek (numpy array)
        threshold: Hodnota pro binarizaci
        min_size: Minimální velikost komponenty pro zachování
    
    Returns:
        Vylepšený binární obraz
    """
    # Binarizace
    binary = (generated_sample > threshold).astype(np.int32)
    
    # Odstranění malých komponent
    labeled, num_features = ndimage.label(binary)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes > min_size
    mask_sizes[0] = 0  # Ignorování pozadí
    binary = mask_sizes[labeled]
    
    # Použití morfologické operace pro vyplnění děr
    binary = ndimage.binary_closing(binary, structure=np.ones((3,3,3))).astype(np.float32)
    
    # Použití mírného vyhlazení pro realistické tvary
    binary = ndimage.gaussian_filter(binary.astype(float), sigma=0.5)
    
    # Konečná binarizace
    binary = (binary > 0.5).astype(np.float32)
    
    return binary

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

# Funkce pro generování nových lézí pomocí natrénovaného modelu
def generate_samples(generator_path, atlas_path, output_dir, num_samples=10, device='cuda'):
    # Vytvoření output adresáře
    os.makedirs(output_dir, exist_ok=True)
    
    # Reportování dostupné paměti
    if device == 'cuda':
        free_memory = report_gpu_memory()
        print(f"Generování začíná s {free_memory:.2f}GB volné GPU paměti")
    
    # Načtení atlasu - efektivnější načítání
    atlas_nii = nib.load(atlas_path)
    atlas = torch.tensor(atlas_nii.get_fdata(dtype=np.float32), dtype=torch.float32).unsqueeze(0).to(device)
    
    # Načtení generátoru
    generator = Generator(LATENT_DIM).to(device)
    
    # Načtení vah - podpora pro oba formáty uložení
    checkpoint = torch.load(generator_path, map_location=device)
    if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    
    # Generování vzorků
    for i in range(num_samples):
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(device)
            with torch.amp.autocast(device_type='cuda', enabled=(device=='cuda')):
                sample = generator(z, atlas)
            
            # Přesun dat na CPU a konverze na NumPy
            sample_cpu = sample.cpu().numpy()
            
            # Post-processing pro zlepšení struktury
            processed_sample = post_process_lesions(sample_cpu[0, 0])
            
            # Uložení výsledku
            sample_img = nib.Nifti1Image(processed_sample, atlas_nii.affine)
            nib.save(sample_img, os.path.join(output_dir, f'generated_sample_{i+1}.nii.gz'))
            
            # Vizualizace středového řezu
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(processed_sample[:, :, processed_sample.shape[2]//2], cmap='gray')
            plt.title(f'Generated Sample {i+1}')
            plt.subplot(1, 2, 2)
            plt.imshow(atlas[0, :, :, atlas.shape[3]//2].cpu(), cmap='hot')
            plt.title('Atlas')
            plt.savefig(os.path.join(output_dir, f'generated_sample_{i+1}.png'))
            plt.close()
            
            print(f"Vygenerován vzorek {i+1}: Sparsity = {np.mean(processed_sample):.6f}")
            
            # Explicitní uvolnění paměti
            del sample, sample_cpu, processed_sample
            if device == 'cuda':
                torch.cuda.empty_cache()

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
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    print(f"Použití zařízení: {device}")
    
    if args.mode == 'train':
        train(args.labels_dir, args.atlas_path, args.output_dir, 
              args.epochs, args.save_interval, device)
    else:
        generate_samples(args.generator_path, args.atlas_path, args.output_dir, 
                        args.num_samples, device)
