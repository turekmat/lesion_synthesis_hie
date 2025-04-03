import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import nibabel as nib
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Konstanty
IMAGE_SIZE = (128, 128, 64)
LATENT_DIM = 100
BATCH_SIZE = 4
EPOCHS = 200
LAMBDA_SPARSITY = 10.0  # Váha pro sparsity loss
LAMBDA_ATLAS = 5.0      # Váha pro atlas guidance loss

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
        
        # Načtení frekvenčního atlasu
        atlas_nii = nib.load(atlas_path)
        self.atlas = torch.tensor(atlas_nii.get_fdata(), dtype=torch.float32)
        
        # Seznam labelů a filtrování prázdných (celočerných)
        self.label_files = []
        for file in os.listdir(labels_dir):
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(labels_dir, file)
                label_data = nib.load(file_path).get_fdata()
                # Kontrola, zda label není celočerný
                if np.sum(label_data) > 0:
                    self.label_files.append(file)
        
        print(f"Načteno {len(self.label_files)} labelů po odfiltrování celočerných")
    
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        label_nii = nib.load(label_path)
        label = torch.tensor(label_nii.get_fdata(), dtype=torch.float32)
        
        # Normalizace dat
        if self.transform:
            label = self.transform(label)
            
        return {'label': label, 'atlas': self.atlas}

# Generátor
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
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

# Diskriminátor
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # První vrstva - bez batchnorm
            nn.Conv3d(2, 16, 4, stride=2, padding=1),  # 64x64x32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(16, 32, 4, stride=2, padding=1),  # 32x32x16
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(32, 64, 4, stride=2, padding=1),  # 16x16x8
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(64, 128, 4, stride=2, padding=1),  # 8x8x4
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(128, 256, 4, stride=2, padding=1),  # 4x4x2
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Výstupní vrstva
        self.output_layer = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, atlas):
        # Spojení vstupu a atlasu jako kanály
        atlas_expanded = atlas.unsqueeze(1)  # Přidání kanálového rozměru
        x = torch.cat([img, atlas_expanded], dim=1)
        
        features = self.model(x)
        features = features.view(features.size(0), -1)
        validity = self.output_layer(features)
        
        return validity

# Custom loss funkce
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

# Hlavní trénovací funkce
def train(labels_dir, atlas_path, output_dir, device='cuda'):
    # Vytvoření output adresáře
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializace datasetů a dataloaderů
    dataset = HIELesionDataset(labels_dir, atlas_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Inicializace modelů
    generator = Generator(LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizery
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss funkce
    adversarial_loss = nn.BCELoss()
    
    # Trénovací smyčka
    for epoch in range(EPOCHS):
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            
            for i, batch in enumerate(tepoch):
                labels = batch['label'].to(device)
                atlas = batch['atlas'].to(device)
                
                # Reálné a falešné labely pro discriminator
                real_labels = torch.ones(labels.size(0), 1).to(device)
                fake_labels = torch.zeros(labels.size(0), 1).to(device)
                
                # -----------------
                # Trénink discriminatoru
                # -----------------
                optimizer_D.zero_grad()
                
                # Diskriminátor na reálných datech
                labels_unsqueeze = labels.unsqueeze(1)  # Přidání kanálového rozměru
                real_validity = discriminator(labels_unsqueeze, atlas)
                d_real_loss = adversarial_loss(real_validity, real_labels)
                
                # Diskriminátor na generovaných datech
                z = torch.randn(labels.size(0), LATENT_DIM).to(device)
                fake_images = generator(z, atlas)
                fake_validity = discriminator(fake_images.detach(), atlas)
                d_fake_loss = adversarial_loss(fake_validity, fake_labels)
                
                # Celková loss discriminatoru
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                
                # -----------------
                # Trénink generátoru
                # -----------------
                optimizer_G.zero_grad()
                
                # Generátor se snaží oklamat diskriminátor
                z = torch.randn(labels.size(0), LATENT_DIM).to(device)
                fake_images = generator(z, atlas)
                fake_validity = discriminator(fake_images, atlas)
                g_adversarial_loss = adversarial_loss(fake_validity, real_labels)
                
                # Atlas-guided loss
                g_atlas_loss = atlas_guided_loss(fake_images, atlas)
                
                # Sparsity loss
                g_sparsity_loss = sparsity_loss(fake_images, target_sparsity=0.01)
                
                # Celková loss generátoru
                g_loss = g_adversarial_loss + LAMBDA_ATLAS * g_atlas_loss + LAMBDA_SPARSITY * g_sparsity_loss
                g_loss.backward()
                optimizer_G.step()
                
                # Aktualizace progress baru
                tepoch.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item(), 
                                  Sparsity=torch.mean(fake_images).item())
        
        # Uložení modelů každých 10 epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            torch.save(generator.state_dict(), os.path.join(output_dir, f'generator_{epoch+1}.pt'))
            torch.save(discriminator.state_dict(), os.path.join(output_dir, f'discriminator_{epoch+1}.pt'))
            
            # Vygenerování vzorků pro vizualizaci
            with torch.no_grad():
                z = torch.randn(1, LATENT_DIM).to(device)
                atlas_sample = dataset[0]['atlas'].unsqueeze(0).to(device)
                sample = generator(z, atlas_sample).cpu().numpy()
                
                # Uložení vzorků jako nii file
                sample_img = nib.Nifti1Image(sample[0, 0], np.eye(4))
                nib.save(sample_img, os.path.join(output_dir, f'sample_epoch_{epoch+1}.nii.gz'))
                
                # Vizualizace středového řezu
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(sample[0, 0, :, :, sample.shape[4]//2], cmap='gray')
                plt.title(f'Generated - Epoch {epoch+1}')
                plt.subplot(1, 2, 2)
                plt.imshow(atlas_sample[0, :, :, atlas_sample.shape[3]//2].cpu(), cmap='hot')
                plt.title('Atlas')
                plt.savefig(os.path.join(output_dir, f'sample_visual_epoch_{epoch+1}.png'))
                plt.close()

# Funkce pro generování nových lézí pomocí natrénovaného modelu
def generate_samples(generator_path, atlas_path, output_dir, num_samples=10, device='cuda'):
    # Vytvoření output adresáře
    os.makedirs(output_dir, exist_ok=True)
    
    # Načtení atlasu
    atlas_nii = nib.load(atlas_path)
    atlas = torch.tensor(atlas_nii.get_fdata(), dtype=torch.float32).unsqueeze(0).to(device)
    
    # Načtení generátoru
    generator = Generator(LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()
    
    # Generování vzorků
    for i in range(num_samples):
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(device)
            sample = generator(z, atlas).cpu().numpy()
            
            # Binarizace generovaných dat
            binary_sample = (sample[0, 0] > 0.5).astype(np.float32)
            
            # Uložení výsledku
            sample_img = nib.Nifti1Image(binary_sample, atlas_nii.affine)
            nib.save(sample_img, os.path.join(output_dir, f'generated_sample_{i+1}.nii.gz'))
            
            # Vizualizace středového řezu
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(binary_sample[:, :, binary_sample.shape[2]//2], cmap='gray')
            plt.title(f'Generated Sample {i+1}')
            plt.subplot(1, 2, 2)
            plt.imshow(atlas[0, :, :, atlas.shape[3]//2].cpu(), cmap='hot')
            plt.title('Atlas')
            plt.savefig(os.path.join(output_dir, f'generated_sample_{i+1}.png'))
            plt.close()
            
            print(f"Vygenerován vzorek {i+1}: Sparsity = {np.mean(binary_sample):.6f}")

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
    parser.add_argument('--gpu', action='store_true', help='použít GPU')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    print(f"Použití zařízení: {device}")
    
    if args.mode == 'train':
        train(args.labels_dir, args.atlas_path, args.output_dir, device)
    else:
        generate_samples(args.generator_path, args.atlas_path, args.output_dir, args.num_samples, device)
