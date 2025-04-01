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

class HIELesionDataset(Dataset):
    """Dataset pro párové normativní/léze obrazy"""
    
    def __init__(self, normal_atlas_path, zadc_dir, label_dir, transform=None):
        """
        Args:
            normal_atlas_path: Cesta k normativnímu atlasu
            zadc_dir: Adresář s registrovanými ZADC mapami
            label_dir: Adresář s registrovanými LABEL mapami
            transform: Volitelné transformace
        """
        self.normal_atlas_path = normal_atlas_path
        self.zadc_files = sorted([os.path.join(zadc_dir, f) for f in os.listdir(zadc_dir) 
                                 if f.endswith('.mha') or f.endswith('.nii.gz')])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                                  if f.endswith('.mha') or f.endswith('.nii.gz')])
        self.transform = transform
        
        # Načtení normativního atlasu
        if normal_atlas_path.endswith('.nii.gz'):
            self.normal_atlas = nib.load(normal_atlas_path).get_fdata()
        else:
            self.normal_atlas = sitk.GetArrayFromImage(sitk.ReadImage(normal_atlas_path))
        
        # Normalizace do rozsahu [0, 1]
        self.normal_atlas = (self.normal_atlas - self.normal_atlas.min()) / (self.normal_atlas.max() - self.normal_atlas.min())
        
    def __len__(self):
        return len(self.zadc_files)
    
    def __getitem__(self, idx):
        # Načtení ZADC a LABEL map
        if self.zadc_files[idx].endswith('.nii.gz'):
            zadc = nib.load(self.zadc_files[idx]).get_fdata()
            label = nib.load(self.label_files[idx]).get_fdata()
        else:
            zadc = sitk.GetArrayFromImage(sitk.ReadImage(self.zadc_files[idx]))
            label = sitk.GetArrayFromImage(sitk.ReadImage(self.label_files[idx]))
            
        # Normalizace do rozsahu [0, 1]
        zadc = (zadc - zadc.min()) / (zadc.max() - zadc.min())
        
        # Převod na PyTorch tensory
        normal_atlas = torch.FloatTensor(self.normal_atlas).unsqueeze(0)  # Přidání kanálové dimenze
        zadc = torch.FloatTensor(zadc).unsqueeze(0)
        label = torch.FloatTensor(label)
        
        if self.transform:
            normal_atlas = self.transform(normal_atlas)
            zadc = self.transform(zadc)
            label = self.transform(label)
            
        return {
            'normal_atlas': normal_atlas,
            'zadc': zadc,
            'label': label
        }

class Generator(nn.Module):
    """Generator pro syntézu HIE lézí"""
    
    def __init__(self, in_channels=2, out_channels=1, features=64):
        """
        Args:
            in_channels: Počet vstupních kanálů (normativní atlas + šum)
            out_channels: Počet výstupních kanálů (syntetický obraz)
            features: Počet základních feature map
        """
        super(Generator, self).__init__()
        
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
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*8, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*4),
            nn.Dropout3d(0.5)
        )
        self.dec2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*4*2, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*2),
            nn.Dropout3d(0.5)
        )
        self.dec3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*2*2, features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features),
            nn.Dropout3d(0.5)
        )
        self.dec4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Výstup v rozsahu [-1, 1]
        )
        
    def forward(self, x, noise):
        """
        Args:
            x: Normativní atlas (B, 1, D, H, W)
            noise: Náhodný šum (B, 1, D, H, W)
        Returns:
            Syntetický obraz s lézemi (B, 1, D, H, W)
        """
        # Spojení vstupu a šumu
        x = torch.cat([x, noise], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Decoder s skip connections
        d1 = self.dec1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d4 = self.dec4(d3)
        
        return d4

class Discriminator(nn.Module):
    """Discriminator pro rozlišení reálných a syntetických obrazů"""
    
    def __init__(self, in_channels=2, features=64):
        """
        Args:
            in_channels: Počet vstupních kanálů (normativní atlas + ZADC/syntetický)
            features: Počet základních feature map
        """
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # První blok bez normalizace
            nn.Conv3d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # Další bloky
            nn.Conv3d(features, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*2),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(features*2, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*4),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(features*4, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*8),
            nn.LeakyReLU(0.2),
            
            # Výstupní vrstva
            nn.Conv3d(features*8, 1, kernel_size=4, stride=1, padding=0)
        )
    
    def forward(self, atlas, x):
        """
        Args:
            atlas: Normativní atlas (B, 1, D, H, W)
            x: Reálný nebo syntetický obraz (B, 1, D, H, W)
        Returns:
            Skóre pravděpodobnosti (B, 1, D', H', W')
        """
        # Spojení normativního atlasu a obrazu
        x = torch.cat([atlas, x], dim=1)
        return self.model(x)

def load_lesion_frequency_atlas(path):
    """Načtení atlasu frekvence lézí"""
    if path.endswith('.nii.gz'):
        atlas = nib.load(path).get_fdata()
    else:
        atlas = sitk.GetArrayFromImage(sitk.ReadImage(path))
    
    # Normalizace do rozsahu [0, 1]
    atlas = (atlas - atlas.min()) / (atlas.max() - atlas.min())
    return atlas

def train_gan(args):
    """Hlavní trénovací funkce pro GAN model"""
    
    # Nastavení zařízení
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Načtení datasetu
    dataset = HIELesionDataset(
        normal_atlas_path=args.normal_atlas_path,
        zadc_dir=args.zadc_dir,
        label_dir=args.label_dir
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Načtení atlasu frekvence lézí (pokud je dostupný)
    if args.lesion_atlas_path:
        lesion_freq_atlas = load_lesion_frequency_atlas(args.lesion_atlas_path)
        lesion_freq_atlas = torch.FloatTensor(lesion_freq_atlas).to(device)
    else:
        lesion_freq_atlas = None
    
    # Inicializace modelů
    generator = Generator(in_channels=2, out_channels=1).to(device)
    discriminator = Discriminator(in_channels=2).to(device)
    
    # Optimizátory
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Loss funkce
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    # Trénovací smyčka
    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            # Přesun dat na správné zařízení
            normal_atlas = batch['normal_atlas'].to(device)
            real_zadc = batch['zadc'].to(device)
            labels = batch['label'].to(device)
            
            # Vytvoření náhodného šumu
            batch_size = normal_atlas.size(0)
            noise = torch.randn_like(normal_atlas).to(device)
            
            # Pokud je k dispozici atlas frekvence lézí, použijeme ho k řízení šumu
            if lesion_freq_atlas is not None:
                # Vážení šumu podle frekvence lézí
                noise = noise * lesion_freq_atlas.unsqueeze(0).expand_as(noise)
            
            # Ground truth labels pro adversarial loss
            real_labels = torch.ones(batch_size, 1, 1, 1, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1, 1).to(device)
            
            # Trénink diskriminátoru
            d_optimizer.zero_grad()
            
            # Reálné vzorky
            real_output = discriminator(normal_atlas, real_zadc)
            d_real_loss = adversarial_loss(real_output, real_labels)
            
            # Generované vzorky
            fake_zadc = generator(normal_atlas, noise)
            fake_output = discriminator(normal_atlas, fake_zadc.detach())
            d_fake_loss = adversarial_loss(fake_output, fake_labels)
            
            # Celková ztráta diskriminátoru
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Trénink generátoru
            g_optimizer.zero_grad()
            
            # Adversarial loss
            fake_output = discriminator(normal_atlas, fake_zadc)
            g_adv_loss = adversarial_loss(fake_output, real_labels)
            
            # L1 loss pro zachování struktury
            g_l1_loss = l1_loss(fake_zadc, real_zadc)
            
            # Celková ztráta generátoru
            g_loss = g_adv_loss + args.lambda_l1 * g_l1_loss
            g_loss.backward()
            g_optimizer.step()
            
            # Výpis stavu tréninku
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Batch {i}/{len(dataloader)}, "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        # Uložení checkpointu
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pt'))
            
            # Uložení vygenerovaného vzorku
            with torch.no_grad():
                sample_atlas = normal_atlas[0:1]
                sample_noise = torch.randn_like(sample_atlas).to(device)
                if lesion_freq_atlas is not None:
                    sample_noise = sample_noise * lesion_freq_atlas.unsqueeze(0).expand_as(sample_noise)
                
                sample_fake = generator(sample_atlas, sample_noise)
                
                # Uložení vzorků jako .nii.gz soubory
                sample_path = os.path.join(args.output_dir, f'sample_epoch{epoch}')
                save_sample(sample_fake[0, 0].cpu().numpy(), f"{sample_path}_fake.nii.gz")
                save_sample(sample_atlas[0, 0].cpu().numpy(), f"{sample_path}_atlas.nii.gz")
                save_sample(real_zadc[0, 0].cpu().numpy(), f"{sample_path}_real.nii.gz")
    
    print("Training completed!")

def save_sample(data, path):
    """Uložení vzorku jako .nii.gz soubor"""
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, path)

def generate_samples(args):
    """Generování vzorků pomocí natrénovaného modelu"""
    
    # Nastavení zařízení
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Načtení normativního atlasu
    if args.normal_atlas_path.endswith('.nii.gz'):
        normal_atlas = nib.load(args.normal_atlas_path).get_fdata()
    else:
        normal_atlas = sitk.GetArrayFromImage(sitk.ReadImage(args.normal_atlas_path))
    
    # Normalizace
    normal_atlas = (normal_atlas - normal_atlas.min()) / (normal_atlas.max() - normal_atlas.min())
    normal_atlas = torch.FloatTensor(normal_atlas).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D, H, W]
    
    # Načtení atlasu frekvence lézí (pokud je dostupný)
    if args.lesion_atlas_path:
        lesion_freq_atlas = load_lesion_frequency_atlas(args.lesion_atlas_path)
        lesion_freq_atlas = torch.FloatTensor(lesion_freq_atlas).unsqueeze(0).to(device)  # [1, D, H, W]
    else:
        lesion_freq_atlas = None
    
    # Inicializace a načtení generátoru
    generator = Generator(in_channels=2, out_channels=1).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Generování vzorků
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i in range(args.num_samples):
        with torch.no_grad():
            # Vytvoření náhodného šumu
            noise = torch.randn_like(normal_atlas).to(device)
            
            # Pokud je k dispozici atlas frekvence lézí, použijeme ho k řízení šumu
            if lesion_freq_atlas is not None:
                noise = noise * lesion_freq_atlas
            
            # Generování vzorku
            fake_sample = generator(normal_atlas, noise)
            
            # Uložení vzorku
            output_path = os.path.join(args.output_dir, f'sample_{i}.nii.gz')
            save_sample(fake_sample[0, 0].cpu().numpy(), output_path)
    
    print(f"Generated {args.num_samples} samples at {args.output_dir}")

def main():
    """Hlavní funkce skriptu"""
    parser = argparse.ArgumentParser(description="HIE Lesion Synthesis using GANs")
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Společné argumenty
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--normal_atlas_path', type=str, required=True,
                              help='Path to normal atlas file')
    parent_parser.add_argument('--lesion_atlas_path', type=str, default=None,
                              help='Path to lesion frequency atlas (optional)')
    parent_parser.add_argument('--output_dir', type=str, default='./output',
                              help='Output directory for saving models and samples')
    
    # Parser pro trénink
    train_parser = subparsers.add_parser('train', parents=[parent_parser],
                                        help='Train a GAN model')
    train_parser.add_argument('--zadc_dir', type=str, required=True,
                             help='Directory with ZADC maps')
    train_parser.add_argument('--label_dir', type=str, required=True,
                             help='Directory with LABEL maps')
    train_parser.add_argument('--batch_size', type=int, default=4,
                             help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of epochs for training')
    train_parser.add_argument('--lr', type=float, default=0.0002,
                             help='Learning rate')
    train_parser.add_argument('--lambda_l1', type=float, default=100.0,
                             help='Weight for L1 loss')
    
    # Parser pro generování
    generate_parser = subparsers.add_parser('generate', parents=[parent_parser],
                                           help='Generate samples using a trained model')
    generate_parser.add_argument('--checkpoint_path', type=str, required=True,
                                help='Path to model checkpoint')
    generate_parser.add_argument('--num_samples', type=int, default=10,
                                help='Number of samples to generate')
    
    args = parser.parse_args()
    
    if args.action == 'train':
        train_gan(args)
    elif args.action == 'generate':
        generate_samples(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 