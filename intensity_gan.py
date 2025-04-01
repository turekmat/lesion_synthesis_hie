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

class IntensityGANDataset(Dataset):
    """Dataset pro IntensityGAN, který upravuje intenzity voxelů dle LABEL map"""
    
    def __init__(self, normal_atlas_path, zadc_dir, label_dir, lesion_atlas_path=None, transform=None):
        """
        Args:
            normal_atlas_path: Cesta k normativnímu atlasu
            zadc_dir: Adresář s registrovanými ZADC mapami
            label_dir: Adresář s registrovanými LABEL mapami (skutečnými nebo generovanými)
            lesion_atlas_path: Volitelná cesta k atlasu frekvence lézí
            transform: Volitelné transformace
        """
        self.normal_atlas_path = normal_atlas_path
        self.zadc_files = sorted([os.path.join(zadc_dir, f) for f in os.listdir(zadc_dir) 
                                 if f.endswith('.mha') or f.endswith('.nii.gz') or f.endswith('.nii')])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                                  if f.endswith('.mha') or f.endswith('.nii.gz') or f.endswith('.nii')])
        self.lesion_atlas_path = lesion_atlas_path
        self.transform = transform
        
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
        return len(self.zadc_files)
    
    def __getitem__(self, idx):
        # Načtení ZADC a LABEL map
        if self.zadc_files[idx].endswith('.nii.gz') or self.zadc_files[idx].endswith('.nii'):
            zadc = nib.load(self.zadc_files[idx]).get_fdata()
            label = nib.load(self.label_files[idx]).get_fdata()
        else:
            zadc = sitk.GetArrayFromImage(sitk.ReadImage(self.zadc_files[idx]))
            label = sitk.GetArrayFromImage(sitk.ReadImage(self.label_files[idx]))
            
        # Normalizace ZADC do rozsahu [0, 1]
        zadc = (zadc - zadc.min()) / (zadc.max() - zadc.min()) if zadc.max() > zadc.min() else zadc
        
        # Binarizace LABEL mapy (zajistíme, že jsou jen hodnoty 0 a 1)
        label = (label > 0).astype(np.float32)
        
        # Převod na PyTorch tensory
        normal_atlas = torch.FloatTensor(self.normal_atlas).unsqueeze(0)  # Přidání kanálové dimenze
        zadc = torch.FloatTensor(zadc).unsqueeze(0)
        label = torch.FloatTensor(label).unsqueeze(0)  # Přidání kanálové dimenze
        
        if self.transform:
            normal_atlas = self.transform(normal_atlas)
            zadc = self.transform(zadc)
            label = self.transform(label)
        
        return_dict = {
            'normal_atlas': normal_atlas,
            'zadc': zadc,
            'label': label
        }
        
        # Přidání atlasu frekvence lézí, pokud je k dispozici
        if self.lesion_atlas is not None:
            lesion_atlas = torch.FloatTensor(self.lesion_atlas).unsqueeze(0)
            if self.transform:
                lesion_atlas = self.transform(lesion_atlas)
            return_dict['lesion_atlas'] = lesion_atlas
            
        return return_dict

class IntensityGenerator(nn.Module):
    """Generator pro úpravu intenzit voxelů podle LABEL map"""
    
    def __init__(self, in_channels=3, features=64, dropout_rate=0.3, use_self_attention=False):
        """
        Args:
            in_channels: Počet vstupních kanálů (normativní atlas + LABEL mapa + šum)
            features: Počet základních feature map
            dropout_rate: Míra dropout v generátoru
            use_self_attention: Použít self-attention mechanismus pro lepší detaily
        """
        super(IntensityGenerator, self).__init__()
        
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
        
        # Výstupní vrstva pro ZADC mapy
        self.intensity_output = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*2, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Výstup v rozsahu [-1, 1] pro ZADC
        )
        
    def forward(self, atlas, label, noise, lesion_atlas=None):
        """
        Args:
            atlas: Normativní atlas (B, 1, D, H, W)
            label: LABEL mapa (B, 1, D, H, W)
            noise: Náhodný šum (B, 1, D, H, W)
            lesion_atlas: Volitelný atlas frekvence lézí (B, 1, D, H, W)
        Returns:
            ZADC mapa
        """
        # Spojení vstupu (atlas, label, šum)
        if lesion_atlas is not None:
            # Pokud máme atlas lézí, vážíme šum podle něj
            weighted_noise = noise * lesion_atlas
            x = torch.cat([atlas, label, weighted_noise], dim=1)
        else:
            x = torch.cat([atlas, label, noise], dim=1)
        
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
        
        # Generování ZADC mapy
        zadc_map = self.intensity_output(d3)
        
        return zadc_map

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

class IntensityDiscriminator(nn.Module):
    """Discriminator pro rozlišení reálných a syntetických ZADC map"""
    
    def __init__(self, in_channels=3, features=64, use_spectral_norm=False):
        """
        Args:
            in_channels: Počet vstupních kanálů (normativní atlas + ZADC + LABEL mapa)
            features: Počet základních feature map
            use_spectral_norm: Použít spektrální normalizaci pro stabilnější trénink
        """
        super(IntensityDiscriminator, self).__init__()
        
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
    
    def forward(self, atlas, zadc, label):
        """
        Args:
            atlas: Normativní atlas (B, 1, D, H, W)
            zadc: Reálný nebo syntetický ZADC obraz (B, 1, D, H, W)
            label: LABEL mapa (B, 1, D, H, W)
        Returns:
            Skóre pravděpodobnosti (B, 1, D', H', W')
        """
        # Spojení normativního atlasu, ZADC a LABEL
        x = torch.cat([atlas, zadc, label], dim=1)
        return self.model(x)

def save_sample(data, path):
    """Uložení vzorku jako .nii.gz soubor"""
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, path)

def train_intensity_gan(args):
    """Hlavní trénovací funkce pro IntensityGAN model"""
    
    # Nastavení zařízení
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Načtení datasetu
    dataset = IntensityGANDataset(
        normal_atlas_path=args.normal_atlas_path,
        zadc_dir=args.zadc_dir,
        label_dir=args.label_dir,
        lesion_atlas_path=args.lesion_atlas_path
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Inicializace modelů
    generator = IntensityGenerator(
        in_channels=3,  # Atlas + LABEL + šum
        features=args.generator_filters,
        dropout_rate=args.dropout_rate,
        use_self_attention=args.use_self_attention
    ).to(device)
    
    discriminator = IntensityDiscriminator(
        in_channels=3,  # Atlas + ZADC + LABEL
        features=args.discriminator_filters,
        use_spectral_norm=args.use_spectral_norm
    ).to(device)
    
    # Optimizátory
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    # Loss funkce
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    # Trénovací smyčka
    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            # Přesun dat na správné zařízení
            normal_atlas = batch['normal_atlas'].to(device)
            real_zadc = batch['zadc'].to(device)
            label = batch['label'].to(device)
            
            # Příprava lesion_atlas, pokud je k dispozici
            lesion_atlas = batch.get('lesion_atlas', None)
            if lesion_atlas is not None:
                lesion_atlas = lesion_atlas.to(device)
            
            # Vytvoření náhodného šumu
            batch_size = normal_atlas.size(0)
            noise = torch.randn_like(normal_atlas).to(device)
            
            # Ground truth labels pro adversarial loss
            real_labels = torch.ones(batch_size, 1, 1, 1, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1, 1).to(device)
            
            # -------------------------
            # Trénink diskriminátoru
            # -------------------------
            d_optimizer.zero_grad()
            
            # Reálné vzorky
            real_output = discriminator(normal_atlas, real_zadc, label)
            d_real_loss = adversarial_loss(real_output, real_labels)
            
            # Generované vzorky
            fake_zadc = generator(normal_atlas, label, noise, lesion_atlas)
            fake_output = discriminator(normal_atlas, fake_zadc.detach(), label)
            d_fake_loss = adversarial_loss(fake_output, fake_labels)
            
            # Celková ztráta diskriminátoru
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # -------------------------
            # Trénink generátoru
            # -------------------------
            g_optimizer.zero_grad()
            
            # Adversarial loss
            fake_output = discriminator(normal_atlas, fake_zadc, label)
            g_adv_loss = adversarial_loss(fake_output, real_labels)
            
            # L1 loss pro zachování struktury ZADC
            g_l1_loss = l1_loss(fake_zadc, real_zadc)
            
            # Specifický loss pro oblasti s lézemi
            # Zajišťujeme, že generované intenzity v oblasti lézí odpovídají realitě
            lesion_mask = (label > 0.5).float()
            non_lesion_mask = 1 - lesion_mask
            
            # L1 loss pro léze
            lesion_l1_loss = l1_loss(fake_zadc * lesion_mask, real_zadc * lesion_mask)
            
            # L1 loss pro normální tkáň
            non_lesion_l1_loss = l1_loss(fake_zadc * non_lesion_mask, real_zadc * non_lesion_mask)
            
            # Intenzitní variabilita: povzbudit variaci intenzit v oblastech lézí
            # Penalizujeme malou směrodatnou odchylku intenzit v oblasti lézí
            intensity_var_loss = 0.0
            if torch.sum(lesion_mask) > 0:  # Jen pokud existují léze
                lesion_intensities = fake_zadc * lesion_mask
                lesion_std = torch.std(lesion_intensities[lesion_mask > 0])
                intensity_var_loss = 1.0 / (lesion_std + 1e-6)
            
            # Celková ztráta generátoru
            g_loss = (
                g_adv_loss +
                args.lambda_l1 * g_l1_loss +
                args.lambda_lesion * lesion_l1_loss +
                args.lambda_non_lesion * non_lesion_l1_loss +
                args.lambda_intensity_var * intensity_var_loss
            )
            g_loss.backward()
            g_optimizer.step()
            
            # Výpis stavu tréninku
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Batch {i}/{len(dataloader)}, "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                      f"ADV: {g_adv_loss.item():.4f}, L1: {g_l1_loss.item():.4f}, "
                      f"Lesion L1: {lesion_l1_loss.item():.4f}, Non-Lesion L1: {non_lesion_l1_loss.item():.4f}")
        
        # Uložení checkpointu
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(args.output_dir, f'intensitygan_checkpoint_epoch{epoch}.pt'))
            
            # Uložení vygenerovaného vzorku
            with torch.no_grad():
                sample_atlas = normal_atlas[0:1]
                sample_label = label[0:1]
                sample_noise = torch.randn_like(sample_atlas).to(device)
                
                if lesion_atlas is not None:
                    sample_lesion_atlas = lesion_atlas[0:1]
                    sample_fake_zadc = generator(sample_atlas, sample_label, sample_noise, sample_lesion_atlas)
                else:
                    sample_fake_zadc = generator(sample_atlas, sample_label, sample_noise)
                
                # Uložení vzorků jako .nii.gz soubory
                sample_path = os.path.join(args.output_dir, f'intensitygan_sample_epoch{epoch}')
                save_sample(sample_fake_zadc[0, 0].cpu().numpy(), f"{sample_path}_fake_zadc.nii.gz")
                save_sample(sample_atlas[0, 0].cpu().numpy(), f"{sample_path}_atlas.nii.gz")
                save_sample(sample_label[0, 0].cpu().numpy(), f"{sample_path}_label.nii.gz")
                save_sample(real_zadc[0, 0].cpu().numpy(), f"{sample_path}_real_zadc.nii.gz")
    
    print("IntensityGAN Training completed!")

def generate_intensity_samples(args):
    """Generování vzorků ZADC map pomocí natrénovaného modelu"""
    
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
    
    # Načtení LABEL map
    label_files = sorted([os.path.join(args.label_dir, f) for f in os.listdir(args.label_dir)
                         if f.endswith('.mha') or f.endswith('.nii.gz') or f.endswith('.nii')])
    
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
    generator = IntensityGenerator(
        in_channels=3,
        features=args.generator_filters,
        dropout_rate=args.dropout_rate,
        use_self_attention=args.use_self_attention
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Generování vzorků
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Vytvoření adresáře pro ZADC mapy
    zadc_output_dir = os.path.join(args.output_dir, "zadc")
    os.makedirs(zadc_output_dir, exist_ok=True)
    
    # Pro každou LABEL mapu vygenerujeme odpovídající ZADC mapu
    for i, label_file in enumerate(label_files):
        if i >= args.num_samples:
            break
            
        # Načtení LABEL mapy
        if label_file.endswith('.nii.gz') or label_file.endswith('.nii'):
            label = nib.load(label_file).get_fdata()
        else:
            label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))
        
        # Binarizace LABEL mapy
        label = (label > 0).astype(np.float32)
        label = torch.FloatTensor(label).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D, H, W]
        
        with torch.no_grad():
            # Vytvoření náhodného šumu
            noise = torch.randn_like(normal_atlas).to(device)
            
            # Náhodná úprava intenzity (variabilita mezi vzorky)
            intensity_factor = torch.rand(1).item() * 0.4 + 0.8  # Náhodný faktor 0.8-1.2
            
            # Kontrola, jestli máme lesion_atlas a pokud ano, použijeme ho
            if lesion_atlas is not None:
                fake_zadc = generator(normal_atlas, label, noise, lesion_atlas)
            else:
                fake_zadc = generator(normal_atlas, label, noise)
            
            # Aplikace intenzitního faktoru na ZADC mapu (pouze nenulové hodnoty)
            mask = fake_zadc != 0
            fake_zadc_adjusted = fake_zadc.clone()
            fake_zadc_adjusted[mask] = fake_zadc[mask] * intensity_factor
            
            # Úprava intenzit v celém mozku pro větší variabilitu
            # Aplikujeme malé lokální šumové modifikace na celou ZADC mapu
            local_intensity_noise = torch.randn_like(fake_zadc) * 0.05  # 5% šum
            brain_mask = fake_zadc_adjusted != 0
            fake_zadc_adjusted[brain_mask] = fake_zadc_adjusted[brain_mask] + local_intensity_noise[brain_mask]
            
            # Uložení vzorku s informativním názvem
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            zadc_output_path = os.path.join(zadc_output_dir, f'{base_name}_zadc.nii.gz')
            save_sample(fake_zadc_adjusted[0, 0].cpu().numpy(), zadc_output_path)
    
    print(f"Generated ZADC maps for {min(len(label_files), args.num_samples)} LABEL maps")
    print(f"ZADC maps saved to: {zadc_output_dir}")

def main():
    """Hlavní funkce skriptu"""
    parser = argparse.ArgumentParser(description="IntensityGAN for HIE Lesion ZADC Map Synthesis")
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
    
    # Parser pro trénink
    train_parser = subparsers.add_parser('train', parents=[parent_parser],
                                        help='Trénink IntensityGAN modelu')
    train_parser.add_argument('--zadc_dir', type=str, required=True,
                             help='Adresář s registrovanými ZADC mapami')
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
    train_parser.add_argument('--lambda_l1', type=float, default=100.0,
                             help='Váha pro L1 loss')
    train_parser.add_argument('--lambda_lesion', type=float, default=50.0,
                             help='Váha pro L1 loss v oblasti lézí')
    train_parser.add_argument('--lambda_non_lesion', type=float, default=25.0,
                             help='Váha pro L1 loss mimo oblasti lézí')
    train_parser.add_argument('--lambda_intensity_var', type=float, default=10.0,
                             help='Váha pro loss intenzitní variability')
    
    # Parser pro generování
    generate_parser = subparsers.add_parser('generate', parents=[parent_parser],
                                           help='Generování vzorků ZADC map pomocí natrénovaného modelu')
    generate_parser.add_argument('--checkpoint_path', type=str, required=True,
                                help='Cesta k checkpointu modelu')
    generate_parser.add_argument('--label_dir', type=str, required=True, 
                                help='Adresář s LABEL mapami pro generování ZADC')
    generate_parser.add_argument('--num_samples', type=int, default=10,
                                help='Maximální počet vzorků k vygenerování')
    
    args = parser.parse_args()
    
    if args.action == 'train':
        train_intensity_gan(args)
    elif args.action == 'generate':
        generate_intensity_samples(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 