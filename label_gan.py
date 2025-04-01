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

class LabelGANDataset(Dataset):
    """Dataset pro LabelGAN, který generuje LABEL mapy HIE lézí"""
    
    def __init__(self, normal_atlas_path, label_dir, lesion_atlas_path=None, transform=None):
        """
        Args:
            normal_atlas_path: Cesta k normativnímu atlasu
            label_dir: Adresář s registrovanými LABEL mapami
            lesion_atlas_path: Volitelná cesta k atlasu frekvence lézí
            transform: Volitelné transformace
        """
        self.normal_atlas_path = normal_atlas_path
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
        
        # Výstupní vrstva pro LABEL mapy
        self.label_output = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(features*2, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Výstup v rozsahu [0, 1] pro LABEL
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
        
        # Generování LABEL mapy
        label_map = self.label_output(d3)
        
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

def save_sample(data, path):
    """Uložení vzorku jako .nii.gz soubor"""
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, path)

def train_label_gan(args):
    """Hlavní trénovací funkce pro LabelGAN model"""
    
    # Nastavení zařízení
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Načtení datasetu
    dataset = LabelGANDataset(
        normal_atlas_path=args.normal_atlas_path,
        label_dir=args.label_dir,
        lesion_atlas_path=args.lesion_atlas_path
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
    
    # Loss funkce
    adversarial_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()  # Pro lepší segmentaci lézí
    
    # Trénovací smyčka
    for epoch in range(args.epochs):
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
            
            # Ground truth labels pro adversarial loss
            real_output = discriminator(normal_atlas, real_label)
            real_labels = torch.ones_like(real_output)  # Cílový tensor se stejným tvarem jako real_output
            d_real_loss = adversarial_loss(real_output, real_labels)

            # Generované vzorky
            fake_label = generator(normal_atlas, noise, lesion_atlas)
            fake_output = discriminator(normal_atlas, fake_label.detach())
            fake_labels = torch.zeros_like(fake_output)  # Cílový tensor se stejným tvarem jako fake_output
            d_fake_loss = adversarial_loss(fake_output, fake_labels)
            
            # -------------------------
            # Trénink diskriminátoru
            # -------------------------
            d_optimizer.zero_grad()
            
            # Reálné vzorky
            real_output = discriminator(normal_atlas, real_label)
            d_real_loss = adversarial_loss(real_output, real_labels)
            
            # Generované vzorky
            fake_label = generator(normal_atlas, noise, lesion_atlas)
            fake_output = discriminator(normal_atlas, fake_label.detach())
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
            fake_output = discriminator(normal_atlas, fake_label)
            g_adv_loss = adversarial_loss(fake_output, real_labels)
            
            # Dice loss pro segmentaci lézí
            g_dice_loss = dice_loss(fake_label, real_label)
            
            # Přidáváme anatomicky konzistentní loss
            # Penalizujeme léze v oblastech, kde mají být vzácné (např. mozeček, ventrální rohy)
            # Toto je zjednodušená implementace - v reálné aplikaci by byla založena na atlasu struktury mozku
            anatomically_informed_loss = 0.0
            if lesion_atlas is not None:
                # Penalizujeme léze, které nejsou v souladu s atlasem frekvence lézí
                # Pro oblasti, kde jsou léze málo pravděpodobné (lesion_atlas < threshold), 
                # penalizujeme generování lézí
                threshold = 0.2  # Práh pro identifikaci oblastí s nízkou pravděpodobností lézí
                low_probability_mask = (lesion_atlas < threshold).float()
                anatomically_informed_loss = (fake_label * low_probability_mask).mean() * 10.0  # Váha pro tuto ztrátu
            
            # Celková ztráta generátoru
            g_loss = (
                g_adv_loss +
                args.lambda_dice * g_dice_loss +
                anatomically_informed_loss
            )
            g_loss.backward()
            g_optimizer.step()
            
            # Výpis stavu tréninku
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Batch {i}/{len(dataloader)}, "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                      f"ADV: {g_adv_loss.item():.4f}, DICE: {g_dice_loss.item():.4f}")
        
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
    
    print("LabelGAN Training completed!")

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
    
    for i in range(args.num_samples):
        with torch.no_grad():
            # Vytvoření náhodného šumu
            noise = torch.randn_like(normal_atlas).to(device)
            
            # Kontrola, jestli máme lesion_atlas a pokud ano, použijeme ho
            if lesion_atlas is not None:
                # Vážení šumu podle frekvence lézí
                fake_label = generator(normal_atlas, noise, lesion_atlas)
            else:
                fake_label = generator(normal_atlas, noise)
            
            # Prahování label mapy pro zajištění binarity
            fake_label_binary = (fake_label > 0.5).float()
            
            # Uložení vzorků s informativními názvy
            label_output_path = os.path.join(label_output_dir, f'sample_{i}_label.nii.gz')
            save_sample(fake_label_binary[0, 0].cpu().numpy(), label_output_path)
    
    print(f"Generated {args.num_samples} LABEL maps")
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