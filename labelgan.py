import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import SimpleITK as sitk
from torchvision import transforms
import argparse
import time
from datetime import datetime
import glob
from tqdm import tqdm

# Nastavení reprodukovatelnosti
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset pro načítání lézí a atlasů
class LesionDataset(Dataset):
    def __init__(self, label_dir, normal_atlas_path, lesion_atlas_path=None, transform=None):
        """
        Dataset pro trénink LabelGAN
        
        Args:
            label_dir: Adresář s registrovanými LABEL mapami (.nii, .nii.gz nebo .mha)
            normal_atlas_path: Cesta k normativnímu atlasu
            lesion_atlas_path: Volitelná cesta k atlasu frekvence lézí
            transform: Transformace pro augmentaci dat
        """
        self.label_dir = label_dir
        self.normal_atlas_path = normal_atlas_path
        self.lesion_atlas_path = lesion_atlas_path
        self.transform = transform
        
        # Načtení všech souborů s lézemi
        self.label_files = []
        for ext in ['*.nii', '*.nii.gz', '*.mha']:
            self.label_files.extend(glob.glob(os.path.join(label_dir, ext)))
        
        print(f"Načteno {len(self.label_files)} LABEL souborů.")
        
        # Načtení normativního atlasu
        if self.normal_atlas_path.endswith('.nii.gz') or self.normal_atlas_path.endswith('.nii'):
            self.normal_atlas = nib.load(normal_atlas_path).get_fdata()
        else:
            self.normal_atlas = sitk.GetArrayFromImage(sitk.ReadImage(normal_atlas_path))
        
        # Normalizace atlasu na rozsah [0, 1]
        self.normal_atlas = (self.normal_atlas - self.normal_atlas.min()) / (self.normal_atlas.max() - self.normal_atlas.min())
        
        # Načtení atlasu frekvence lézí, pokud je dostupný
        if self.lesion_atlas_path:
            if self.lesion_atlas_path.endswith('.nii.gz') or self.lesion_atlas_path.endswith('.nii'):
                self.lesion_atlas = nib.load(self.lesion_atlas_path).get_fdata()
            else:
                self.lesion_atlas = sitk.GetArrayFromImage(sitk.ReadImage(self.lesion_atlas_path))
            
            # Normalizace atlasu lézí na rozsah [0, 1]
            self.lesion_atlas = (self.lesion_atlas - self.lesion_atlas.min()) / (self.lesion_atlas.max() - self.lesion_atlas.min())
        else:
            self.lesion_atlas = None
    
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        # Načtení LABEL mapy
        label_path = self.label_files[idx]
        
        if label_path.endswith('.nii.gz') or label_path.endswith('.nii'):
            label = nib.load(label_path).get_fdata()
        else:
            label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        
        # Konverze na binární masku (0 nebo 1)
        label = (label > 0).astype(np.float32)
        
        # Konverze na tensor
        label = torch.FloatTensor(label).unsqueeze(0)  # Přidání kanálové dimenze
        normal_atlas = torch.FloatTensor(self.normal_atlas).unsqueeze(0)  # Přidání kanálové dimenze
        
        # Aplikace transformace, pokud je definována
        if self.transform:
            label = self.transform(label)
            normal_atlas = self.transform(normal_atlas)
        
        # Vytvoření slovníku pro návratové hodnoty
        return_dict = {
            'label': label,
            'normal_atlas': normal_atlas,
            'path': label_path
        }
        
        # Přidání atlasu frekvence lézí, pokud je k dispozici
        if self.lesion_atlas is not None:
            lesion_atlas = torch.FloatTensor(self.lesion_atlas).unsqueeze(0)
            if self.transform:
                lesion_atlas = self.transform(lesion_atlas)
            return_dict['lesion_atlas'] = lesion_atlas
        
        return return_dict

# 3D konvoluční blok pro generátor
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=True, use_relu=True, use_dropout=False, dropout_prob=0.2):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_norm = use_norm
        self.use_relu = use_relu
        self.use_dropout = use_dropout
        
        if use_norm:
            self.norm = nn.InstanceNorm3d(out_channels)
        
        if use_relu:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
            
        if use_dropout:
            self.dropout = nn.Dropout3d(dropout_prob)
    
    def forward(self, x):
        x = self.conv(x)
        
        if self.use_norm:
            x = self.norm(x)
            
        if self.use_relu:
            x = self.relu(x)
            
        if self.use_dropout:
            x = self.dropout(x)
            
        return x

# 3D dekonvoluční blok pro generátor
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, use_norm=True, use_relu=True, use_dropout=False, dropout_prob=0.2):
        super(DeconvBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.use_norm = use_norm
        self.use_relu = use_relu
        self.use_dropout = use_dropout
        
        if use_norm:
            self.norm = nn.InstanceNorm3d(out_channels)
        
        if use_relu:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
            
        if use_dropout:
            self.dropout = nn.Dropout3d(dropout_prob)
    
    def forward(self, x):
        x = self.deconv(x)
        
        if self.use_norm:
            x = self.norm(x)
            
        if self.use_relu:
            x = self.relu(x)
            
        if self.use_dropout:
            x = self.dropout(x)
            
        return x

# Generator s attention mechanismem
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_maps=64, use_attention=True):
        """
        3D U-Net generátor pro LabelGAN
        
        Args:
            in_channels: Počet vstupních kanálů (normativní atlas + šum + volitelně atlas lézí)
            out_channels: Počet výstupních kanálů (1 pro binární masku lézí)
            feature_maps: Počet základních feature map
            use_attention: Použít self-attention mechanismus
        """
        super(Generator, self).__init__()
        
        self.use_attention = use_attention
        
        # Encoder (downsample)
        self.conv1 = ConvBlock(in_channels, feature_maps)
        self.conv2 = ConvBlock(feature_maps, feature_maps*2, stride=2)
        self.conv3 = ConvBlock(feature_maps*2, feature_maps*4, stride=2)
        self.conv4 = ConvBlock(feature_maps*4, feature_maps*8, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(feature_maps*8, feature_maps*8, use_dropout=True)
        
        # Self-attention na bottlenecku
        if use_attention:
            self.attention = SelfAttention3D(feature_maps*8)
        
        # Decoder (upsample)
        self.deconv4 = DeconvBlock(feature_maps*8, feature_maps*4)
        self.deconv3 = DeconvBlock(feature_maps*8, feature_maps*2)  # *8 kvůli skip connection
        self.deconv2 = DeconvBlock(feature_maps*4, feature_maps)    # *4 kvůli skip connection
        
        # Finální vrstva
        self.final = nn.Conv3d(feature_maps*2, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, lesion_atlas=None):
        """
        Forward pass
        
        Args:
            x: Vstupní tensor [B, in_channels, D, H, W]
            lesion_atlas: Volitelný atlas frekvence lézí [B, 1, D, H, W]
        
        Returns:
            Binární maska lézí [B, 1, D, H, W]
        """
        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Self-attention na bottlenecku
        if self.use_attention:
            b = self.attention(b)
        
        # Decoder s skip connections
        d4 = self.deconv4(b)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.deconv3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.deconv2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        # Finální vrstva
        out = self.final(d2)
        out = self.sigmoid(out)
        
        # Pokud máme atlas lézí, můžeme jej použít k modulaci výstupu
        if lesion_atlas is not None:
            # Modulace pravděpodobnosti výskytu lézí podle atlasu
            # Použijeme vážený průměr mezi naším výstupem a atlasem lézí
            alpha = 0.7  # Váha pro náš výstup (vs. atlas)
            out = alpha * out + (1 - alpha) * lesion_atlas * (out > 0.1).float()
        
        return out

# Self-attention mechanismus pro 3D data
class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        
        self.query = nn.Conv3d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        
        # Flatten prostorové dimenze
        query = self.query(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, depth * height * width)
        value = self.value(x).view(batch_size, -1, depth * height * width)
        
        # Attention mapa
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=1)
        
        # Výstup
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height, width)
        
        # Residual connection
        out = self.gamma * out + x
        
        return out

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feature_maps=64, use_attention=True):
        """
        PatchGAN diskriminátor pro 3D data
        
        Args:
            in_channels: Počet vstupních kanálů (normativní atlas + léze + volitelně atlas lézí)
            feature_maps: Počet základních feature map
            use_attention: Použít self-attention mechanismus
        """
        super(Discriminator, self).__init__()
        
        self.use_attention = use_attention
        
        # Sequence of convolutional layers
        self.conv1 = ConvBlock(in_channels, feature_maps, use_norm=False)
        self.conv2 = ConvBlock(feature_maps, feature_maps*2)
        self.conv3 = ConvBlock(feature_maps*2, feature_maps*4)
        self.conv4 = ConvBlock(feature_maps*4, feature_maps*8)
        
        # Self-attention layer
        if use_attention:
            self.attention = SelfAttention3D(feature_maps*4)
        
        # Output layer
        self.output = nn.Conv3d(feature_maps*8, 1, kernel_size=3, padding=1)
        
    def forward(self, atlas, label, lesion_atlas=None):
        """
        Forward pass
        
        Args:
            atlas: Normativní atlas [B, 1, D, H, W]
            label: Léze (binární maska) [B, 1, D, H, W]
            lesion_atlas: Volitelný atlas frekvence lézí [B, 1, D, H, W]
            
        Returns:
            PatchGAN výstupy [B, 1, D//8, H//8, W//8]
        """
        # Spojení vstupu - vždy zahrnout atlas a label
        inputs = [atlas, label]
        
        # Pokud je k dispozici lesion_atlas, přidáme jej
        if lesion_atlas is not None:
            inputs.append(lesion_atlas)
        
        # Spojení všech vstupů
        x = torch.cat(inputs, dim=1)
        
        # Sequence of convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Self-attention
        if self.use_attention:
            x = self.attention(x)
            
        x = self.conv4(x)
        
        # Output layer
        x = self.output(x)
        
        return x

# Dice Loss pro segmentaci
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Zploštění predikcí a cílů
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Výpočet Dice koeficientu
        intersection = (pred_flat * target_flat).sum()
        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        return 1 - dice

# Focal Loss pro nevyvážená data
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        # Focal term
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Vážení pozitivních a negativních příkladů
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Kombinace
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()

# Funkce pro uložení 3D obrazu ve formátu .nii nebo .nii.gz
def save_nifti(data, output_path, reference_nifti=None):
    """
    Uloží data jako NIFTI soubor
    
    Args:
        data: 3D numpy array
        output_path: Výstupní cesta
        reference_nifti: Volitelně referenční NIFTI objekt pro zachování metadat
    """
    if reference_nifti is not None:
        # Zachování metadat z referenčního NIFTI
        nifti_img = nib.Nifti1Image(data, reference_nifti.affine, reference_nifti.header)
    else:
        # Vytvoření nového NIFTI
        nifti_img = nib.Nifti1Image(data, np.eye(4))
    
    # Uložení
    nib.save(nifti_img, output_path)
    print(f"Uloženo: {output_path}")

# Funkce pro výpočet metrik
def calculate_metrics(pred, target):
    """
    Výpočet metrik pro vyhodnocení kvality segmentace
    
    Args:
        pred: Predikovaná binární maska
        target: Cílová binární maska
        
    Returns:
        Dictionary s metrikami
    """
    # Konverze na binární masky
    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)
    
    # True positives, false positives, etc.
    tp = np.sum(pred_binary * target_binary)
    fp = np.sum(pred_binary * (1 - target_binary))
    fn = np.sum((1 - pred_binary) * target_binary)
    tn = np.sum((1 - pred_binary) * (1 - target_binary))
    
    # Metriky
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Procento voxelů s lézemi
    lesion_percentage = np.sum(pred_binary) / pred_binary.size * 100
    
    return {
        'precision': precision,
        'recall': recall,
        'dice': dice,
        'iou': iou,
        'lesion_percentage': lesion_percentage
    }

# Funkce pro vytvoření augmentací
def create_transform_pipeline():
    """
    Vytvoří pipeline transformací pro augmentaci dat
    
    Returns:
        Transformace pro 3D data
    """
    class RandomFlip3D:
        def __init__(self, p=0.5):
            self.p = p
            
        def __call__(self, tensor):
            if random.random() < self.p:
                # Náhodný flip podél jedné z os
                axis = random.randint(1, 3)  # Dimension 0 is channels
                tensor = torch.flip(tensor, [axis])
            return tensor
    
    class RandomRotate3D:
        def __init__(self, max_angle=15, p=0.5):
            self.max_angle = max_angle
            self.p = p
            
        def __call__(self, tensor):
            if random.random() < self.p:
                # Náhodná rotace kolem jedné z os
                angle = random.uniform(-self.max_angle, self.max_angle)
                axis = random.randint(1, 3)
                
                # Převod na numpy, rotace, převod zpět na tensor
                np_tensor = tensor.numpy()
                
                # Implementace rotace (simplified)
                # V reálném kódu by zde byla plná implementace 3D rotace
                # Pro jednoduchost zde uvádím zjednodušenou verzi
                if axis == 1:
                    # Rotace kolem osy Z
                    pass
                elif axis == 2:
                    # Rotace kolem osy Y
                    pass
                else:
                    # Rotace kolem osy X
                    pass
                
                tensor = torch.from_numpy(np_tensor)
            return tensor
    
    # Vytvoření kompozitní transformace
    transform = transforms.Compose([
        RandomFlip3D(p=0.5),
        # RandomRotate3D(max_angle=10, p=0.3),  # Komentováno pro jednoduchost, vyžaduje komplexní implementaci
    ])
    
    return transform

# Funkce pro trénování LabelGAN
def train_labelgan(args):
    """
    Hlavní funkce pro trénování LabelGAN
    
    Args:
        args: Argumenty příkazové řádky
    """
    # Nastavení reprodukovatelnosti
    set_seed(args.seed)
    
    # Určení zařízení
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Použité zařízení: {device}")
    
    # Vytvoření výstupních adresářů
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, "models")
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Vytvoření transformací pro augmentaci
    transform = create_transform_pipeline()
    
    # Načtení datasetu
    dataset = LesionDataset(
        label_dir=args.label_dir,
        normal_atlas_path=args.normal_atlas_path,
        lesion_atlas_path=args.lesion_atlas_path,
        transform=transform
    )
    
    # Rozdělení datasetu na trénovací a validační
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.val_split * dataset_size))
    
    if args.shuffle:
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Vytvoření SubsetRandomSampler
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Vytvoření dataloaderů
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers
    )
    
    print(f"Trénovací vzorky: {len(train_indices)}, Validační vzorky: {len(val_indices)}")
    
    # Konfigurace vstupních kanálů
    if args.lesion_atlas_path:
        disc_in_channels = 3  # atlas + label + lesion atlas
    else:
        disc_in_channels = 2  # atlas + label
    
    # Inicializace modelů
    generator = Generator(in_channels=disc_in_channels, out_channels=1, feature_maps=args.feature_maps).to(device)
    discriminator = Discriminator(in_channels=disc_in_channels, feature_maps=args.feature_maps).to(device)
    
    # Inicializace optimizerů
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    # Inicializace loss funkcí
    adversarial_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    focal_loss = FocalLoss(alpha=0.8, gamma=2.0)
    
    # Tréninkový cyklus
    best_val_dice = 0.0
    
    for epoch in range(args.epochs):
        # Tréninkový mód
        generator.train()
        discriminator.train()
        
        # Metriky
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        train_metrics = {
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'iou': 0.0
        }
        
        # Training loop
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in train_loader:
                # Příprava dat
                real_labels = batch['label'].to(device)
                normal_atlas = batch['normal_atlas'].to(device)
                
                # Příprava lesion_atlas, pokud je k dispozici
                lesion_atlas = batch.get('lesion_atlas', None)
                if lesion_atlas is not None:
                    lesion_atlas = lesion_atlas.to(device)
                
                batch_size = real_labels.size(0)
                
                # Vygenerování náhodného šumu
                noise = torch.randn(batch_size, 1, *real_labels.shape[2:]).to(device)
                
                # ---------------------
                #  Trénink Diskriminátoru
                # ---------------------
                
                d_optimizer.zero_grad()
                
                # Vstup pro diskriminátor - skutečné léze
                real_outputs = discriminator(normal_atlas, real_labels, lesion_atlas)
                
                # Generování falešných lézí
                if lesion_atlas is not None:
                    # Spojení atlasu, šumu a atlasu lézí
                    gen_input = torch.cat([normal_atlas, noise, lesion_atlas], dim=1)
                    fake_labels = generator(gen_input, lesion_atlas)
                else:
                    # Spojení atlasu a šumu
                    gen_input = torch.cat([normal_atlas, noise], dim=1)
                    fake_labels = generator(gen_input)
                
                # Vstup pro diskriminátor - falešné léze
                fake_outputs = discriminator(normal_atlas, fake_labels.detach(), lesion_atlas)
                
                # Výpočet lossu diskriminátoru
                d_real_loss = adversarial_loss(real_outputs, torch.ones_like(real_outputs))
                d_fake_loss = adversarial_loss(fake_outputs, torch.zeros_like(fake_outputs))
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                # Backpropagation pro diskriminátor
                d_loss.backward()
                d_optimizer.step()
                
                # ---------------------
                #  Trénink Generátoru
                # ---------------------
                
                g_optimizer.zero_grad()
                
                # Generování falešných lézí (znovu, protože detach)
                if lesion_atlas is not None:
                    # Spojení atlasu, šumu a atlasu lézí
                    gen_input = torch.cat([normal_atlas, noise, lesion_atlas], dim=1)
                    fake_labels = generator(gen_input, lesion_atlas)
                else:
                    # Spojení atlasu a šumu
                    gen_input = torch.cat([normal_atlas, noise], dim=1)
                    fake_labels = generator(gen_input)
                
                # Diskriminátor klasifikuje falešné léze
                fake_outputs = discriminator(normal_atlas, fake_labels, lesion_atlas)
                
                # Výpočet lossu generátoru
                g_adv_loss = adversarial_loss(fake_outputs, torch.ones_like(fake_outputs))
                g_dice_loss = dice_loss(fake_labels, real_labels)
                g_focal_loss = focal_loss(fake_labels, real_labels)
                
                # Kombinace lossů
                g_loss = g_adv_loss + args.dice_weight * g_dice_loss + args.focal_weight * g_focal_loss
                
                # Backpropagation pro generátor
                g_loss.backward()
                g_optimizer.step()
                
                # Aktualizace metrik
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                
                # Výpočet metrik pro tento batch
                with torch.no_grad():
                    # Použití prahování pro binární masku
                    fake_np = fake_labels.detach().cpu().numpy()
                    real_np = real_labels.detach().cpu().numpy()
                    
                    for i in range(batch_size):
                        metrics = calculate_metrics(fake_np[i, 0], real_np[i, 0])
                        for key in train_metrics:
                            train_metrics[key] += metrics[key] / batch_size / len(train_loader)
                
                # Aktualizace progress baru
                pbar.update(1)
                pbar.set_postfix({
                    'g_loss': g_loss.item(),
                    'd_loss': d_loss.item(),
                    'dice': train_metrics['dice'] * len(train_loader) / (pbar.n + 1e-8)
                })
        
        # Validace
        generator.eval()
        discriminator.eval()
        
        val_metrics = {
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'iou': 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # Příprava dat
                real_labels = batch['label'].to(device)
                normal_atlas = batch['normal_atlas'].to(device)
                
                # Příprava lesion_atlas, pokud je k dispozici
                lesion_atlas = batch.get('lesion_atlas', None)
                if lesion_atlas is not None:
                    lesion_atlas = lesion_atlas.to(device)
                
                batch_size = real_labels.size(0)
                
                # Vygenerování náhodného šumu
                noise = torch.randn(batch_size, 1, *real_labels.shape[2:]).to(device)
                
                # Generování falešných lézí
                if lesion_atlas is not None:
                    # Spojení atlasu, šumu a atlasu lézí
                    gen_input = torch.cat([normal_atlas, noise, lesion_atlas], dim=1)
                    fake_labels = generator(gen_input, lesion_atlas)
                else:
                    # Spojení atlasu a šumu
                    gen_input = torch.cat([normal_atlas, noise], dim=1)
                    fake_labels = generator(gen_input)
                
                # Výpočet metrik pro tento batch
                fake_np = fake_labels.detach().cpu().numpy()
                real_np = real_labels.detach().cpu().numpy()
                
                for i in range(batch_size):
                    metrics = calculate_metrics(fake_np[i, 0], real_np[i, 0])
                    for key in val_metrics:
                        val_metrics[key] += metrics[key] / batch_size / len(val_loader)
        
        # Výpis metrik
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train - G Loss: {epoch_g_loss/len(train_loader):.4f}, D Loss: {epoch_d_loss/len(train_loader):.4f}")
        print(f"Train - Dice: {train_metrics['dice']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}")
        print(f"Val - Dice: {val_metrics['dice']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        # Uložení vzorků
        if (epoch + 1) % args.sample_interval == 0:
            generator.eval()
            with torch.no_grad():
                # Výběr vzorku z validačního setu
                sample_batch = next(iter(val_loader))
                real_labels = sample_batch['label'].to(device)
                normal_atlas = sample_batch['normal_atlas'].to(device)
                
                # Příprava lesion_atlas, pokud je k dispozici
                lesion_atlas = sample_batch.get('lesion_atlas', None)
                if lesion_atlas is not None:
                    lesion_atlas = lesion_atlas.to(device)
                
                # Vygenerování náhodného šumu
                noise = torch.randn(1, 1, *real_labels.shape[2:]).to(device)
                
                # Generování falešných lézí
                if lesion_atlas is not None:
                    # Spojení atlasu, šumu a atlasu lézí
                    gen_input = torch.cat([normal_atlas[0:1], noise, lesion_atlas[0:1]], dim=1)
                    fake_label = generator(gen_input, lesion_atlas[0:1])
                else:
                    # Spojení atlasu a šumu
                    gen_input = torch.cat([normal_atlas[0:1], noise], dim=1)
                    fake_label = generator(gen_input)
                
                # Uložení vzorku
                sample_path = os.path.join(sample_dir, f"sample_epoch_{epoch+1}")
                
                # Převod na numpy a uložení
                fake_np = fake_label[0, 0].cpu().numpy()
                real_np = real_labels[0, 0].cpu().numpy()
                atlas_np = normal_atlas[0, 0].cpu().numpy()
                
                # Binarizace výstupu
                fake_binary = (fake_np > 0.5).astype(np.float32)
                
                # Uložení jako NIFTI
                save_nifti(fake_np, f"{sample_path}_raw.nii.gz")
                save_nifti(fake_binary, f"{sample_path}_binary.nii.gz")
                save_nifti(real_np, f"{sample_path}_real.nii.gz")
                save_nifti(atlas_np, f"{sample_path}_atlas.nii.gz")
        
        # Uložení modelů
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            # Uložení nejlepšího modelu
            torch.save(generator.state_dict(), os.path.join(model_dir, "best_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, "best_discriminator.pth"))
            print(f"Uložen nejlepší model s Dice: {best_val_dice:.4f}")
        
        # Uložení checkpointu
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(generator.state_dict(), os.path.join(model_dir, f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator_epoch_{epoch+1}.pth"))
    
    # Uložení finálního modelu
    torch.save(generator.state_dict(), os.path.join(model_dir, "final_generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(model_dir, "final_discriminator.pth"))
    print("Trénink LabelGAN dokončen!")

# Funkce pro generování syntetických lézí pomocí natrénovaného modelu
def generate_lesions(args):
    """
    Generuje syntetické léze pomocí natrénovaného generátoru
    
    Args:
        args: Argumenty příkazové řádky
    """
    # Nastavení reprodukovatelnosti
    set_seed(args.seed)
    
    # Určení zařízení
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Použité zařízení: {device}")
    
    # Vytvoření výstupního adresáře
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Načtení normativního atlasu
    if args.normal_atlas_path.endswith('.nii.gz') or args.normal_atlas_path.endswith('.nii'):
        normal_atlas_data = nib.load(args.normal_atlas_path)
        normal_atlas = normal_atlas_data.get_fdata()
    else:
        normal_atlas_sitk = sitk.ReadImage(args.normal_atlas_path)
        normal_atlas = sitk.GetArrayFromImage(normal_atlas_sitk)
        normal_atlas_data = None  # Nemáme nibabel objekt pro metadata
    
    # Normalizace atlasu na rozsah [0, 1]
    normal_atlas = (normal_atlas - normal_atlas.min()) / (normal_atlas.max() - normal_atlas.min())
    
    # Načtení atlasu frekvence lézí, pokud je dostupný
    lesion_atlas = None
    lesion_atlas_data = None
    
    if args.lesion_atlas_path:
        if args.lesion_atlas_path.endswith('.nii.gz') or args.lesion_atlas_path.endswith('.nii'):
            lesion_atlas_data = nib.load(args.lesion_atlas_path)
            lesion_atlas = lesion_atlas_data.get_fdata()
        else:
            lesion_atlas_sitk = sitk.ReadImage(args.lesion_atlas_path)
            lesion_atlas = sitk.GetArrayFromImage(lesion_atlas_sitk)
        
        # Normalizace atlasu lézí na rozsah [0, 1]
        lesion_atlas = (lesion_atlas - lesion_atlas.min()) / (lesion_atlas.max() - lesion_atlas.min())
    
    # Načtení modelu generátoru
    # Konfigurace vstupních kanálů
    if args.lesion_atlas_path:
        in_channels = 3  # atlas + šum + lesion atlas
    else:
        in_channels = 2  # atlas + šum
    
    # Inicializace modelu
    generator = Generator(in_channels=in_channels, out_channels=1, feature_maps=args.feature_maps).to(device)
    
    # Načtení vah
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.eval()
    
    # Nastavení modelu do eval módu
    generator.eval()
    
    # Konverze atlasů na tensory
    normal_atlas_tensor = torch.FloatTensor(normal_atlas).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D, H, W]
    
    if lesion_atlas is not None:
        lesion_atlas_tensor = torch.FloatTensor(lesion_atlas).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D, H, W]
    
    # Generování zadaného počtu syntetických lézí
    with torch.no_grad():
        for i in tqdm(range(args.num_samples), desc="Generuji léze"):
            # Vygenerování náhodného šumu
            noise = torch.randn(1, 1, *normal_atlas.shape).to(device)
            
            # Generování léze
            if lesion_atlas is not None:
                # Spojení atlasu, šumu a atlasu lézí
                gen_input = torch.cat([normal_atlas_tensor, noise, lesion_atlas_tensor], dim=1)
                fake_label = generator(gen_input, lesion_atlas_tensor)
            else:
                # Spojení atlasu a šumu
                gen_input = torch.cat([normal_atlas_tensor, noise], dim=1)
                fake_label = generator(gen_input)
            
            # Převod na numpy
            fake_np = fake_label[0, 0].cpu().numpy()
            
            # Binarizace výstupu (použití prahu)
            if args.threshold:
                fake_binary = (fake_np > args.threshold).astype(np.float32)
            else:
                # Použití adaptivního prahu, který zachová přibližně stejné procento lézí jako v trénovacích datech
                # Typický rozsah je 1-5% voxelů
                sorted_values = np.sort(fake_np.flatten())
                threshold_index = int(0.95 * len(sorted_values))  # Assuming approximately 5% of voxels are lesions
                adaptive_threshold = sorted_values[threshold_index]
                fake_binary = (fake_np > adaptive_threshold).astype(np.float32)
            
            # Uložení jako NIFTI
            output_path = os.path.join(args.output_dir, f"synthetic_lesion_{i+1:04d}.nii.gz")
            
            if normal_atlas_data is not None:
                save_nifti(fake_binary, output_path, normal_atlas_data)
            else:
                save_nifti(fake_binary, output_path)
    
    print(f"Vygenerováno {args.num_samples} syntetických lézí do adresáře: {args.output_dir}")

# Hlavní funkce
def main():
    parser = argparse.ArgumentParser(description='LabelGAN pro generování syntetických lézí u HIE')
    
    # Sdílené argumenty
    parser.add_argument('--normal_atlas_path', type=str, required=True,
                        help='Cesta k normativnímu atlasu')
    parser.add_argument('--lesion_atlas_path', type=str, default=None,
                        help='Cesta k atlasu frekvence lézí (volitelné)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Výstupní adresář')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed pro generátor náhodných čísel')
    parser.add_argument('--feature_maps', type=int, default=64,
                        help='Počet základních feature map v modelu')
    
    # Podparsery pro různé módy
    subparsers = parser.add_subparsers(dest='mode', help='Mód operace')
    
    # Parser pro trénink
    train_parser = subparsers.add_parser('train', help='Trénování LabelGAN')
    train_parser.add_argument('--label_dir', type=str, required=True,
                             help='Adresář s registrovanými LABEL mapami')
    train_parser.add_argument('--batch_size', type=int, default=4,
                             help='Velikost dávky pro trénování')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Počet epoch pro trénování')
    train_parser.add_argument('--lr', type=float, default=0.0002,
                             help='Learning rate')
    train_parser.add_argument('--beta1', type=float, default=0.5,
                             help='Beta1 parametr pro Adam optimizer')
    train_parser.add_argument('--val_split', type=float, default=0.2,
                             help='Poměr dat pro validaci (0.0-1.0)')
    train_parser.add_argument('--shuffle', type=bool, default=True,
                             help='Zamíchat data před rozdělením')
    train_parser.add_argument('--num_workers', type=int, default=4,
                             help='Počet worker procesů pro načítání dat')
    train_parser.add_argument('--sample_interval', type=int, default=10,
                             help='Interval epoch pro ukládání vzorků')
    train_parser.add_argument('--checkpoint_interval', type=int, default=25,
                             help='Interval epoch pro ukládání checkpointů')
    train_parser.add_argument('--dice_weight', type=float, default=10.0,
                             help='Váha pro Dice Loss')
    train_parser.add_argument('--focal_weight', type=float, default=5.0,
                             help='Váha pro Focal Loss')
    
    # Parser pro generování
    generate_parser = subparsers.add_parser('generate', help='Generování syntetických lézí')
    generate_parser.add_argument('--model_path', type=str, required=True,
                                help='Cesta k natrénovanému generátoru')
    generate_parser.add_argument('--num_samples', type=int, default=10,
                                help='Počet syntetických lézí k vygenerování')
    generate_parser.add_argument('--threshold', type=float, default=None,
                                help='Práh pro binarizaci výstupu (0.0-1.0, None pro adaptivní práh)')
    
    # Parsování argumentů
    args = parser.parse_args()
    
    # Spuštění podle módu
    if args.mode == 'train':
        train_labelgan(args)
    elif args.mode == 'generate':
        generate_lesions(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
