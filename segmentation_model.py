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
import matplotlib.pyplot as plt
from tqdm import tqdm

class HIESegmentationDataset(Dataset):
    """Dataset pro segmentaci HIE lézí"""
    
    def __init__(self, image_dir, label_dir, transform=None, use_augmentation=False):
        """
        Args:
            image_dir: Adresář s obrazovými daty (ZADC mapy)
            label_dir: Adresář s label mapami
            transform: Volitelné transformace
            use_augmentation: Použít augmentaci dat
        """
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                                 if f.endswith('.mha') or f.endswith('.nii.gz')])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                                  if f.endswith('.mha') or f.endswith('.nii.gz')])
        self.transform = transform
        self.use_augmentation = use_augmentation
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Načtení obrazu a label mapy
        if self.image_files[idx].endswith('.nii.gz'):
            image = nib.load(self.image_files[idx]).get_fdata()
            label = nib.load(self.label_files[idx]).get_fdata()
        else:
            image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_files[idx]))
            label = sitk.GetArrayFromImage(sitk.ReadImage(self.label_files[idx]))
            
        # Normalizace do rozsahu [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        # Binarizace label mapy
        label = (label > 0).astype(np.float32)
        
        # Převod na PyTorch tensory
        image = torch.FloatTensor(image).unsqueeze(0)  # Přidání kanálové dimenze
        label = torch.FloatTensor(label).unsqueeze(0)
        
        # Aplikace transformací
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            
        # Augmentace dat (pokud je povolena)
        if self.use_augmentation:
            image, label = self._augment(image, label)
            
        return {
            'image': image,
            'label': label
        }
    
    def _augment(self, image, label):
        """Jednoduchá augmentace dat - náhodné rotace a překlopení"""
        # Náhodné rotace
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)  # 1, 2, nebo 3 rotace o 90°
            image = torch.rot90(image, k, dims=[2, 3])
            label = torch.rot90(label, k, dims=[2, 3])
        
        # Náhodné překlopení
        if np.random.rand() > 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
        if np.random.rand() > 0.5:
            image = torch.flip(image, dims=[3])
            label = torch.flip(label, dims=[3])
            
        return image, label

class UNet3D(nn.Module):
    """3D U-Net model pro segmentaci lézí"""
    
    def __init__(self, in_channels=1, out_channels=1, features=32):
        """
        Args:
            in_channels: Počet vstupních kanálů
            out_channels: Počet výstupních kanálů
            features: Počet základních feature map
        """
        super(UNet3D, self).__init__()
        
        # Encoder
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features*2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features*2, features*4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features*4, features*8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features*8, features*16, name="bottleneck")
        
        # Decoder
        self.upconv4 = nn.ConvTranspose3d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features*16, features*8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose3d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features*8, features*4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose3d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features*4, features*2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose3d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features*2, features, name="dec1")
        
        # Výstupní vrstva
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Výstupní vrstva
        output = self.conv(dec1)
        return torch.sigmoid(output)
    
    def _block(self, in_channels, features, name):
        """Helper funkce pro vytvoření dvojice konvolučních vrstev"""
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )

def dice_loss(pred, target):
    """Dice Loss pro segmentační úlohy"""
    smooth = 1.0
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return 1 - ((2. * intersection + smooth) / 
                (pred_flat.sum() + target_flat.sum() + smooth))

def train_segmentation_model(args):
    """Hlavní trénovací funkce pro segmentační model"""
    
    # Nastavení zařízení
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Příprava datasetů
    train_dataset = HIESegmentationDataset(
        image_dir=args.train_image_dir,
        label_dir=args.train_label_dir,
        use_augmentation=args.use_augmentation
    )
    
    # Přidání syntetických dat (pokud jsou k dispozici)
    if args.synthetic_image_dir and args.synthetic_label_dir:
        synthetic_dataset = HIESegmentationDataset(
            image_dir=args.synthetic_image_dir,
            label_dir=args.synthetic_label_dir,
            use_augmentation=False  # Syntetická data už jsou augmentovaná
        )
        
        # Kombinace reálných a syntetických dat
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([train_dataset, synthetic_dataset])
        print(f"Trénink na {len(train_dataset)} vzorcích (reálná + syntetická data)")
    else:
        print(f"Trénink pouze na {len(train_dataset)} reálných vzorcích")
    
    # Vytvoření dataloaderů
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Validační dataset (pokud je k dispozici)
    if args.val_image_dir and args.val_label_dir:
        val_dataset = HIESegmentationDataset(
            image_dir=args.val_image_dir,
            label_dir=args.val_label_dir
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers
        )
        print(f"Validace na {len(val_dataset)} vzorcích")
    else:
        val_loader = None
        print("Validace nebude prováděna")
    
    # Inicializace modelu
    model = UNet3D(in_channels=1, out_channels=1, features=args.features).to(device)
    
    # Optimizátor
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )
    
    # Trénovací cyklus
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Trénink
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Train)"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Výpočet loss
            loss = dice_loss(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Průměrná ztráta za epochu
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validace
        if val_loader:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Val)"):
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(images)
                    loss = dice_loss(outputs, labels)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Aktualizace learning rate
            scheduler.step(val_loss)
            
            # Ukládání nejlepšího modelu
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
                print(f"Uložen nový nejlepší model s validační ztrátou {val_loss:.4f}")
            
            print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
        else:
            # Bez validace ukládáme model každých n epoch
            if (epoch + 1) % 10 == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch{epoch+1}.pt'))
            
            print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}")
    
    # Uložení finálního modelu
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    
    # Vizualizace průběhu tréninku
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Průběh tréninku')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'training_curve.png'))
    
    print("Trénink dokončen!")

def predict_segmentation(args):
    """Segmentace pomocí natrénovaného modelu"""
    
    # Nastavení zařízení
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Inicializace a načtení modelu
    model = UNet3D(in_channels=1, out_channels=1, features=args.features).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Seznam souborů k segmentaci
    image_files = sorted([f for f in Path(args.input_dir).glob("*") if f.suffix in ['.mha', '.nii.gz']])
    
    if not image_files:
        print("Nenalezeny žádné vstupní soubory!")
        return
    
    # Vytvoření výstupního adresáře
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Segmentace každého obrazu
    for image_path in tqdm(image_files, desc="Segmentace"):
        # Načtení obrazu
        if image_path.suffix == '.nii.gz':
            image_obj = nib.load(str(image_path))
            image = image_obj.get_fdata()
            affine = image_obj.affine
        else:
            image_sitk = sitk.ReadImage(str(image_path))
            image = sitk.GetArrayFromImage(image_sitk)
            affine = None
        
        # Normalizace
        image = (image - image.min()) / (image.max() - image.min())
        
        # Převod na tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
        
        # Segmentace
        with torch.no_grad():
            output = model(image_tensor)
            
            # Prahování pro získání binární masky
            if args.threshold < 0:  # Automatický práh pomocí Otsu
                from skimage.filters import threshold_otsu
                pred_np = output.cpu().numpy()[0, 0]
                threshold = threshold_otsu(pred_np)
                pred_binary = (pred_np > threshold).astype(np.uint8)
            else:
                pred_binary = (output.cpu().numpy()[0, 0] > args.threshold).astype(np.uint8)
        
        # Uložení výsledku
        output_path = output_dir / f"seg_{image_path.name}"
        
        if image_path.suffix == '.nii.gz':
            # Uložení jako nifti
            output_nii = nib.Nifti1Image(pred_binary, affine)
            nib.save(output_nii, str(output_path))
        else:
            # Uložení jako mha
            output_sitk = sitk.GetImageFromArray(pred_binary)
            output_sitk.CopyInformation(image_sitk)
            sitk.WriteImage(output_sitk, str(output_path))
        
        print(f"Uložena segmentace: {output_path}")

def main():
    """Hlavní funkce skriptu"""
    parser = argparse.ArgumentParser(description="Segmentace HIE lézí")
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Parser pro trénink
    train_parser = subparsers.add_parser('train', help='Trénink segmentačního modelu')
    train_parser.add_argument('--train_image_dir', type=str, required=True,
                             help='Adresář s trénovacími obrazy')
    train_parser.add_argument('--train_label_dir', type=str, required=True,
                             help='Adresář s trénovacími label mapami')
    train_parser.add_argument('--val_image_dir', type=str, default=None,
                             help='Adresář s validačními obrazy')
    train_parser.add_argument('--val_label_dir', type=str, default=None,
                             help='Adresář s validačními label mapami')
    train_parser.add_argument('--synthetic_image_dir', type=str, default=None,
                             help='Adresář se syntetickými obrazy')
    train_parser.add_argument('--synthetic_label_dir', type=str, default=None,
                             help='Adresář se syntetickými label mapami')
    train_parser.add_argument('--output_dir', type=str, default='./segmentation_model',
                             help='Výstupní adresář pro uložení modelu')
    train_parser.add_argument('--batch_size', type=int, default=2,
                             help='Velikost dávky')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Počet epoch')
    train_parser.add_argument('--lr', type=float, default=0.0001,
                             help='Learning rate')
    train_parser.add_argument('--features', type=int, default=32,
                             help='Počet základních feature map v UNet')
    train_parser.add_argument('--use_augmentation', action='store_true',
                             help='Použít augmentaci dat')
    train_parser.add_argument('--num_workers', type=int, default=4,
                             help='Počet worker procesů pro DataLoader')
    
    # Parser pro predikci
    predict_parser = subparsers.add_parser('predict', help='Segmentace pomocí natrénovaného modelu')
    predict_parser.add_argument('--input_dir', type=str, required=True,
                               help='Adresář se vstupními obrazy')
    predict_parser.add_argument('--output_dir', type=str, default='./segmentation_results',
                               help='Výstupní adresář pro segmentace')
    predict_parser.add_argument('--model_path', type=str, required=True,
                               help='Cesta k natrénovanému modelu')
    predict_parser.add_argument('--threshold', type=float, default=0.5,
                               help='Práh pro binarizaci (hodnota < 0 použije Otsu)')
    predict_parser.add_argument('--features', type=int, default=32,
                               help='Počet základních feature map v UNet (musí odpovídat modelu)')
    
    args = parser.parse_args()
    
    if args.action == 'train':
        train_segmentation_model(args)
    elif args.action == 'predict':
        predict_segmentation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 