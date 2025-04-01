import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
import torch
from medpy.metric.binary import dc, hd95
from skimage.metrics import structural_similarity as ssim

def load_image(image_path):
    """Načte obraz z cesty a vrátí numpy array"""
    if str(image_path).endswith('.nii.gz'):
        return nib.load(str(image_path)).get_fdata()
    else:
        return sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))

def dice_coefficient(y_true, y_pred):
    """Vypočítá Dice koeficient mezi dvěma binárními maskami"""
    return dc(y_pred, y_true)

def hausdorff_distance(y_true, y_pred):
    """Vypočítá 95% Hausdorff Distance mezi dvěma binárními maskami"""
    return hd95(y_pred, y_true)

def evaluate_lesion_synthesis(real_image_path, synthetic_image_path, real_label_path, synthetic_label_path):
    """
    Vyhodnotí kvalitu syntetizovaných lézí pomocí různých metrik
    
    Args:
        real_image_path: Cesta k reálnému obrazu
        synthetic_image_path: Cesta k syntetickému obrazu
        real_label_path: Cesta k reálné label mapě
        synthetic_label_path: Cesta k syntetické label mapě
        
    Returns:
        Dictionary s hodnotami metrik
    """
    # Načtení obrazů
    real_image = load_image(real_image_path)
    synthetic_image = load_image(synthetic_image_path)
    real_label = load_image(real_label_path).astype(bool)
    synthetic_label = load_image(synthetic_label_path).astype(bool)
    
    # Výpočet metrik pro obrazy
    image_ssim = ssim(real_image, synthetic_image, 
                     data_range=max(real_image.max(), synthetic_image.max()) - 
                              min(real_image.min(), synthetic_image.min()))
    
    # Výpočet metrik pro label mapy
    dice = dice_coefficient(real_label, synthetic_label)
    
    # Hausdorff distance pouze pokud jsou v obou maskách léze
    if np.any(real_label) and np.any(synthetic_label):
        hausdorff = hausdorff_distance(real_label, synthetic_label)
    else:
        hausdorff = float('inf')
    
    return {
        'image_ssim': image_ssim,
        'dice': dice,
        'hausdorff': hausdorff
    }

def visualize_comparison(real_image, synthetic_image, real_label, synthetic_label, slice_index=None, output_path=None):
    """
    Vytvoří vizualizaci porovnání reálného a syntetického obrazu s lézemi
    
    Args:
        real_image: Reálný obraz jako numpy array
        synthetic_image: Syntetický obraz jako numpy array
        real_label: Reálná label mapa jako numpy array
        synthetic_label: Syntetická label mapa jako numpy array
        slice_index: Index řezu pro vizualizaci (pokud None, bude vybrán automaticky)
        output_path: Cesta pro uložení vizualizace (pokud None, bude zobrazena interaktivně)
    """
    # Pokud není zadán index řezu, vyber řez s největším počtem lézí
    if slice_index is None:
        if np.any(real_label):
            slice_indices = np.sum(real_label, axis=(1, 2))
            slice_index = np.argmax(slice_indices)
        else:
            slice_index = real_image.shape[0] // 2
    
    # Vytvoření obrázku
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reálný obraz
    axes[0, 0].imshow(real_image[slice_index], cmap='gray')
    axes[0, 0].set_title('Reálný obraz')
    axes[0, 0].axis('off')
    
    # Syntetický obraz
    axes[0, 1].imshow(synthetic_image[slice_index], cmap='gray')
    axes[0, 1].set_title('Syntetický obraz')
    axes[0, 1].axis('off')
    
    # Rozdíl obrazů
    diff = np.abs(real_image[slice_index] - synthetic_image[slice_index])
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Rozdíl obrazů')
    axes[0, 2].axis('off')
    
    # Reálná label mapa
    axes[1, 0].imshow(real_image[slice_index], cmap='gray')
    if np.any(real_label):
        real_mask = np.ma.masked_where(~real_label[slice_index], real_label[slice_index])
        axes[1, 0].imshow(real_mask, cmap='autumn', alpha=0.7)
    axes[1, 0].set_title('Reálné léze')
    axes[1, 0].axis('off')
    
    # Syntetická label mapa
    axes[1, 1].imshow(synthetic_image[slice_index], cmap='gray')
    if np.any(synthetic_label):
        synthetic_mask = np.ma.masked_where(~synthetic_label[slice_index], synthetic_label[slice_index])
        axes[1, 1].imshow(synthetic_mask, cmap='winter', alpha=0.7)
    axes[1, 1].set_title('Syntetické léze')
    axes[1, 1].axis('off')
    
    # Překryv lézí
    axes[1, 2].imshow(real_image[slice_index], cmap='gray')
    if np.any(real_label) or np.any(synthetic_label):
        overlap = np.zeros_like(real_label[slice_index], dtype=np.uint8)
        overlap[real_label[slice_index] & synthetic_label[slice_index]] = 3  # překryv
        overlap[real_label[slice_index] & ~synthetic_label[slice_index]] = 1  # pouze reálné
        overlap[~real_label[slice_index] & synthetic_label[slice_index]] = 2  # pouze syntetické
        
        overlap_mask = np.ma.masked_where(overlap == 0, overlap)
        axes[1, 2].imshow(overlap_mask, cmap='viridis', alpha=0.7)
    axes[1, 2].set_title('Překryv lézí')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def evaluate_all_samples(args):
    """Vyhodnotí všechny vzorky v adresářích"""
    # Získání seznamu souborů
    synthetic_images = sorted(Path(args.synthetic_image_dir).glob('*.nii.gz'))
    synthetic_labels = sorted(Path(args.synthetic_label_dir).glob('*.nii.gz'))
    real_images = sorted(Path(args.real_image_dir).glob('*.nii.gz'))
    real_labels = sorted(Path(args.real_label_dir).glob('*.nii.gz'))
    
    # Kontrola počtu souborů
    min_count = min(len(synthetic_images), len(synthetic_labels), len(real_images), len(real_labels))
    
    if min_count == 0:
        print("Nenalezeny žádné soubory pro vyhodnocení!")
        return
    
    # Vytvoření výstupního adresáře
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Souhrnné statistiky
    all_metrics = []
    
    # Vyhodnocení každého vzorku
    for i in range(min_count):
        print(f"Vyhodnocení vzorku {i+1}/{min_count}")
        
        # Načtení obrazů
        real_image = load_image(real_images[i])
        synthetic_image = load_image(synthetic_images[i])
        real_label = load_image(real_labels[i]).astype(bool)
        synthetic_label = load_image(synthetic_labels[i]).astype(bool)
        
        # Výpočet metrik
        metrics = evaluate_lesion_synthesis(
            real_images[i], synthetic_images[i], real_labels[i], synthetic_labels[i]
        )
        all_metrics.append(metrics)
        
        # Vytvoření vizualizace
        output_path = output_dir / f"comparison_{i}.png"
        visualize_comparison(
            real_image, synthetic_image, real_label, synthetic_label,
            output_path=output_path
        )
        
        # Výpis metrik
        print(f"  SSIM: {metrics['image_ssim']:.4f}, Dice: {metrics['dice']:.4f}, HD95: {metrics['hausdorff']:.4f}")
    
    # Výpočet průměrných metrik
    avg_metrics = {
        'image_ssim': np.mean([m['image_ssim'] for m in all_metrics]),
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'hausdorff': np.mean([m['hausdorff'] for m in all_metrics if m['hausdorff'] != float('inf')])
    }
    
    # Uložení souhrnných výsledků
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Počet vyhodnocených vzorků: {min_count}\n")
        f.write(f"Průměrné hodnoty metrik:\n")
        f.write(f"  SSIM: {avg_metrics['image_ssim']:.4f}\n")
        f.write(f"  Dice: {avg_metrics['dice']:.4f}\n")
        f.write(f"  HD95: {avg_metrics['hausdorff']:.4f}\n")
    
    print("\nSouhrnné výsledky:")
    print(f"Průměrný SSIM: {avg_metrics['image_ssim']:.4f}")
    print(f"Průměrný Dice: {avg_metrics['dice']:.4f}")
    print(f"Průměrný HD95: {avg_metrics['hausdorff']:.4f}")
    print(f"Výsledky uloženy v: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Vyhodnocení kvality syntetizovaných lézí")
    
    parser.add_argument('--real_image_dir', type=str, required=True,
                        help='Adresář s reálnými obrazy')
    parser.add_argument('--synthetic_image_dir', type=str, required=True,
                        help='Adresář se syntetickými obrazy')
    parser.add_argument('--real_label_dir', type=str, required=True,
                        help='Adresář s reálnými label mapami')
    parser.add_argument('--synthetic_label_dir', type=str, required=True,
                        help='Adresář se syntetickými label mapami')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Výstupní adresář pro uložení výsledků')
    
    args = parser.parse_args()
    
    evaluate_all_samples(args)

if __name__ == "__main__":
    main() 