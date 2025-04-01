import os
import argparse
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib

def load_image(image_path):
    """Načte obraz z cesty a vrátí numpy array"""
    if str(image_path).endswith('.nii.gz'):
        return nib.load(str(image_path)).get_fdata()
    else:
        return sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))

def visualize_registration_results(original_path, registered_path, label_path=None, output_path=None, num_slices=3):
    """Vizualizuje výsledky registrace porovnáním původního a registrovaného obrazu"""
    original_img = load_image(original_path)
    registered_img = load_image(registered_path)
    
    # Normalizace dat pro vizualizaci
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
    registered_img = (registered_img - registered_img.min()) / (registered_img.max() - registered_img.min())
    
    # Výběr řezů pro vizualizaci
    depth = original_img.shape[0]
    slice_indices = [int(depth * i / (num_slices + 1)) for i in range(1, num_slices + 1)]
    
    # Vykreslení porovnání
    if label_path:
        label_img = load_image(label_path).astype(bool)
        fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5 * num_slices))
        
        for i, slice_idx in enumerate(slice_indices):
            # Původní obraz
            axes[i, 0].imshow(original_img[slice_idx], cmap='gray')
            axes[i, 0].set_title(f'Původní obraz (řez {slice_idx})')
            axes[i, 0].axis('off')
            
            # Registrovaný obraz
            axes[i, 1].imshow(registered_img[slice_idx], cmap='gray')
            axes[i, 1].set_title(f'Registrovaný obraz (řez {slice_idx})')
            axes[i, 1].axis('off')
            
            # Registrovaný obraz s překryvem label mapy
            axes[i, 2].imshow(registered_img[slice_idx], cmap='gray')
            if slice_idx < label_img.shape[0]:
                mask = np.ma.masked_where(~label_img[slice_idx], label_img[slice_idx])
                axes[i, 2].imshow(mask, cmap='autumn', alpha=0.7)
            axes[i, 2].set_title(f'Registrovaný obraz s lézemi (řez {slice_idx})')
            axes[i, 2].axis('off')
    else:
        fig, axes = plt.subplots(num_slices, 2, figsize=(10, 5 * num_slices))
        
        for i, slice_idx in enumerate(slice_indices):
            # Původní obraz
            axes[i, 0].imshow(original_img[slice_idx], cmap='gray')
            axes[i, 0].set_title(f'Původní obraz (řez {slice_idx})')
            axes[i, 0].axis('off')
            
            # Registrovaný obraz
            axes[i, 1].imshow(registered_img[slice_idx], cmap='gray')
            axes[i, 1].set_title(f'Registrovaný obraz (řez {slice_idx})')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Vizualizace uložena do: {output_path}")
    else:
        plt.show()

def visualize_gan_results(normal_atlas_path, synthetic_image_path, real_image_path=None, output_path=None, num_slices=3):
    """Vizualizuje výsledky GAN porovnáním normativního, syntetického a reálného obrazu"""
    normal_atlas = load_image(normal_atlas_path)
    synthetic_img = load_image(synthetic_image_path)
    
    # Normalizace dat pro vizualizaci
    normal_atlas = (normal_atlas - normal_atlas.min()) / (normal_atlas.max() - normal_atlas.min())
    synthetic_img = (synthetic_img - synthetic_img.min()) / (synthetic_img.max() - synthetic_img.min())
    
    # Výběr řezů pro vizualizaci
    depth = normal_atlas.shape[0]
    slice_indices = [int(depth * i / (num_slices + 1)) for i in range(1, num_slices + 1)]
    
    # Nastavení velikosti subplotů podle toho, zda máme reálný obraz
    if real_image_path:
        real_img = load_image(real_image_path)
        real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min())
        fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5 * num_slices))
        
        for i, slice_idx in enumerate(slice_indices):
            # Normativní atlas
            axes[i, 0].imshow(normal_atlas[slice_idx], cmap='gray')
            axes[i, 0].set_title(f'Normativní atlas (řez {slice_idx})')
            axes[i, 0].axis('off')
            
            # Syntetický obraz
            axes[i, 1].imshow(synthetic_img[slice_idx], cmap='gray')
            axes[i, 1].set_title(f'Syntetický obraz (řez {slice_idx})')
            axes[i, 1].axis('off')
            
            # Reálný obraz
            axes[i, 2].imshow(real_img[slice_idx], cmap='gray')
            axes[i, 2].set_title(f'Reálný obraz (řez {slice_idx})')
            axes[i, 2].axis('off')
    else:
        fig, axes = plt.subplots(num_slices, 2, figsize=(10, 5 * num_slices))
        
        for i, slice_idx in enumerate(slice_indices):
            # Normativní atlas
            axes[i, 0].imshow(normal_atlas[slice_idx], cmap='gray')
            axes[i, 0].set_title(f'Normativní atlas (řez {slice_idx})')
            axes[i, 0].axis('off')
            
            # Syntetický obraz
            axes[i, 1].imshow(synthetic_img[slice_idx], cmap='gray')
            axes[i, 1].set_title(f'Syntetický obraz (řez {slice_idx})')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Vizualizace uložena do: {output_path}")
    else:
        plt.show()

def visualize_segmentation_results(image_path, segmentation_path, output_path=None, num_slices=3):
    """Vizualizuje výsledky segmentace porovnáním vstupního obrazu a segmentace"""
    image = load_image(image_path)
    segmentation = load_image(segmentation_path).astype(bool)
    
    # Normalizace dat pro vizualizaci
    image = (image - image.min()) / (image.max() - image.min())
    
    # Výběr řezů pro vizualizaci - preferujeme řezy s největším počtem segmentovaných pixelů
    depth = image.shape[0]
    if np.any(segmentation):
        # Spočítáme počet segmentovaných pixelů v každém řezu
        seg_pixels_per_slice = np.sum(segmentation, axis=(1, 2))
        # Vybereme řezy s nejvíce segmentovanými pixely
        slice_indices = np.argsort(seg_pixels_per_slice)[-num_slices:]
    else:
        # Pokud segmentace neobsahuje žádné pixely, použijeme rovnoměrně rozložené řezy
        slice_indices = [int(depth * i / (num_slices + 1)) for i in range(1, num_slices + 1)]
    
    # Vykreslení porovnání
    fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5 * num_slices))
    
    for i, slice_idx in enumerate(slice_indices):
        # Vstupní obraz
        axes[i, 0].imshow(image[slice_idx], cmap='gray')
        axes[i, 0].set_title(f'Vstupní obraz (řez {slice_idx})')
        axes[i, 0].axis('off')
        
        # Segmentace
        if slice_idx < segmentation.shape[0]:
            axes[i, 1].imshow(segmentation[slice_idx], cmap='hot')
            axes[i, 1].set_title(f'Segmentace (řez {slice_idx})')
            axes[i, 1].axis('off')
        
        # Překryv
        axes[i, 2].imshow(image[slice_idx], cmap='gray')
        if slice_idx < segmentation.shape[0]:
            mask = np.ma.masked_where(~segmentation[slice_idx], segmentation[slice_idx])
            axes[i, 2].imshow(mask, cmap='autumn', alpha=0.7)
        axes[i, 2].set_title(f'Překryv (řez {slice_idx})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Vizualizace uložena do: {output_path}")
    else:
        plt.show()

def run_command(cmd):
    """Spustí příkaz a vypíše výstup"""
    print(f"Spouštím: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode != 0:
        print(f"Příkaz selhal s návratovým kódem {process.returncode}")
        return False
    
    return True

def run_pipeline(args):
    """Spustí celou pipeline generování a segmentace lézí"""
    
    # Vytvoření adresářů
    registered_data_dir = Path(args.output_dir) / "registered_data"
    gan_output_dir = Path(args.output_dir) / "gan_output"
    synthetic_samples_dir = Path(args.output_dir) / "synthetic_samples"
    synthetic_labels_dir = Path(args.output_dir) / "synthetic_labels"
    segmentation_model_dir = Path(args.output_dir) / "segmentation_model"
    segmentation_results_dir = Path(args.output_dir) / "segmentation_results"
    
    # Vytvoření adresáře pro vizualizace, pokud je požadováno
    if args.visualize:
        visualization_dir = Path(args.output_dir) / "visualizations"
        visualization_dir.mkdir(parents=True, exist_ok=True)
    
    for dir_path in [registered_data_dir, gan_output_dir, synthetic_samples_dir, 
                   synthetic_labels_dir, segmentation_model_dir, segmentation_results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Registrace dat na normativní atlas
    if args.run_registration:
        print("\n=== KROK 1: Registrace dat na normativní atlas ===\n")
        
        registration_cmd = [
            "python", "registration_tool.py",
            "--normal_atlas_path", args.normal_atlas_path,
            "--zadc_dir", args.zadc_dir,
            "--label_dir", args.label_dir,
            "--output_dir", str(registered_data_dir),
            "--output_format", "nii.gz"
        ]
        
        if not run_command(registration_cmd):
            print("Registrace selhala. Ukončuji pipeline.")
            return
        
        # Vizualizace výsledků registrace
        if args.visualize:
            print("\nVytvářím vizualizaci výsledků registrace...")
            zadc_files = sorted(list(Path(args.zadc_dir).glob("*.[mn][hi][aa]*")))
            registered_zadc_files = sorted(list((registered_data_dir / "registered_zadc").glob("*.nii.gz")))
            registered_label_files = sorted(list((registered_data_dir / "registered_label").glob("*.nii.gz")))
            
            if zadc_files and registered_zadc_files:
                # Vybereme první soubor pro vizualizaci
                original_path = zadc_files[0]
                registered_path = registered_zadc_files[0]
                label_path = registered_label_files[0] if registered_label_files else None
                
                vis_output_path = visualization_dir / "registration_results.png"
                visualize_registration_results(
                    original_path, registered_path, label_path, vis_output_path, num_slices=args.vis_slices
                )
    
    # 2. Trénink GAN modelu
    if args.run_gan_training:
        print("\n=== KROK 2: Trénink GAN modelu ===\n")
        
        gan_train_cmd = [
            "python", "hie_lesion_synthesis.py", "train",
            "--normal_atlas_path", args.normal_atlas_path,
            "--zadc_dir", str(registered_data_dir / "registered_zadc"),
            "--label_dir", str(registered_data_dir / "registered_label"),
            "--output_dir", str(gan_output_dir),
            "--epochs", str(args.gan_epochs),
            "--batch_size", str(args.gan_batch_size),
            "--lr", str(args.gan_lr)
        ]
        
        if args.lesion_atlas_path:
            gan_train_cmd.extend(["--lesion_atlas_path", args.lesion_atlas_path])
        
        if not run_command(gan_train_cmd):
            print("Trénink GAN modelu selhal. Ukončuji pipeline.")
            return
        
        # Vizualizace výsledků GAN tréninku
        if args.visualize:
            print("\nVytvářím vizualizaci výsledků GAN tréninku...")
            # Hledáme poslední vygenerovaný vzorek
            sample_files = sorted(list(gan_output_dir.glob("sample_epoch*_fake.nii.gz")))
            atlas_files = sorted(list(gan_output_dir.glob("sample_epoch*_atlas.nii.gz")))
            real_files = sorted(list(gan_output_dir.glob("sample_epoch*_real.nii.gz")))
            
            if sample_files and atlas_files:
                # Vybereme poslední soubory pro vizualizaci
                synthetic_path = sample_files[-1]
                atlas_path = atlas_files[-1]
                real_path = real_files[-1] if real_files else None
                
                vis_output_path = visualization_dir / "gan_training_results.png"
                visualize_gan_results(
                    atlas_path, synthetic_path, real_path, vis_output_path, num_slices=args.vis_slices
                )
    
    # 3. Generování syntetických vzorků
    if args.run_gan_generation:
        print("\n=== KROK 3: Generování syntetických vzorků ===\n")
        
        # Nalezení posledního checkpointu
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
        else:
            checkpoints = list(gan_output_dir.glob("checkpoint_epoch*.pt"))
            if not checkpoints:
                print("Nenalezeny žádné checkpointy. Přeskakuji generování vzorků.")
                return
            else:
                # Výběr posledního checkpointu
                checkpoint_path = str(sorted(checkpoints, key=lambda x: int(x.stem.split("epoch")[1]))[-1])
        
        gan_generate_cmd = [
            "python", "hie_lesion_synthesis.py", "generate",
            "--normal_atlas_path", args.normal_atlas_path,
            "--output_dir", str(synthetic_samples_dir),
            "--checkpoint_path", checkpoint_path,
            "--num_samples", str(args.num_synthetic_samples)
        ]
        
        if args.lesion_atlas_path:
            gan_generate_cmd.extend(["--lesion_atlas_path", args.lesion_atlas_path])
        
        if not run_command(gan_generate_cmd):
            print("Generování syntetických vzorků selhalo. Ukončuji pipeline.")
            return
        
        # Vizualizace vygenerovaných vzorků
        if args.visualize:
            print("\nVytvářím vizualizaci vygenerovaných vzorků...")
            synthetic_files = sorted(list(synthetic_samples_dir.glob("sample_*.nii.gz")))
            
            if synthetic_files and len(synthetic_files) > 0:
                # Vybereme několik vzorků pro vizualizaci
                num_samples_to_visualize = min(3, len(synthetic_files))
                sample_indices = list(range(0, len(synthetic_files), len(synthetic_files) // num_samples_to_visualize))[:num_samples_to_visualize]
                
                for i, idx in enumerate(sample_indices):
                    synthetic_path = synthetic_files[idx]
                    vis_output_path = visualization_dir / f"synthetic_sample_{i}.png"
                    visualize_gan_results(
                        args.normal_atlas_path, synthetic_path, output_path=vis_output_path, num_slices=args.vis_slices
                    )
    
    # 4. Trénink segmentačního modelu
    if args.run_segmentation_training:
        print("\n=== KROK 4: Trénink segmentačního modelu ===\n")
        
        # Rozdělení dat na trénovací a validační
        import random
        import shutil
        
        real_image_files = list((registered_data_dir / "registered_zadc").glob("*.nii.gz"))
        real_label_files = list((registered_data_dir / "registered_label").glob("*.nii.gz"))
        
        # Vytvoření adresářů pro rozdělená data
        train_image_dir = registered_data_dir / "train_images"
        train_label_dir = registered_data_dir / "train_labels"
        val_image_dir = registered_data_dir / "val_images"
        val_label_dir = registered_data_dir / "val_labels"
        
        for dir_path in [train_image_dir, train_label_dir, val_image_dir, val_label_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Náhodné rozdělení na trénovací a validační sady (80/20)
        indices = list(range(len(real_image_files)))
        random.shuffle(indices)
        
        split_idx = int(len(indices) * 0.8)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Kopírování souborů do příslušných adresářů
        for idx in train_indices:
            shutil.copy(real_image_files[idx], train_image_dir / real_image_files[idx].name)
            shutil.copy(real_label_files[idx], train_label_dir / real_label_files[idx].name)
        
        for idx in val_indices:
            shutil.copy(real_image_files[idx], val_image_dir / real_image_files[idx].name)
            shutil.copy(real_label_files[idx], val_label_dir / real_label_files[idx].name)
        
        # Spuštění tréninku segmentačního modelu
        segment_train_cmd = [
            "python", "segmentation_model.py", "train",
            "--train_image_dir", str(train_image_dir),
            "--train_label_dir", str(train_label_dir),
            "--val_image_dir", str(val_image_dir),
            "--val_label_dir", str(val_label_dir),
            "--output_dir", str(segmentation_model_dir),
            "--epochs", str(args.segmentation_epochs),
            "--batch_size", str(args.segmentation_batch_size),
            "--lr", str(args.segmentation_lr),
            "--features", str(args.segmentation_features),
            "--num_workers", str(args.num_workers)
        ]
        
        # Přidání syntetických dat, pokud jsou k dispozici
        if args.use_synthetic_data and args.run_gan_generation:
            segment_train_cmd.extend([
                "--synthetic_image_dir", str(synthetic_samples_dir),
                "--synthetic_label_dir", str(synthetic_labels_dir)
            ])
        
        if args.use_augmentation:
            segment_train_cmd.append("--use_augmentation")
        
        if not run_command(segment_train_cmd):
            print("Trénink segmentačního modelu selhal. Ukončuji pipeline.")
            return
    
    # 5. Segmentace testovacích dat
    if args.run_segmentation_prediction and args.test_image_dir:
        print("\n=== KROK 5: Segmentace testovacích dat ===\n")
        
        # Nalezení nejlepšího modelu
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = str(segmentation_model_dir / "best_model.pt")
            if not Path(model_path).exists():
                model_path = str(segmentation_model_dir / "final_model.pt")
            if not Path(model_path).exists():
                print("Nenalezen žádný natrénovaný model. Přeskakuji segmentaci.")
                return
        
        segment_predict_cmd = [
            "python", "segmentation_model.py", "predict",
            "--input_dir", args.test_image_dir,
            "--output_dir", str(segmentation_results_dir),
            "--model_path", model_path,
            "--threshold", str(args.threshold),
            "--features", str(args.segmentation_features)
        ]
        
        if not run_command(segment_predict_cmd):
            print("Segmentace testovacích dat selhala.")
            return
        
        # Vizualizace výsledků segmentace
        if args.visualize:
            print("\nVytvářím vizualizaci výsledků segmentace...")
            test_files = sorted(list(Path(args.test_image_dir).glob("*.[mn][hi][aa]*")))
            segmentation_files = sorted(list(segmentation_results_dir.glob("seg_*.[mn][hi][aa]*")))
            
            if test_files and segmentation_files and len(test_files) == len(segmentation_files):
                # Vybereme několik vzorků pro vizualizaci
                num_samples_to_visualize = min(3, len(test_files))
                sample_indices = list(range(0, len(test_files), max(1, len(test_files) // num_samples_to_visualize)))[:num_samples_to_visualize]
                
                for i, idx in enumerate(sample_indices):
                    image_path = test_files[idx]
                    segmentation_path = segmentation_files[idx]
                    vis_output_path = visualization_dir / f"segmentation_result_{i}.png"
                    visualize_segmentation_results(
                        image_path, segmentation_path, vis_output_path, num_slices=args.vis_slices
                    )
    
    print("\n=== Pipeline dokončena úspěšně! ===\n")
    print(f"Výsledky jsou k dispozici v adresáři: {args.output_dir}")
    
    if args.visualize:
        print(f"Vizualizace jsou k dispozici v adresáři: {visualization_dir}")

def construct_path(data_root, relative_path):
    """Složí cestu z kořenového adresáře a relativní cesty"""
    if data_root and not Path(relative_path).is_absolute():
        return str(Path(data_root) / relative_path)
    return relative_path

def main():
    parser = argparse.ArgumentParser(description="HIE Lesion Synthesis and Segmentation Pipeline")
    
    # Parametry pro datové cesty
    data_path_group = parser.add_argument_group('Parametry datových cest')
    data_path_group.add_argument('--data_root', type=str, default=None,
                        help='Kořenový adresář s daty (bude předpona pro všechny cesty)')
    data_path_group.add_argument('--bonbid_data_dir', type=str, default="data/BONBID2023_Train",
                        help='Adresář s BONBID daty')
    data_path_group.add_argument('--normal_atlases_dir', type=str, default="data/archive/normal_atlases",
                        help='Adresář s normativními atlasy')
    data_path_group.add_argument('--lesion_atlases_dir', type=str, default="data/archive/lesion_atlases",
                        help='Adresář s lézemi atlasy')
    data_path_group.add_argument('--normal_atlas_name', type=str, default="atlas_week0-1_masked.nii.gz",
                        help='Název souboru normativního atlasu')
    data_path_group.add_argument('--lesion_atlas_name', type=str, default="lesion_atlas.nii.gz",
                        help='Název souboru atlasu léze')
    
    # Hlavní parametry
    parser.add_argument('--normal_atlas_path', type=str, default=None,
                        help='Cesta k normativnímu atlasu (přepíše --normal_atlases_dir a --normal_atlas_name)')
    parser.add_argument('--lesion_atlas_path', type=str, default=None,
                        help='Cesta k atlasu frekvence lézí (přepíše --lesion_atlases_dir a --lesion_atlas_name)')
    parser.add_argument('--zadc_dir', type=str, default=None,
                        help='Adresář s ZADC mapami (přepíše --bonbid_data_dir)')
    parser.add_argument('--label_dir', type=str, default=None,
                        help='Adresář s LABEL mapami (přepíše --bonbid_data_dir)')
    parser.add_argument('--test_image_dir', type=str, default=None,
                        help='Adresář s testovacími obrazy pro segmentaci')
    parser.add_argument('--output_dir', type=str, default="./pipeline_output",
                        help='Výstupní adresář pro všechny výsledky')
    
    # Parametry pro spouštění jednotlivých kroků
    steps_group = parser.add_argument_group('Parametry kroků pipeline')
    steps_group.add_argument('--run_registration', action='store_true',
                        help='Spustit registraci dat')
    steps_group.add_argument('--run_gan_training', action='store_true',
                        help='Spustit trénink GAN modelu')
    steps_group.add_argument('--run_gan_generation', action='store_true',
                        help='Spustit generování syntetických vzorků')
    steps_group.add_argument('--run_segmentation_training', action='store_true',
                        help='Spustit trénink segmentačního modelu')
    steps_group.add_argument('--run_segmentation_prediction', action='store_true',
                        help='Spustit segmentaci testovacích dat')
    steps_group.add_argument('--run_all', action='store_true',
                        help='Spustit všechny kroky pipeline')
    
    # Parametry pro vizualizaci
    vis_group = parser.add_argument_group('Parametry vizualizace')
    vis_group.add_argument('--visualize', action='store_true',
                        help='Zapnout vizualizaci výsledků po krocích')
    vis_group.add_argument('--vis_slices', type=int, default=3,
                        help='Počet řezů pro vizualizaci')
    
    # Parametry GAN modelu
    gan_group = parser.add_argument_group('Parametry GAN modelu')
    gan_group.add_argument('--gan_epochs', type=int, default=100,
                        help='Počet epoch pro trénink GAN modelu')
    gan_group.add_argument('--gan_batch_size', type=int, default=2,
                        help='Velikost dávky pro trénink GAN modelu')
    gan_group.add_argument('--gan_lr', type=float, default=0.0002,
                        help='Learning rate pro trénink GAN modelu')
    gan_group.add_argument('--checkpoint_path', type=str, default=None,
                        help='Cesta ke konkrétnímu checkpointu GAN modelu')
    gan_group.add_argument('--num_synthetic_samples', type=int, default=20,
                        help='Počet syntetických vzorků k vygenerování')
    
    # Parametry segmentačního modelu
    seg_group = parser.add_argument_group('Parametry segmentačního modelu')
    seg_group.add_argument('--segmentation_epochs', type=int, default=100,
                        help='Počet epoch pro trénink segmentačního modelu')
    seg_group.add_argument('--segmentation_batch_size', type=int, default=2,
                        help='Velikost dávky pro trénink segmentačního modelu')
    seg_group.add_argument('--segmentation_lr', type=float, default=0.0001,
                        help='Learning rate pro trénink segmentačního modelu')
    seg_group.add_argument('--segmentation_features', type=int, default=32,
                        help='Počet základních feature map v UNet modelu')
    seg_group.add_argument('--use_synthetic_data', action='store_true',
                        help='Použít syntetická data pro trénink segmentace')
    seg_group.add_argument('--use_augmentation', action='store_true',
                        help='Použít augmentaci dat pro trénink segmentace')
    seg_group.add_argument('--model_path', type=str, default=None,
                        help='Cesta ke konkrétnímu segmentačnímu modelu')
    seg_group.add_argument('--threshold', type=float, default=0.5,
                        help='Práh pro binarizaci segmentace')
    seg_group.add_argument('--num_workers', type=int, default=4,
                        help='Počet worker procesů pro DataLoader')
    
    args = parser.parse_args()
    
    # Sestavení cest na základě kořenového adresáře a jmen souborů
    if not args.normal_atlas_path:
        args.normal_atlas_path = construct_path(args.data_root, 
                                            Path(args.normal_atlases_dir) / args.normal_atlas_name)
    
    if not args.lesion_atlas_path:
        args.lesion_atlas_path = construct_path(args.data_root, 
                                             Path(args.lesion_atlases_dir) / args.lesion_atlas_name)
    
    if not args.zadc_dir:
        args.zadc_dir = construct_path(args.data_root, 
                                    Path(args.bonbid_data_dir) / "2Z_ADC")
    
    if not args.label_dir:
        args.label_dir = construct_path(args.data_root, 
                                    Path(args.bonbid_data_dir) / "3LABEL")
    
    # Pokud je zvolená možnost run_all, spustíme všechny kroky
    if args.run_all:
        args.run_registration = True
        args.run_gan_training = True
        args.run_gan_generation = True
        args.run_segmentation_training = True
        args.run_segmentation_prediction = True
    
    # Výpis sestavených cest
    print("\n=== Použité cesty ===")
    print(f"Normativní atlas: {args.normal_atlas_path}")
    print(f"Atlas lézí: {args.lesion_atlas_path}")
    print(f"Adresář ZADC: {args.zadc_dir}")
    print(f"Adresář LABEL: {args.label_dir}")
    print(f"Výstupní adresář: {args.output_dir}")
    if args.test_image_dir:
        print(f"Testovací adresář: {args.test_image_dir}")
    
    # Kontrola existence cest
    paths_to_check = [
        ("Normativní atlas", args.normal_atlas_path),
        ("Atlas lézí", args.lesion_atlas_path),
        ("Adresář ZADC", args.zadc_dir),
        ("Adresář LABEL", args.label_dir)
    ]
    if args.test_image_dir:
        paths_to_check.append(("Testovací adresář", args.test_image_dir))
    
    all_paths_exist = True
    for name, path in paths_to_check:
        if not Path(path).exists():
            print(f"CHYBA: {name} neexistuje: {path}")
            all_paths_exist = False
    
    if not all_paths_exist:
        print("Některé cesty neexistují. Ukončuji pipeline.")
        return
    
    run_pipeline(args)

if __name__ == "__main__":
    main() 