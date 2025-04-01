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
    if str(image_path).endswith('.nii.gz') or str(image_path).endswith('.nii'):
        return nib.load(str(image_path)).get_fdata()
    else:
        return sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))

def find_file_with_extensions(base_path, extensions=['.nii', '.nii.gz']):
    """Zkusí najít soubor s danou cestou a jednou z uvedených přípon"""
    base_path = Path(base_path)
    
    # Odstraníme případnou příponu z base_path
    for ext in extensions:
        if str(base_path).endswith(ext):
            base_path = Path(str(base_path)[:-len(ext)])
            break
    
    # Zkusíme jednotlivé přípony
    for ext in extensions:
        path_with_ext = Path(str(base_path) + ext)
        if path_with_ext.exists():
            return path_with_ext
    
    # Pokud nic nenajdeme, vrátíme původní cestu
    return base_path

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
    """Spustí pipeline pro syntézu HIE lézí pomocí GAN modelu"""
    
    # Vyhledání souborů s podporou pro oba typy přípon
    args.normal_atlas_path = str(find_file_with_extensions(args.normal_atlas_path))
    args.lesion_atlas_path = str(find_file_with_extensions(args.lesion_atlas_path))
    
    # Vytvoření adresářů
    registered_data_dir = Path(args.output_dir) / "registered_data"
    gan_output_dir = Path(args.output_dir) / "gan_output"
    synthetic_samples_dir = Path(args.output_dir) / "synthetic_samples"
    
    # Vytvoření adresáře pro vizualizace, pokud je požadováno
    if args.visualize:
        visualization_dir = Path(args.output_dir) / "visualizations"
        visualization_dir.mkdir(parents=True, exist_ok=True)
    
    for dir_path in [registered_data_dir, gan_output_dir, synthetic_samples_dir]:
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
            "--lr", str(args.gan_lr),
            "--latent_dim", str(args.gan_latent_dim),
            "--generator_filters", str(args.gan_generator_filters),
            "--discriminator_filters", str(args.gan_discriminator_filters),
            "--dropout_rate", str(args.gan_dropout_rate),
            "--beta1", str(args.gan_beta1),
            "--beta2", str(args.gan_beta2)
        ]
        
        if args.use_spectral_norm:
            gan_train_cmd.append("--use_spectral_norm")
        
        if args.use_self_attention:
            gan_train_cmd.append("--use_self_attention")
            
        if args.use_instance_noise:
            gan_train_cmd.append("--use_instance_noise")
        
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
            "--num_samples", str(args.num_synthetic_samples),
            "--latent_dim", str(args.gan_latent_dim),
            "--generator_filters", str(args.gan_generator_filters)
        ]
        
        if args.use_spectral_norm:
            gan_generate_cmd.append("--use_spectral_norm")
        
        if args.use_self_attention:
            gan_generate_cmd.append("--use_self_attention")
        
        if args.lesion_atlas_path:
            gan_generate_cmd.extend(["--lesion_atlas_path", args.lesion_atlas_path])
        
        if args.lesion_interpolation:
            gan_generate_cmd.append("--lesion_interpolation")
            
        if args.lesion_intensity_range:
            gan_generate_cmd.extend(["--lesion_intensity_range", args.lesion_intensity_range])
        
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
    parser = argparse.ArgumentParser(description="HIE Lesion Synthesis Pipeline")
    
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
    data_path_group.add_argument('--normal_atlas_name', type=str, default="atlas_week0-1_masked.nii",
                        help='Název souboru normativního atlasu')
    data_path_group.add_argument('--lesion_atlas_name', type=str, default="lesion_atlas.nii",
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
    gan_group.add_argument('--gan_latent_dim', type=int, default=128,
                        help='Velikost latentního prostoru generátoru')
    gan_group.add_argument('--gan_generator_filters', type=int, default=64,
                        help='Počáteční počet filtrů v generátoru')
    gan_group.add_argument('--gan_discriminator_filters', type=int, default=64,
                        help='Počáteční počet filtrů v diskriminátoru')
    gan_group.add_argument('--gan_dropout_rate', type=float, default=0.3,
                        help='Míra dropout v generátoru')
    gan_group.add_argument('--gan_beta1', type=float, default=0.5,
                        help='Beta1 parametr pro Adam optimizátor')
    gan_group.add_argument('--gan_beta2', type=float, default=0.999,
                        help='Beta2 parametr pro Adam optimizátor')
    gan_group.add_argument('--use_spectral_norm', action='store_true',
                        help='Použít spektrální normalizaci v diskriminátoru')
    gan_group.add_argument('--use_self_attention', action='store_true',
                        help='Použít self-attention vrstvy v generátoru a diskriminátoru')
    gan_group.add_argument('--use_instance_noise', action='store_true',
                        help='Použít instance noise pro stabilizaci tréninku GAN')
    
    # Parametry pro generování vzorků
    gen_group = parser.add_argument_group('Parametry generování vzorků')
    gen_group.add_argument('--checkpoint_path', type=str, default=None,
                        help='Cesta ke konkrétnímu checkpointu GAN modelu')
    gen_group.add_argument('--num_synthetic_samples', type=int, default=20,
                        help='Počet syntetických vzorků k vygenerování')
    gen_group.add_argument('--lesion_interpolation', action='store_true',
                        help='Generovat vzorky s interpolací mezi různými lézemi')
    gen_group.add_argument('--lesion_intensity_range', type=str, default=None,
                        help='Rozsah intenzity léze ve formátu "min,max" (např. "0.3,0.8")')
    
    # Nový parametr
    parser.add_argument('--pdf_viz_registration', action='store_true',
                        help='Vytvoří PDF vizualizaci registračních výsledků')
    
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
    
    # Výpis sestavených cest
    print("\n=== Použité cesty ===")
    print(f"Normativní atlas: {args.normal_atlas_path}")
    print(f"Atlas lézí: {args.lesion_atlas_path}")
    print(f"Adresář ZADC: {args.zadc_dir}")
    print(f"Adresář LABEL: {args.label_dir}")
    print(f"Výstupní adresář: {args.output_dir}")
    
    # Kontrola existence cest
    paths_to_check = [
        ("Normativní atlas", args.normal_atlas_path),
        ("Atlas lézí", args.lesion_atlas_path),
        ("Adresář ZADC", args.zadc_dir),
        ("Adresář LABEL", args.label_dir)
    ]
    
    all_paths_exist = True
    for name, path in paths_to_check:
        # Kontrola se synonymickými příponami (.nii nebo .nii.gz)
        if name in ["Normativní atlas", "Atlas lézí"]:
            path_obj = find_file_with_extensions(path)
            if not path_obj.exists():
                print(f"CHYBA: {name} neexistuje: {path}")
                print(f"       Zkusil jsem také hledat verzi s jinou příponou (.nii/.nii.gz).")
                all_paths_exist = False
        elif not Path(path).exists():
            print(f"CHYBA: {name} neexistuje: {path}")
            all_paths_exist = False
    
    if not all_paths_exist:
        print("Některé cesty neexistují. Ukončuji pipeline.")
        return
    
    run_pipeline(args)

if __name__ == "__main__":
    main() 