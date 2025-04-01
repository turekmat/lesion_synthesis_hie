import os
import argparse
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pathlib import Path
import ants
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

def register_image_to_atlas(moving_image_path, fixed_image_path, output_path):
    """
    Registruje vstupní obraz na atlas pomocí ANTs (pokročilá registrace)
    
    Args:
        moving_image_path: Cesta k obrazu, který bude registrován
        fixed_image_path: Cesta k obrazu atlasu
        output_path: Cesta pro uložení registrovaného obrazu
    """
    print(f"Registering {moving_image_path} to {fixed_image_path}")
    
    # Načtení obrazů pomocí ANTsPy
    fixed_ant = ants.image_read(str(fixed_image_path))
    moving_ant = ants.image_read(str(moving_image_path))
    # Načtení původního obrazu (bez normalizace) pro získání cílového rozsahu
    orig_moving_ant = ants.image_read(str(moving_image_path))
    
    # Normalizace intenzity vstupních obrazů pro lepší registraci
    fixed_ant = ants.iMath(fixed_ant, "Normalize")
    moving_ant = ants.iMath(moving_ant, "Normalize")
    
    try:
        # Dvoustupňová registrace: Nejprve afinní a poté SyN (nelineární)
        print("Provádím afinní registraci...")
        init_reg = ants.registration(
            fixed=fixed_ant, 
            moving=moving_ant,
            type_of_transform='Affine', 
            verbose=False
        )
        
        print("Provádím SyN registraci...")
        reg = ants.registration(
            fixed=fixed_ant, 
            moving=moving_ant,
            type_of_transform='SyN',
            initial_transform=init_reg['fwdtransforms'][0],
            reg_iterations=[50, 30, 20],   # Počet iterací pro každou úroveň
            verbose=False
        )
        
        # Aplikace transformace na pohyblivý obraz
        registered_image = ants.apply_transforms(
            fixed=fixed_ant,
            moving=moving_ant,
            transformlist=reg['fwdtransforms'],
            interpolator='linear',
            verbose=False
        )
        
        # Vytvoření masky mozku z atlasu
        print("Vytvářím masku mozku z atlasu...")
        brain_mask = ants.threshold_image(fixed_ant, 0.05, 2.0)  # Vytvoření binární masky pro hodnoty > 0.05
        
        # Aplikace masky na registrovaný obraz
        print("Aplikuji masku na registrovaný obraz...")
        masked_registered_image = registered_image * brain_mask
        
        # Získání původního dynamického rozsahu z původního obrazu
        orig_moving_np = orig_moving_ant.numpy()
        desired_min, desired_max = orig_moving_np.min(), orig_moving_np.max()
        # Vypočítáme statistiky původního obrazu
        orig_avg = orig_moving_np.mean()
        orig_zero_pct = np.sum(orig_moving_np == 0) / orig_moving_np.size * 100
        print(f"Originální image statistiky: min={desired_min}, max={desired_max}, avg={orig_avg}, %0={orig_zero_pct:.2f}%")
        
        # Zpětné škálování intenzit na cílový rozsah (z původního obrazu)
        print("Aplikuji zpětné škálování intenzit na registrovaný obraz...")
        masked_np = masked_registered_image.numpy()
        mask = masked_np != 0  # škálujeme pouze pixely uvnitř masky
        if np.any(mask):
            current_min = masked_np[mask].min()
            current_max = masked_np[mask].max()
        else:
            current_min, current_max = 0, 1  # záložní hodnoty, pokud je maska prázdná
        
        scaled_np = np.copy(masked_np)
        scaled_np[mask] = (masked_np[mask] - current_min) / (current_max - current_min) * (desired_max - desired_min) + desired_min
        
        # Vytvoření nového ANTs image ze škálovaného numpy pole a zachování geometrie původního obrazu
        scaled_registered_image = ants.from_numpy(scaled_np)
        scaled_registered_image.set_origin(masked_registered_image.origin)
        scaled_registered_image.set_spacing(masked_registered_image.spacing)
        scaled_registered_image.set_direction(masked_registered_image.direction)
        
        # Vypočítáme statistiky registrovaného obrazu
        scaled_avg = scaled_np.mean()
        scaled_zero_pct = np.sum(scaled_np == 0) / scaled_np.size * 100
        scaled_min = scaled_np.min()
        scaled_max = scaled_np.max()
        print(f"Registrovaný image statistiky: min={scaled_min}, max={scaled_max}, avg={scaled_avg}, %0={scaled_zero_pct:.2f}%")
        
        # Uložení registrovaného obrazu se škálováním
        print(f"Ukládám registrovaný obraz (se zpětným škálováním) do {output_path}")
        ants.image_write(scaled_registered_image, str(output_path))
        
        # Vrácení transformačních parametrů a referencí (vracíme škálovanou verzi)
        return {
            'fwdtransforms': reg['fwdtransforms'],
            'invtransforms': reg['invtransforms']
        }, fixed_ant, moving_ant, scaled_registered_image
        
    except Exception as e:
        print(f"Chyba při registraci: {e}")
        raise

def transform_label_map(label_map_path, transforms, fixed_image, output_path):
    """
    Aplikuje transformaci na label mapu
    
    Args:
        label_map_path: Cesta k label mapě
        transforms: Transformační parametry získané z registrace
        fixed_image: Referenční obraz v ANTs formátu
        output_path: Cesta pro uložení transformované label mapy
    """
    print(f"Transforming label map {label_map_path}")
    
    # Načtení label mapy pomocí ANTsPy
    label_map_ant = ants.image_read(str(label_map_path))
    
    # Aplikace transformace na label mapu s nearest neighbor interpolací pro zachování diskrétních hodnot
    transformed_label = ants.apply_transforms(
        fixed=fixed_image,
        moving=label_map_ant,
        transformlist=transforms['fwdtransforms'],
        interpolator='nearestNeighbor'
    )
    
    # Uložení transformované label mapy
    print(f"Ukládám transformovanou label mapu do {output_path}")
    ants.image_write(transformed_label, str(output_path))
    
    return label_map_ant, transformed_label

def create_registration_visualization_pdf(atlas_image, orig_zadc, reg_zadc, orig_label, reg_label, output_pdf_path):
    """
    Vytvoří PDF vizualizaci registrace s 5 sloupci: atlas, původní ZADC, registrovaná ZADC,
    původní label, registrovaná label.
    
    Args:
        atlas_image: Normativní atlas (ANTs image)
        orig_zadc: Původní ZADC mapa (ANTs image)
        reg_zadc: Registrovaná ZADC mapa (ANTs image)
        orig_label: Původní label mapa (ANTs image)
        reg_label: Registrovaná label mapa (ANTs image)
        output_pdf_path: Cesta pro uložení PDF vizualizace
    """
    print(f"Vytvářím vizualizaci registrace do PDF: {output_pdf_path}")
    
    # Konverze ANTs obrazů na numpy pole
    atlas_np = atlas_image.numpy()
    orig_zadc_np = orig_zadc.numpy()
    reg_zadc_np = reg_zadc.numpy()
    orig_label_np = orig_label.numpy()
    reg_label_np = reg_label.numpy()
    
    # Vytvoření binární masky z atlasu (nenulové hodnoty)
    brain_mask = (atlas_np > 0.05)
    
    # Aplikace masky na registrovanou ZADC mapu - nastavíme hodnoty mimo masku na 0
    masked_reg_zadc_np = np.copy(reg_zadc_np)
    for slice_idx in range(masked_reg_zadc_np.shape[2]):
        if slice_idx < brain_mask.shape[2]:
            masked_reg_zadc_np[:, :, slice_idx] = masked_reg_zadc_np[:, :, slice_idx] * brain_mask[:, :, slice_idx]
    
    # Zjištění počtu řezů pro každý obraz
    atlas_slices = atlas_np.shape[2]
    orig_zadc_slices = orig_zadc_np.shape[2]
    reg_zadc_slices = masked_reg_zadc_np.shape[2]
    orig_label_slices = orig_label_np.shape[2]
    reg_label_slices = reg_label_np.shape[2]
    
    print(f"Počet řezů - Atlas: {atlas_slices}, Původní ZADC: {orig_zadc_slices}, Reg ZADC: {reg_zadc_slices}, " +
          f"Původní Label: {orig_label_slices}, Reg Label: {reg_label_slices}")
    
    # Vytvoříme PDF
    with PdfPages(output_pdf_path) as pdf:
        max_slices = max(atlas_slices, orig_zadc_slices, reg_zadc_slices, orig_label_slices, reg_label_slices)
        
        for slice_idx in range(max_slices):
            plt.figure(figsize=(20, 5))
            gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1])
            
            ax1 = plt.subplot(gs[0])
            if slice_idx < atlas_slices:
                atlas_slice = atlas_np[:, :, slice_idx] if atlas_np.ndim == 3 else atlas_np[:, :, slice_idx, 0]
                ax1.imshow(atlas_slice, cmap='gray')
            ax1.set_title(f'Atlas (řez {slice_idx+1})')
            ax1.axis('off')
            
            ax2 = plt.subplot(gs[1])
            if slice_idx < orig_zadc_slices:
                orig_zadc_slice = orig_zadc_np[:, :, slice_idx] if orig_zadc_np.ndim == 3 else orig_zadc_np[:, :, slice_idx, 0]
                ax2.imshow(orig_zadc_slice, cmap='gray')
            ax2.set_title(f'Původní ZADC (řez {slice_idx+1})')
            ax2.axis('off')
            
            ax3 = plt.subplot(gs[2])
            if slice_idx < reg_zadc_slices:
                reg_zadc_slice = masked_reg_zadc_np[:, :, slice_idx] if masked_reg_zadc_np.ndim == 3 else masked_reg_zadc_np[:, :, slice_idx, 0]
                ax3.imshow(reg_zadc_slice, cmap='gray')
            ax3.set_title(f'Registrovaná ZADC (řez {slice_idx+1})')
            ax3.axis('off')
            
            ax4 = plt.subplot(gs[3])
            if slice_idx < orig_label_slices:
                orig_label_slice = orig_label_np[:, :, slice_idx] if orig_label_np.ndim == 3 else orig_label_np[:, :, slice_idx, 0]
                ax4.imshow(orig_label_slice, cmap='hot')
            ax4.set_title(f'Původní Label (řez {slice_idx+1})')
            ax4.axis('off')
            
            ax5 = plt.subplot(gs[4])
            if slice_idx < reg_label_slices:
                reg_label_slice = reg_label_np[:, :, slice_idx] if reg_label_np.ndim == 3 else reg_label_np[:, :, slice_idx, 0]
                ax5.imshow(reg_label_slice, cmap='hot')
            ax5.set_title(f'Registrovaná Label (řez {slice_idx+1})')
            ax5.axis('off')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    
    print(f"PDF vizualizace uložena do: {output_pdf_path}")

def process_dataset(args):
    """
    Zpracuje celý dataset registrací na normativní atlas
    
    Args:
        args: Argumenty příkazové řádky
    """
    zadc_output_dir = Path(args.output_dir) / "registered_zadc"
    label_output_dir = Path(args.output_dir) / "registered_label"
    
    if args.pdf_viz_registration:
        pdf_dir = Path(args.output_dir) / "registration_visualizations"
        pdf_dir.mkdir(parents=True, exist_ok=True)
    
    zadc_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)
    
    zadc_files = sorted([f for f in Path(args.zadc_dir).glob("*") if f.suffix in ['.mha', '.nii.gz', '.nii']])
    label_files = sorted([f for f in Path(args.label_dir).glob("*") if f.suffix in ['.mha', '.nii.gz', '.nii']])
    
    if len(zadc_files) != len(label_files):
        print(f"Varování: Rozdílný počet ZADC ({len(zadc_files)}) a LABEL ({len(label_files)}) souborů!")
    
    for i, (zadc_path, label_path) in enumerate(zip(zadc_files, label_files)):
        print(f"\n=== Zpracování {i+1}/{len(zadc_files)}: {zadc_path.name} ===")
        zadc_output_path = zadc_output_dir / f"reg_{zadc_path.name}"
        label_output_path = label_output_dir / f"reg_{label_path.name}"
        
        if args.output_format == 'nii.gz':
            zadc_output_path = zadc_output_path.with_suffix('.nii.gz')
            label_output_path = label_output_path.with_suffix('.nii.gz')
        
        try:
            transforms, fixed_ref, orig_zadc, reg_zadc = register_image_to_atlas(
                zadc_path,
                args.normal_atlas_path,
                zadc_output_path
            )
            
            orig_label, reg_label = transform_label_map(
                label_path,
                transforms,
                fixed_ref,
                label_output_path
            )
            
            print(f"Hotovo: {zadc_output_path} a {label_output_path}")
            
            if args.pdf_viz_registration:
                pdf_path = pdf_dir / f"registration_viz_{zadc_path.stem}.pdf"
                create_registration_visualization_pdf(
                    atlas_image=fixed_ref,
                    orig_zadc=orig_zadc,
                    reg_zadc=reg_zadc,
                    orig_label=orig_label,
                    reg_label=reg_label,
                    output_pdf_path=str(pdf_path)
                )
            
        except Exception as e:
            print(f"Zpracování selhalo pro pár {zadc_path.name} a {label_path.name}: {e}")
            print("Pokračuji s dalším párem...")
            continue

def main():
    parser = argparse.ArgumentParser(description="Registrace ZADC a LABEL map na normativní atlas")
    
    parser.add_argument('--normal_atlas_path', type=str, required=True,
                        help='Cesta k normativnímu atlasu')
    parser.add_argument('--zadc_dir', type=str, required=True,
                        help='Adresář s ZADC mapami')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Adresář s LABEL mapami')
    parser.add_argument('--output_dir', type=str, default='./registered_data',
                        help='Výstupní adresář pro registrovaná data')
    parser.add_argument('--output_format', type=str, choices=['original', 'nii.gz'], default='nii.gz',
                        help='Formát výstupních souborů')
    parser.add_argument('--pdf_viz_registration', action='store_true',
                        help='Vytvoří PDF vizualizaci registračních výsledků')
    
    args = parser.parse_args()
    
    paths_to_check = [
        ("Normativní atlas", args.normal_atlas_path),
        ("Adresář ZADC", args.zadc_dir),
        ("Adresář LABEL", args.label_dir)
    ]
    
    all_paths_exist = True
    for name, path in paths_to_check:
        if not Path(path).exists():
            print(f"CHYBA: {name} neexistuje: {path}")
            all_paths_exist = False
    
    if not all_paths_exist:
        print("Některé cesty neexistují. Ukončuji script.")
        return
    
    process_dataset(args)

if __name__ == "__main__":
    main()
