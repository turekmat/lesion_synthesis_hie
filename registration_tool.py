import os
import argparse
import ants
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pathlib import Path

def register_image_to_atlas(moving_image_path, fixed_image_path, output_path):
    """
    Registruje vstupní obraz na atlas pomocí ANTsPy s afinní a následnou SyN registrací.
    
    Args:
        moving_image_path: Cesta k obrazu, který bude registrován.
        fixed_image_path: Cesta k obrazu atlasu.
        output_path: Cesta pro uložení registrovaného obrazu.
        
    Returns:
        registration: Slovník s výsledky registrace (obsahuje transformace).
        fixed_ant: ANTsPy obraz atlasu (referenční obraz).
    """
    print(f"Registering {moving_image_path} to {fixed_image_path}")
    
    # Načtení obrazů pomocí ANTsPy
    moving_ant = ants.image_read(str(moving_image_path))
    fixed_ant = ants.image_read(str(fixed_image_path))
    
    try:
        # Nejprve afinní registrace
        init_reg = ants.registration(fixed=fixed_ant, moving=moving_ant,
                                     type_of_transform='Affine', verbose=True)
        # Následně SyN registrace s použitím výsledku afinní transformace jako počáteční hodnoty
        reg = ants.registration(fixed=fixed_ant, moving=moving_ant,
                                type_of_transform='SyN',
                                initial_transform=init_reg['fwdtransforms'][0],
                                verbose=True)
    except Exception as e:
        print(f"Registration failed for {moving_image_path}: {e}")
        return None, None

    # Aplikace získaných transformací na pohyblivý obraz
    warped_moving = ants.apply_transforms(fixed=fixed_ant, moving=moving_ant,
                                          transformlist=reg['fwdtransforms'],
                                          interpolator='linear')
    ants.image_write(warped_moving, str(output_path))
    
    return reg, fixed_ant

def transform_label_map(label_map_path, registration, reference_image, output_path):
    """
    Aplikuje transformaci (získanou pomocí ANTsPy) na label mapu.
    
    Args:
        label_map_path: Cesta k label mapě.
        registration: Výsledná registrace ze zvolené metody ANTsPy.
        reference_image: Referenční ANTsPy obraz (atlas).
        output_path: Cesta pro uložení transformované label mapy.
    """
    print(f"Transforming label map {label_map_path}")
    
    # Načtení label mapy pomocí ANTsPy
    label_map_ant = ants.image_read(str(label_map_path))
    
    # Aplikace stejné transformace na label mapu s použitím interpolátoru nearestNeighbor
    warped_label = ants.apply_transforms(fixed=reference_image, moving=label_map_ant,
                                         transformlist=registration['fwdtransforms'],
                                         interpolator='nearestNeighbor')
    
    ants.image_write(warped_label, str(output_path))

def process_dataset(args):
    """
    Zpracuje celý dataset registrací na normativní atlas pomocí ANTsPy.
    
    Args:
        args: Argumenty příkazové řádky.
    """
    # Vytvoření výstupních adresářů
    zadc_output_dir = Path(args.output_dir) / "registered_zadc"
    label_output_dir = Path(args.output_dir) / "registered_label"
    
    zadc_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Seznam souborů ZADC a odpovídajících LABEL
    zadc_files = sorted([f for f in Path(args.zadc_dir).glob("*") if f.suffix in ['.mha', '.nii.gz', '.nii']])
    label_files = sorted([f for f in Path(args.label_dir).glob("*") if f.suffix in ['.mha', '.nii.gz', '.nii']])
    
    # Kontrola, zda máme stejný počet ZADC a LABEL souborů
    if len(zadc_files) != len(label_files):
        print(f"Varování: Rozdílný počet ZADC ({len(zadc_files)}) a LABEL ({len(label_files)}) souborů!")
    
    # Načtení normativního atlasu (referenčního obrazu) pomocí ANTsPy
    reference_atlas = ants.image_read(str(args.normal_atlas_path))
    
    # Zpracování každého páru ZADC a LABEL
    for i, (zadc_path, label_path) in enumerate(zip(zadc_files, label_files)):
        print(f"Zpracování {i+1}/{len(zadc_files)}: {zadc_path.name}")
        
        # Generování názvů výstupních souborů
        zadc_output_path = zadc_output_dir / f"reg_{zadc_path.stem}"
        label_output_path = label_output_dir / f"reg_{label_path.stem}"
        
        # Změna přípony na .nii.gz, pokud je požadována
        if args.output_format == 'nii.gz':
            zadc_output_path = zadc_output_path.with_suffix('.nii.gz')
            label_output_path = label_output_path.with_suffix('.nii.gz')
        
        # Registrace ZADC mapy na atlas pomocí ANTsPy
        registration, registered_reference = register_image_to_atlas(
            zadc_path,
            args.normal_atlas_path,
            zadc_output_path
        )
        
        # Pokud registrace selhala, přeskočíme tento pár
        if registration is None:
            print(f"Registrace selhala pro {zadc_path.name}. Přeskakuji.")
            continue
        
        # Aplikace stejné transformace na LABEL mapu
        transform_label_map(
            label_path,
            registration,
            registered_reference,  # Použijeme referenční atlas z ANTsPy
            label_output_path
        )
        
        print(f"Hotovo: {zadc_output_path} a {label_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Registrace ZADC a LABEL map na normativní atlas pomocí ANTsPy")
    
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
    
    args = parser.parse_args()
    
    process_dataset(args)

if __name__ == "__main__":
    main()
