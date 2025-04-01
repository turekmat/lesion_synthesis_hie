import os
import argparse
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pathlib import Path
import ants

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
            verbose=True
        )
        
        print("Provádím SyN registraci...")
        reg = ants.registration(
            fixed=fixed_ant, 
            moving=moving_ant,
            type_of_transform='SyN',
            initial_transform=init_reg['fwdtransforms'][0],
            reg_iterations=[50, 30, 20],   # Počet iterací pro každou úroveň
            verbose=True
        )
        
        # Aplikace transformace na pohyblivý obraz
        registered_image = ants.apply_transforms(
            fixed=fixed_ant,
            moving=moving_ant,
            transformlist=reg['fwdtransforms'],
            interpolator='linear'
        )
        
        # Uložení registrovaného obrazu
        print(f"Ukládám registrovaný obraz do {output_path}")
        ants.image_write(registered_image, str(output_path))
        
        # Vrácení transformačních parametrů a referencí
        return {
            'fwdtransforms': reg['fwdtransforms'],
            'invtransforms': reg['invtransforms']
        }, fixed_ant
        
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

def process_dataset(args):
    """
    Zpracuje celý dataset registrací na normativní atlas
    
    Args:
        args: Argumenty příkazové řádky
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
    
    # Zpracování každého páru ZADC a LABEL
    for i, (zadc_path, label_path) in enumerate(zip(zadc_files, label_files)):
        print(f"\n=== Zpracování {i+1}/{len(zadc_files)}: {zadc_path.name} ===")
        
        # Generování názvů výstupních souborů
        zadc_output_path = zadc_output_dir / f"reg_{zadc_path.name}"
        label_output_path = label_output_dir / f"reg_{label_path.name}"
        
        # Změna přípony na .nii.gz, pokud je požadována
        if args.output_format == 'nii.gz':
            zadc_output_path = zadc_output_path.with_suffix('.nii.gz')
            label_output_path = label_output_path.with_suffix('.nii.gz')
        
        try:
            # Registrace ZADC na atlas
            transforms, fixed_ref = register_image_to_atlas(
                zadc_path,
                args.normal_atlas_path,
                zadc_output_path
            )
            
            # Aplikace stejné transformace na LABEL mapu
            transform_label_map(
                label_path,
                transforms,
                fixed_ref,
                label_output_path
            )
            
            print(f"Hotovo: {zadc_output_path} a {label_output_path}")
            
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
    
    args = parser.parse_args()
    
    # Kontrola, zda existují všechny potřebné cesty
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