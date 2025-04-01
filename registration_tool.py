import os
import argparse
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pathlib import Path

def register_image_to_atlas(moving_image_path, fixed_image_path, output_path):
    """
    Registruje vstupní obraz na atlas pomocí SimpleITK
    
    Args:
        moving_image_path: Cesta k obrazu, který bude registrován
        fixed_image_path: Cesta k obrazu atlasu
        output_path: Cesta pro uložení registrovaného obrazu
    """
    print(f"Registering {moving_image_path} to {fixed_image_path}")
    
    # Načtení obrazů
    if moving_image_path.endswith('.nii.gz'):
        moving_image = sitk.ReadImage(str(moving_image_path))
    else:
        moving_image = sitk.ReadImage(str(moving_image_path))
        
    if fixed_image_path.endswith('.nii.gz'):
        fixed_image = sitk.ReadImage(str(fixed_image_path))
    else:
        fixed_image = sitk.ReadImage(str(fixed_image_path))
    
    # Inicializace parametrů registrace
    parameter_map = sitk.GetDefaultParameterMap("bspline")
    parameter_map["FinalGridSpacingInPhysicalUnits"] = ["10.0", "10.0", "10.0"]
    parameter_map["NumberOfResolutions"] = ["4"]
    parameter_map["MaximumNumberOfIterations"] = ["500"]
    
    # Vytvoření elastix objektu pro registraci
    elastix = sitk.ElastixImageFilter()
    elastix.SetParameterMap(parameter_map)
    elastix.SetFixedImage(fixed_image)
    elastix.SetMovingImage(moving_image)
    
    # Provedení registrace
    elastix.Execute()
    
    # Uložení registrovaného obrazu
    result_image = elastix.GetResultImage()
    sitk.WriteImage(result_image, str(output_path))
    
    # Získání transformačního parametrického souboru pro další použití
    transform_parameter_map = elastix.GetTransformParameterMap()
    
    return transform_parameter_map

def transform_label_map(label_map_path, transform_parameter_map, output_path):
    """
    Aplikuje transformaci na label mapu
    
    Args:
        label_map_path: Cesta k label mapě
        transform_parameter_map: Parametry transformace získané z registrace
        output_path: Cesta pro uložení transformované label mapy
    """
    print(f"Transforming label map {label_map_path}")
    
    # Načtení label mapy
    if label_map_path.endswith('.nii.gz'):
        label_map = sitk.ReadImage(str(label_map_path))
    else:
        label_map = sitk.ReadImage(str(label_map_path))
    
    # Vytvoření transformix objektu pro aplikaci transformace
    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transform_parameter_map)
    transformix.SetMovingImage(label_map)
    
    # Aplikace transformace
    transformix.Execute()
    
    # Uložení transformované label mapy
    result_label = transformix.GetResultImage()
    
    # Při transformaci label mapy je potřeba zachovat diskrétní hodnoty
    # (předejít interpolaci mezi hodnotami)
    result_label_array = sitk.GetArrayFromImage(result_label)
    result_label_array = (result_label_array > 0.5).astype(np.uint8)
    result_label_discrete = sitk.GetImageFromArray(result_label_array)
    result_label_discrete.CopyInformation(result_label)
    
    sitk.WriteImage(result_label_discrete, str(output_path))

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
    zadc_files = sorted([f for f in Path(args.zadc_dir).glob("*") if f.suffix in ['.mha', '.nii.gz']])
    label_files = sorted([f for f in Path(args.label_dir).glob("*") if f.suffix in ['.mha', '.nii.gz']])
    
    # Kontrola, zda máme stejný počet ZADC a LABEL souborů
    if len(zadc_files) != len(label_files):
        print(f"Varování: Rozdílný počet ZADC ({len(zadc_files)}) a LABEL ({len(label_files)}) souborů!")
    
    # Zpracování každého páru ZADC a LABEL
    for i, (zadc_path, label_path) in enumerate(zip(zadc_files, label_files)):
        print(f"Zpracování {i+1}/{len(zadc_files)}: {zadc_path.name}")
        
        # Generování názvů výstupních souborů
        zadc_output_path = zadc_output_dir / f"reg_{zadc_path.name}"
        label_output_path = label_output_dir / f"reg_{label_path.name}"
        
        # Změna přípony na .nii.gz, pokud je požadována
        if args.output_format == 'nii.gz':
            zadc_output_path = zadc_output_path.with_suffix('.nii.gz')
            label_output_path = label_output_path.with_suffix('.nii.gz')
        
        # Registrace ZADC na atlas
        transform_params = register_image_to_atlas(
            zadc_path,
            args.normal_atlas_path,
            zadc_output_path
        )
        
        # Aplikace stejné transformace na LABEL mapu
        transform_label_map(
            label_path,
            transform_params,
            label_output_path
        )
        
        print(f"Hotovo: {zadc_output_path} a {label_output_path}")

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
    
    process_dataset(args)

if __name__ == "__main__":
    main() 