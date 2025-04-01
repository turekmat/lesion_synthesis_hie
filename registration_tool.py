import os
import argparse
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pathlib import Path

def register_image_to_atlas(moving_image_path, fixed_image_path, output_path):
    """
    Registruje vstupní obraz na atlas pomocí vícestupňové registrace:
    1) Afinní registrace (Euler3DTransform)
    2) Následná B-spline deformovatelná registrace
    
    Args:
        moving_image_path: Cesta k obrazu, který bude registrován
        fixed_image_path: Cesta k obrazu atlasu (referenční obraz)
        output_path: Cesta pro uložení registrovaného obrazu
        
    Returns:
        final_transform: Výsledná (kompozitní) transformace
        fixed_image: Referenční (atlasový) obraz
    """
    print(f"Registering {moving_image_path} to {fixed_image_path}")

    # Načtení obrazů
    moving_image = sitk.ReadImage(str(moving_image_path))
    fixed_image = sitk.ReadImage(str(fixed_image_path))

    # ------------------------ #
    # KROK 1: Afinní registrace
    # ------------------------ #

    # Počáteční transformace: Euler3DTransform (rotačně-translační, lze rozšířit na plně afinní)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Nastavení metody registrace (afinní krok)
    registration_method = sitk.ImageRegistrationMethod()
    
    # Metoda vyhodnocení podobnosti
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # Interpolace
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Multi-resolution (pyramid)
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreInPhysicalUnitsOn()
    
    # Nastavení optimalizace
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Počáteční transformace
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # Spuštění registrace
    final_affine_transform = registration_method.Execute(fixed_image, moving_image)

    # ------------------------------------- #
    # KROK 2: B-spline (deformovatelná) fáze
    # ------------------------------------- #

    # Inicializace B-spline transformace
    transform_domain_mesh_size = [8, 8, 8]  # lze upravit podle velikosti dat
    bspline_transform = sitk.BSplineTransformInitializer(fixed_image, transform_domain_mesh_size)
    
    # Vytvoření kompozitní transformace (afinní + B-spline)
    composite_transform = sitk.Transform(final_affine_transform)
    composite_transform.AddTransform(bspline_transform)

    # Registrace B-spline
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Multi-resolution i pro B-spline
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreInPhysicalUnitsOn()
    
    # Pro B-spline lze použít např. LBFGSB nebo i jinou optimalizaci
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e+7
    )
    
    registration_method.SetInitialTransform(composite_transform, inPlace=False)
    
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # ------------------------ #
    # Aplikace výsledné transformace
    # ------------------------ #
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    
    result_image = resampler.Execute(moving_image)
    
    # Uložení registrovaného obrazu
    sitk.WriteImage(result_image, str(output_path))
    
    return final_transform, fixed_image

def transform_label_map(label_map_path, transform, reference_image, output_path):
    """
    Aplikuje transformaci na label mapu
    
    Args:
        label_map_path: Cesta k label mapě
        transform: Transformace získaná z registrace
        reference_image: Referenční obraz pro geometrické informace
        output_path: Cesta pro uložení transformované label mapy
    """
    print(f"Transforming label map {label_map_path}")
    
    label_map = sitk.ReadImage(str(label_map_path))
    
    # Aplikace transformace na label mapu
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    # Pro label mapy je vhodné použít nearest neighbor (zachování diskrétních hodnot)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    
    result_label = resampler.Execute(label_map)
    
    # Uložit přímo tak, aby nedošlo k interpolaci hodnot labelů
    # (Pokud byste dělali threshold, lze zde ponechat, ale 
    #  v nearest neighbor režimu by nemělo docházet k nežádoucímu prolínání.)
    sitk.WriteImage(result_label, str(output_path))

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
    
    # Načteme normativní atlas (referenční obraz)
    reference_image = sitk.ReadImage(str(args.normal_atlas_path))
    
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
        transform, registered_reference = register_image_to_atlas(
            zadc_path,
            args.normal_atlas_path,
            zadc_output_path
        )
        
        # Aplikace stejné transformace na LABEL mapu
        transform_label_map(
            label_path,
            transform,
            registered_reference,
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
