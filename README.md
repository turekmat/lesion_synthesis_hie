# Dvou-GAN systém pro syntézu HIE lézí

Tento projekt implementuje dvoustupňový GAN přístup pro syntézu realistických hypoxicko-ischemických encefalopatických (HIE) lézí v MRI obrazech novorozenců.

## Přehled

Systém se skládá ze dvou specializovaných GANů:

1. **LabelGAN** - Generuje binární segmentační mapy lézí (LABEL mapy) na základě normativního atlasu mozku a volitelného atlasu frekvence lézí.
2. **IntensityGAN** - Upravuje intenzity voxelů v normativním atlasu podle vygenerovaných LABEL map, což vede k realistickým ZADC mapám s lézemi.

Tento dvoustupňový přístup umožňuje lepší kontrolu nad generačním procesem a dosahuje vyšší kvality syntetizovaných dat, než by bylo možné s jediným modelem.

## Komponenty systému

### 1. LabelGAN (label_gan.py)

LabelGAN se specializuje na generování binárních segmentačních map lézí. Tento model:

- Využívá normativní atlas jako vstup
- Může používat volitelný atlas frekvence lézí pro klinicky relevantnější distribuci lézí
- Generuje binární mapy, které označují lokace lézí
- Obsahuje anatomicky informovanou ztrátovou funkci pro penalizaci lézí v nepravděpodobných oblastech

### 2. IntensityGAN (intensity_gan.py)

IntensityGAN se specializuje na úpravu intenzit voxelů podle segmentačních map lézí. Tento model:

- Využívá normativní atlas a LABEL mapu jako vstup
- Generuje ZADC mapy s realistickými intenzitami v místech lézí
- Upravuje také okolní tkáň pro větší variabilitu mezi vzorky
- Obsahuje specializované ztrátové funkce pro oblasti s lézemi a bez lézí

### 3. Integrační pipeline (synthetic_pipeline.py)

Skript `synthetic_pipeline.py` propojuje oba GAN modely a umožňuje spustit celý proces syntézy dat v jednom kroku.

## Požadavky

- Python 3.6+
- PyTorch 1.8+
- CUDA (doporučeno pro trénink)
- SimpleITK
- Nibabel
- NumPy
- ZADC mapy a LABEL mapy pro trénink
- Normativní atlas mozku novorozence

## Instalace

```bash
# Instalace závislostí
pip install torch torchvision nibabel SimpleITK numpy
```

## Použití

Systém lze používat několika způsoby:

### 1. Kompletní pipeline (trénink a generování)

```bash
python synthetic_pipeline.py --run_complete_pipeline \
    --normal_atlas_path /cesta/k/atlasu.nii.gz \
    --lesion_atlas_path /cesta/k/atlasu_lezi.nii.gz \
    --zadc_dir /cesta/k/zadc_mapam \
    --label_dir /cesta/k/label_mapam \
    --synthetic_data_dir ./output/syntetická_data \
    --num_samples 200
```

### 2. Samostatný trénink LabelGAN

```bash
python label_gan.py train \
    --normal_atlas_path /cesta/k/atlasu.nii.gz \
    --lesion_atlas_path /cesta/k/atlasu_lezi.nii.gz \
    --label_dir /cesta/k/label_mapam \
    --output_dir ./output/labelgan \
    --epochs 100 \
    --batch_size 2
```

### 3. Samostatné generování LABEL map

```bash
python label_gan.py generate \
    --normal_atlas_path /cesta/k/atlasu.nii.gz \
    --lesion_atlas_path /cesta/k/atlasu_lezi.nii.gz \
    --checkpoint_path ./output/labelgan/labelgan_checkpoint_epoch99.pt \
    --output_dir ./output/label_mapy \
    --num_samples 200
```

### 4. Samostatný trénink IntensityGAN

```bash
python intensity_gan.py train \
    --normal_atlas_path /cesta/k/atlasu.nii.gz \
    --zadc_dir /cesta/k/zadc_mapam \
    --label_dir /cesta/k/label_mapam \
    --output_dir ./output/intensitygan \
    --epochs 100 \
    --batch_size 2
```

### 5. Samostatné generování ZADC map

```bash
python intensity_gan.py generate \
    --normal_atlas_path /cesta/k/atlasu.nii.gz \
    --label_dir ./output/label_mapy/label \
    --checkpoint_path ./output/intensitygan/intensitygan_checkpoint_epoch99.pt \
    --output_dir ./output/zadc_mapy \
    --num_samples 200
```

## Doporučené hyperparametry

Pro dosažení nejlepších výsledků doporučujeme následující nastavení:

```bash
# LabelGAN
--label_generator_filters 64 \
--label_discriminator_filters 64 \
--lambda_dice 50.0 \
--use_self_attention \
--dropout_rate 0.3

# IntensityGAN
--intensity_generator_filters 64 \
--intensity_discriminator_filters 64 \
--lambda_l1 100.0 \
--lambda_lesion 50.0 \
--lambda_non_lesion 25.0 \
--lambda_intensity_var 10.0 \
--use_self_attention
```

## Výstupní adresářová struktura

Po spuštění kompletního pipeline bude vytvořena následující adresářová struktura:

```
output/
  ├── labelgan/                      # Výstup tréninku LabelGAN
  │   ├── labelgan_checkpoint_epoch*.pt    # Checkpointy modelu
  │   └── labelgan_sample_epoch*_*.nii.gz  # Vzorky z tréninku
  │
  ├── intensitygan/                  # Výstup tréninku IntensityGAN
  │   ├── intensitygan_checkpoint_epoch*.pt   # Checkpointy modelu
  │   └── intensitygan_sample_epoch*_*.nii.gz # Vzorky z tréninku
  │
  └── synthetic_data/                # Výsledná syntetická data
      ├── label/                     # Vygenerované LABEL mapy
      │   └── sample_*_label.nii.gz  # Jednotlivé LABEL mapy
      │
      └── zadc/                      # Vygenerované ZADC mapy
          └── sample_*_zadc.nii.gz   # Jednotlivé ZADC mapy
```

## Poznámky k použití

- Pro trénink jsou potřeba registrované ZADC mapy a LABEL mapy se stejnými rozměry jako normativní atlas
- Použití atlasu frekvence lézí je volitelné, ale může výrazně zlepšit klinickou relevanci generovaných dat
- Pro trénink na CPU snižte velikost vstupu nebo velikost batch

## Rozšíření a budoucí vývoj

- Přidání podmíněné generace podle specifických parametrů (např. věk, závažnost)
- Rozšíření na více modalit MRI (T1, T2, FLAIR)
- Implementace 3D patchů pro trénink na větších objemových datech
- Modelování progresivního růstu lézí

## Citování

Pokud použijete tento kód ve svém výzkumu, prosím citujte naši práci:

```
@article{hie_lesion_synthesis,
  title={Dvou-GAN přístup pro syntézu HIE lézí v MRI obrazech novorozenců},
  author={Váš Autor},
  journal={Váš Journal},
  year={2023}
}
```

## Licence

Tento projekt je licencován pod MIT licencí - viz soubor LICENSE pro více detailů.
