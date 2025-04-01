# HIE Lesion Synthesis Framework

Tento framework slouží k syntéze realistických HIE (Hypoxic-Ischemic Encephalopathy) lézí v neonatálních mozcích pomocí GAN modelů.

## Popis projektu

Projekt se zaměřuje na segmentaci HIE lézí v rámci HIE-BONBID datasetu, který obsahuje 89 3D MRI skenů neonatálních mozků. Hlavním cílem je vytvořit generativní model, který dokáže syntetizovat realistické HIE léze, jež mohou být použity pro augmentaci trénovacích dat v úlohách segmentace.

### Datové sady

- **HIE-BONBID dataset**: 89 3D .mha imagů s neonatálními mozky obsahující ADC mapy, ZADC mapy a LABEL mapy
- **Normal atlases**: Vytvořené z 13 normativních ADC map skenovaných v 0-14 dnech po narození
- **Lesion atlases**: Kvantifikují frekvenci lézí v jednotlivých voxelech
- **HumanConnectome data**: 150 3D DWI imagů, ze kterých byly vytvořeny ADC mapy

## Struktura projektu

```
.
├── hie_lesion_synthesis.py     # Hlavní GAN model pro syntézu lézí
├── registration_tool.py        # Nástroj pro registraci dat na normativní atlas
├── evaluation.py               # Skript pro vyhodnocení a vizualizaci výsledků
├── requirements.txt            # Seznam závislostí
└── README.md                   # Tento soubor
```

## Instalace závislostí

```bash
pip install -r requirements.txt
```

## Pipeline zpracování

### 1. Registrace dat na normativní atlas

Nejprve je potřeba registrovat všechny ZADC mapy a odpovídající LABEL mapy na společný prostor normativního atlasu.

```bash
python registration_tool.py --normal_atlas_path data/archive/normal_atlases/atlas_week0-1_masked.nii.gz --zadc_dir data/BONBID2023_Train/2Z_ADC --label_dir data/BONBID2023_Train/3LABEL --output_dir ./registered_data
```

### 2. Trénink GAN modelu

Následně natrénujeme GAN model, který se naučí generovat realistické léze v prostoru normativního atlasu.

```bash
python hie_lesion_synthesis.py train --normal_atlas_path data/archive/normal_atlases/atlas_week0-1_masked.nii.gz --lesion_atlas_path data/archive/lesion_atlases/lesion_atlas.nii.gz --zadc_dir ./registered_data/registered_zadc --label_dir ./registered_data/registered_label --output_dir ./gan_output --epochs 200
```

### 3. Generování syntetických vzorků

Po natrénování můžeme vygenerovat syntetické vzorky:

```bash
python hie_lesion_synthesis.py generate --normal_atlas_path data/archive/normal_atlases/atlas_week0-1_masked.nii.gz --lesion_atlas_path data/archive/lesion_atlases/lesion_atlas.nii.gz --checkpoint_path ./gan_output/checkpoint_epoch199.pt --output_dir ./synthetic_samples --num_samples 20
```

### 4. Vyhodnocení výsledků

Nakonec lze vyhodnotit kvalitu syntetizovaných lézí:

```bash
python evaluation.py --real_image_dir ./registered_data/registered_zadc --synthetic_image_dir ./synthetic_samples --real_label_dir ./registered_data/registered_label --synthetic_label_dir ./synthetic_labels --output_dir ./evaluation_results
```

## Metodika

### GAN model

Implementovaný model využívá kondiční GAN (cGAN) architekturu:

- **Generator**: U-Net architektura, která na vstupu přijímá normativní atlas a náhodný šum
- **Discriminator**: PatchGAN diskriminátor pro rozlišení reálných a syntetických obrazů

### Řízená syntéza lézí

Model využívá atlas frekvence lézí k řízení procesu syntézy:

1. Náhodný šum je vážený podle frekvence výskytu lézí v daných lokacích
2. Generator se učí generovat léze v pravděpodobných anatomických lokacích
3. Diskriminátor se učí rozlišovat mezi reálnými a syntetickými lézemi

### Metriky evaluace

Pro vyhodnocení kvality syntézy používáme:

- **SSIM (Structural Similarity Index)**: Měří podobnost struktury mezi obrazy
- **Dice koeficient**: Měří prostorový překryv mezi segmentacemi lézí
- **95% Hausdorff Distance**: Měří maximální vzdálenost mezi hranicemi lézí

## Závislosti

Seznam hlavních závislostí:

- Python 3.7+
- PyTorch 1.8+
- SimpleITK
- nibabel
- numpy
- matplotlib
- scikit-image
- medpy
