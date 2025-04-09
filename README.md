# HIE Lesion Inpainting GAN

This repository contains a conditional GAN model for inpainting synthetic Hypoxic-Ischemic Encephalopathy (HIE) lesions into brain MRI scans. The model is designed to realistically insert synthetic lesions into brain ZADC maps, preserving the surrounding tissue structure while modifying the ADC values to reflect the pathology.

## Overview

The model uses a SwinUNETR (transformer-based U-Net) as a generator and a 3D PatchGAN discriminator to create realistic lesion inpainting in brain MRI scans. The goal is to generate synthetic but biologically plausible data for data augmentation in segmentation tasks.

## Data Structure

The model works with the following data structure:

- `data/BONBID2023_Train/2Z_ADC/`: ZADC maps (.mha files)
- `data/BONBID2023_Train/3LABEL/`: Binary masks of existing lesions (.mha files)
- `data/registered_lesions/`: Synthetic lesions registered to specific brain scans

## Requirements

```
torch>=1.10.0
monai>=0.9.0
SimpleITK>=2.1.0
numpy>=1.20.0
matplotlib>=3.4.0
tqdm>=4.60.0
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Model Architecture

### Generator

- 3D SwinUNETR (transformer-based U-Net)
- Input: 3D healthy brain ZADC map + synthetic lesion mask (2 channels)
- Output: 3D ZADC map with realistically inpainted lesion

### Discriminator

- 3D PatchGAN discriminator
- Input: 3D ZADC map + lesion mask (2 channels)
- Output: Binary classification (real/fake) focused on the lesion area

## Training

To train the model:

```bash
python inpainting_gan.py
```

By default, this will:

1. Load the ZADC maps and lesion masks
2. Create training and validation datasets
3. Train the model for 200 epochs
4. Save checkpoints every 10 epochs
5. Generate visualizations of the results
6. Save training curves

### Training Process

The training process follows these steps:

1. For each brain scan, we extract patches centered on lesions
2. The generator learns to create realistic lesions in the specified mask areas
3. The discriminator learns to distinguish between real and synthetic lesions
4. Various loss functions guide the training:
   - Adversarial loss: Makes the generated lesions look realistic
   - L1 loss for lesion area: Ensures the lesion has appropriate ADC values
   - Context loss for surrounding tissue: Preserves the structural integrity

## Inference

Once trained, the model can be used to inpaint new synthetic lesions:

```python
from inpainting_gan import LesionInpaintingGAN, apply_inpainting

# Load model from checkpoint
model = LesionInpaintingGAN()
model.load_models('output/lesion_inpainting/checkpoint_epoch_200.pt')

# Apply inpainting
apply_inpainting(
    model,
    zadc_file='path/to/zadc.mha',
    synthetic_lesion_file='path/to/synthetic_lesion.mha',
    output_file='path/to/output.mha'
)
```

## Results

The model produces:

1. Inpainted brain scans with realistic lesions
2. Visualizations showing the input, mask, and generated output
3. Training curves to monitor progress

## Use Cases

The primary use case is data augmentation for training segmentation models (such as SwinUNETR) for HIE lesion detection, by generating additional realistic examples of brains with lesions in varying locations.

## Notes

- Inpainted lesions preserve the surrounding brain structure while modifying only the masked area
- The model learns to adapt the ADC values appropriately for different brain regions
- The transformer-based architecture captures long-range dependencies in the brain structure

## License

[Include your license information here]

# HIE BONBID Lesion Synthesis

Tento projekt implementuje pokročilé metody pro syntézu a inpainting lézí HIE (Hypoxic-Ischemic Encephalopathy) v mozku pomocí MRI snímků z datasetu BONBID.

## Přehled implementovaných metod

### 1. Inpainting lézí pomocí CGAN s SwinUNETR

Nejnovější implementací je metoda pro inpainting lézí do pseudo-zdravých mozků pomocí Conditional GAN (CGAN) s architekturou SwinUNETR jako generátorem. Tato metoda je implementována v souboru `inpainting_gan.py`.

Hlavní výhody této metody:

- Využívá state-of-the-art architekturu SwinUNETR, která kombinuje výhody Swin Transformerů a U-Net architektur
- Využívá podmíněné generování (CGAN) pro řízené vkládání lézí podle binární masky
- Obsahuje specializované loss funkce pro zajištění kvality a realismu inpaintovaných lézí
- Je optimalizovaná pro práci s 3D volumetrickými daty

#### Datový tok:

1. **Vstupní data**:

   - Pseudo-zdravý mozek (ADC mapa bez lézí)
   - Binární maska léze (označuje, kde mají být inpaintovány léze)

2. **Výstupní data**:
   - ADC mapa s inpaintovanými lézemi

### 2. Generování lézí pomocí SwinGAN (implementovaný v `swin_gan.py`)

### 3. Příprava pseudo-zdravých mozků (implementováno v `pseudohealthy_brains.py`)

## Instalace

Pro instalaci požadovaných balíčků použijte:

```bash
pip install -r requirements.txt
```

## Použití modelu pro inpainting lézí (CGAN-SwinUNETR)

### Trénování modelu na celých objemech (full volume)

Pro trénování modelu na celých volumetrických datech (bez patchů) spusťte:

```bash
python inpainting_gan.py --mode train \
    --pseudo_healthy_dir /cesta/k/pseudo_zdravym_mozkum \
    --adc_dir /cesta/k/adc_mapam_s_lezemi \
    --lesion_mask_dir /cesta/k/maskam_lezi \
    --output_dir ./output_inpainting \
    --num_epochs 100 \
    --batch_size 1 \
    --use_amp
```

Model automaticky detekuje velikost vstupních dat a vyplní je na nejbližší vyšší mocninu 2 pro každý rozměr, což je potřebné pro SwinUNETR architekturu.

Můžete také specifikovat cílovou velikost manuálně, pokud potřebujete konkrétní rozměry:

```bash
python inpainting_gan.py --mode train \
    --pseudo_healthy_dir /cesta/k/pseudo_zdravym_mozkum \
    --adc_dir /cesta/k/adc_mapam_s_lezemi \
    --lesion_mask_dir /cesta/k/maskam_lezi \
    --output_dir ./output_inpainting \
    --num_epochs 100 \
    --batch_size 1 \
    --target_size 32,128,128 \
    --use_amp
```

Další užitečné parametry specifické pro full volume trénink:

- `--crop_foreground`: Ořezat objemy na bounding box mozku před tréninkem (šetří paměť)
- `--target_size`: Manuálně specifikujte cílovou velikost objemů, např. "32,128,128"
- `--batch_size`: Pro full volume trénink doporučujeme 1 nebo 2 (v závislosti na velikosti dat a paměti GPU)

### Standardní parametry trénování:

- `--lr_g`: Learning rate pro generátor (výchozí: 0.0002)
- `--lr_d`: Learning rate pro diskriminátor (výchozí: 0.0001)
- `--lambda_l1`: Váha pro L1 loss (výchozí: 10.0)
- `--lambda_identity`: Váha pro identity loss mimo masku (výchozí: 5.0)
- `--lambda_ssim`: Váha pro SSIM loss (výchozí: 1.0)
- `--lambda_edge`: Váha pro edge loss (výchozí: 1.0)
- `--checkpoint`: Cesta k checkpointu pro pokračování v tréninku

### Inference s využitím celých objemů

Pro aplikaci natrénovaného modelu na nová data:

```bash
python inpainting_gan.py --mode inference \
    --checkpoint /cesta/k/checkpointu/best_model.pth \
    --inference_input /cesta/k/pseudo_zdravemu_mozku.mha \
    --inference_mask /cesta/k/masce_leze.mha \
    --inference_output /cesta/kam/ulozit/vystup.mha
```

Při inferenci je vstupní objem automaticky doplněn paddingem na konzistentní velikost (mocninu 2), zpracován modelem a poté oříznut zpět na původní velikost.

## Architektura modelu CGAN-SwinUNETR

### Generátor

Generátor využívá architekturu SwinUNETR, která kombinuje Swin Transformer bloky s U-Net strukturou:

- **Vstupy**: Pseudo-zdravý mozek a binární maska léze (spojené jako 2-kanálový vstup)
- **Architektura**: SwinUNETR s parametrizovatelným počtem feature maps
- **Výstup**: ADC mapa s inpaintovanou lézí

Výhodou SwinUNETR je schopnost zachytit jak globální kontext (pomocí transformerů), tak lokální detaily (pomocí U-Net struktury), což je ideální pro inpainting lézí, které musí být konzistentní s okolní tkání.

### Diskriminátor

Diskriminátor je implementován jako PatchGAN, který klasifikuje každý patch obrázku jako reálný nebo falešný:

- **Vstupy**: Pseudo-zdravý mozek, maska léze a ADC mapa (reálná nebo generovaná)
- **Architektura**: Série 3D konvolučních vrstev s LeakyReLU aktivacemi a BatchNorm
- **Výstup**: Mapa skóre, která indikuje, zda každá oblast obsahuje reálnou nebo generovanou lézi

### Loss funkce

Model používá kombinaci více loss funkcí pro optimální výsledky:

1. **Adversarial Loss**: Standardní GAN loss pro realistické generování
2. **L1 Reconstruction Loss**: Zajišťuje podobnost generovaných lézí s reálnými (pouze v oblasti masky)
3. **Identity Loss**: Zajišťuje, že oblasti mimo masku zůstanou nezměněné
4. **Localized SSIM Loss**: Podporuje strukturální podobnost v oblasti léze
5. **Edge/Gradient Loss**: Zajišťuje plynulý přechod na hranici léze bez ostrých artefaktů

## Předzpracování dat

Implementace zahrnuje následující kroky předzpracování:

1. Oříznutí dat pomocí bounding boxu zaměřeného na mozek
2. Normalizace hodnot do rozsahu [0, 1]
3. Náhodné vzorkování patchů pro efektivní trénink
4. Konzistentní aplikace transformací pro všechny trojice (pseudo-zdravý, ADC s lézí, maska)

## Citace a odkazy

- SwinUNETR: [Hatamizadeh et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images"](https://arxiv.org/abs/2201.01266)
- MONAI: [Medical Open Network for AI](https://monai.io/)

## Kontakt

Pro další informace nebo dotazy ohledně projektu kontaktujte [email@example.com].
