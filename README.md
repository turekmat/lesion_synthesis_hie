# HIE Lesion Inpainting Model

Model pro inpainting (vkládání) HIE (hypoxicko-ischemické encefalopatie) lézí do mozků na základě ZADC map a syntetických lézí.

## Popis projektu

Tento projekt implementuje model založený na Conditional GAN (CycleGAN) pro realistické vkládání syntetických HIE lézí do mozků na základě ZADC map. Model se učí modifikovat ZADC hodnoty v oblasti léze tak, aby výsledná léze byla vizuálně a funkčně nerozlišitelná od skutečných lézí.

### Účel projektu

- Vytvořit syntetická, biologicky věrohodná data pro augmentaci trénovacího datasetu pro segmentační sítě (SwinUNETR)
- Umožnit generování různých variant lézí v mozku, což rozšíří variabilitu trénovacích dat
- Pomoci zlepšit přesnost segmentačních modelů díky většímu množství kvalitních trénovacích dat

## Architektura modelu

### Generator

- **Architektura**: 3D SwinUNETR (transformer-based U-Net)
- **Vstup**: 3D ZADC mapy + binární maska oblasti, kam chceme syntetickou lézi vložit (celkem 2 kanály)
- **Výstup**: 3D ZADC mapy s realisticky vloženou lézí v oblasti masky
- **Klíčové vlastnosti**:
  - Skip-connections (U-Net struktura)
  - Transformer bloky pro lepší kontextovou reprezentaci

### Discriminator

- **Architektura**: 3D PatchGAN discriminator
- **Vstup**: 3D ZADC s lézí + maska oblasti léze (celkem 2 kanály)
- **Výstup**: Binární klasifikace (realistický/syntetický)
- **Klíčové vlastnosti**:
  - Posuzuje realističnost pouze v oblasti masky (lokální realismus)
  - Spektrální normalizace pro stabilitu tréninku

## Data

Projekt pracuje s následujícími datovými složkami:

- `data/BONBID2023_Train/2Z_ADC`: ZADC mapy mozků (MHA soubory)
- `data/BONBID2023_Train/3LABEL`: Binární masky existujících lézí (ground truth)
- `data/registered_lesions`: Syntetické léze, které jsou registrovány na jednotlivé mozky

## Trénink modelu

Model je trénován s patch-based přístupem:

- Velikost patchů: 64x64x32
- Intenzitní augmentace a rotace pro zvýšení robustnosti
- Fokus na oblasti s lézemi pro efektivní učení
- Multi-komponentní loss funkce:
  - Adversarial loss (GAN)
  - L1 loss pro celý obraz
  - Specializovaná L1 loss pro oblast léze (s vyšší váhou)

## Jak použít model

### Instalace závislostí

```bash
pip install -r requirements.txt
```

### Trénink modelu

```bash
python inpainting_gan.py --mode train
```

### Validace a testování modelu

```bash
python inpainting_gan.py --mode test --checkpoint path/to/checkpoint
```

### Pokračování v tréninku z checkpointu

```bash
python inpainting_gan.py --mode train --checkpoint path/to/checkpoint
```

## Vizualizace výsledků

Během tréninku model ukládá vizualizace do složky `visualizations`, které ukazují:

- Původní ZADC mapu
- Cílovou ZADC mapu s lézí
- Vygenerovanou ZADC mapu s vloženou lézí
- Použitou masku léze

## Požadavky

Viz soubor `requirements.txt` pro kompletní seznam závislostí.

## Poznámky

- Model upravuje pouze hodnoty v oblasti masky léze a zachovává původní hodnoty mimo masku
- HIE léze jsou modelovány jako oblasti s nižšími ADC hodnotami (restricted diffusion)
- Doporučuje se používat GPU pro rychlejší trénink modelu
