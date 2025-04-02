# HIE Lesion Synthesis GAN

Generativní adversariální síť (GAN) pro syntézu realistických hypoxicko-ischemických encefalopatických (HIE) lézí na základě lézního atlasu a mozečkové masky.

## Popis

Tento projekt implementuje pokročilý GAN model, který generuje realistické HIE léze na základě:

1. Lézního atlasu, který definuje pravděpodobnost výskytu lézí v různých částech mozku
2. Mozečkové masky, která omezuje generování lézí pouze na reálné tkáně mozku

Model využívá několik pokročilých technik:

- Atlas-aware attention mechanism - směruje pozornost modelu na oblasti s vyšší pravděpodobností výskytu lézí
- Probabilistic atlas integration - využívá pravděpodobnostní informace z atlasu pro lepší generaci lézí
- Multiple specialized loss functions - kombinace loss funkcí pro dosažení realističtějších výsledků
- Distribution matching - zajišťuje, že distribuce lézí odpovídá atlasu

## Instalace

Doporučujeme použít virtuální prostředí pro instalaci požadovaných knihoven:

```bash
# Vytvoření virtuálního prostředí
python -m venv env

# Aktivace prostředí
# Pro Linux/Mac:
source env/bin/activate
# Pro Windows:
env\Scripts\activate

# Instalace závislostí
pip install -r requirements.txt
```

## Použití

Model lze trénovat pomocí následujícího příkazu:

```bash
python labelgan.py --label_dir <cesta_k_datům> --atlas_path <cesta_k_atlasu> --brain_mask_path <cesta_k_mozečkové_masce> --batch_size 4 --epochs 100
```

### Příklad:

```bash
python labelgan.py --label_dir /data/registered_label --atlas_path /data/atlases/lesion_atlas.nii --brain_mask_path /data/atlases/brainmask.nii --batch_size 4 --epochs 100
```

### Parametry:

- `--label_dir`: Adresář obsahující léze ve formátu NIfTI
- `--atlas_path`: Cesta k léznímu atlasu ve formátu NIfTI
- `--brain_mask_path`: Cesta k mozečkové masce ve formátu NIfTI
- `--batch_size`: Velikost batche pro trénink (výchozí 4)
- `--epochs`: Počet epochy tréninku (výchozí 100)
- `--learning_rate`: Learning rate optimizéru (výchozí 0.0002)
- `--z_dim`: Dimenze latentního prostoru (výchozí 128)
- `--save_dir`: Adresář pro ukládání výsledků (výchozí './results')
- `--debug`: Flag pro zapnutí debug módu pro podrobnější výpisy

## Výstupy

Po tréninku model ukládá:

1. Checkpointy modelu (každých 10 epoch)
2. Vizualizace generovaných lézí
3. Statistiky o generovaných lézích
4. TensorBoard logy pro sledování tréninku

Generované léze lze prohlížet v adresáři `<save_dir>/samples`.

## Struktura kódu

- `HIELesionDataset`: Třída pro načítání dat
- `Generator`: 3D generativní síť s atlasovým mechanismem
- `Discriminator`: 3D diskriminační síť s PatchGAN architekturou
- `HIELesionGANTrainer`: Třída pro trénink modelu a generování vzorků

## Požadavky

Viz soubor `requirements.txt` pro seznam požadovaných knihoven.

## Poznámky

Varování typu "Unable to register cuFFT factory" jsou způsobena konfliktem TensorFlow a PyTorch CUDA knihoven a lze je bezpečně ignorovat, jelikož neovlivňují chod modelu.
