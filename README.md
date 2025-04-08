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
