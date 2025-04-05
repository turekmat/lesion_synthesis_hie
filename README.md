# SwinGAN for HIE Lesion Synthesis

This repository contains the implementation of a model for generating synthetic Hypoxic-Ischemic Encephalopathy (HIE) lesions using a SwinUNETR-based GAN architecture.

## Overview

The model takes a lesion frequency atlas as input and generates realistic 3D binary lesion masks. The lesions are constrained to appear only in regions where the atlas has non-zero values, with higher probabilities in regions with higher atlas values.

Key features:

- SwinUNETR backbone for the generator
- GAN architecture for realistic synthesis
- Focal Loss to handle the sparse nature of lesions
- Constraint mechanism to ensure generated lesions follow anatomical constraints

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Data Preparation

The model requires:

1. A lesion frequency atlas (.nii file) with dimensions 128x128x64
2. Training lesion masks (.nii files) also with dimensions 128x128x64

Ensure that the data is properly normalized to the same brain space.

## Command-Line Usage

The model can be used from the command line for both training and generation.

### Training

To train the model:

```bash
python swin_gan.py train --lesion_dir /path/to/lesion/data --lesion_atlas /path/to/lesion_atlas.nii --output_dir ./swin_gan_output
```

#### Required Arguments

- `--lesion_dir`: Directory containing lesion .nii files for training
- `--lesion_atlas`: Path to the lesion frequency atlas .nii file

#### Optional Arguments

- `--output_dir`: Output directory for saving model checkpoints (default: './swin_gan_output')
- `--val_lesion_dir`: Directory containing lesion .nii files for validation
- `--batch_size`: Batch size for training (default: 2)
- `--epochs`: Number of epochs to train (default: 100)
- `--lr_g`: Learning rate for generator (default: 0.0002)
- `--lr_d`: Learning rate for discriminator (default: 0.0001)
- `--feature_size`: Feature size for SwinUNETR (default: 24)
- `--dropout_rate`: Dropout rate for generator (default: 0.0)
- `--lambda_focal`: Weight for focal loss (default: 10.0)
- `--lambda_l1`: Weight for L1 loss (default: 5.0)
- `--focal_alpha`: Alpha parameter for focal loss (default: 0.75)
- `--focal_gamma`: Gamma parameter for focal loss (default: 2.0)
- `--min_non_zero`: Minimum percentage of non-zero voxels to include sample (default: 0.001)
- `--save_interval`: Interval for saving full model checkpoints (default: 5)
- `--generator_save_interval`: Interval for saving generator-only checkpoints (default: 4)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--device`: Device to use (cuda or cpu) (default: 'cuda' if available, otherwise 'cpu')
- `--disable_amp`: Disable automatic mixed precision training

#### Checkpoints

The training process saves two types of checkpoints:

1. **Full model checkpoints**: Saved every `save_interval` epochs in the specified output directory, containing the entire model state (generator, discriminator, optimizers, and losses).
2. **Generator-only checkpoints**: Saved every `generator_save_interval` epochs in a subdirectory called `generator_checkpoints`. These are lighter checkpoints containing only the generator part of the model, which is all you need for inference.

### Generation

To generate synthetic lesions using a trained model:

```bash
python swin_gan.py generate --model_checkpoint /path/to/checkpoint.pt --lesion_atlas /path/to/lesion_atlas.nii --output_file ./generated_lesion.nii
```

You can use either a full model checkpoint or a generator-only checkpoint with the `--model_checkpoint` parameter.

#### Required Arguments

- `--model_checkpoint`: Path to the trained model checkpoint (full model or generator-only)
- `--lesion_atlas`: Path to the lesion frequency atlas .nii file
- `--output_file`: Path to save the generated lesion .nii file

#### Optional Arguments

- `--threshold`: Threshold for binarizing the generated lesions (default: 0.5)
- `--feature_size`: Feature size for SwinUNETR (must match training) (default: 24)
- `--device`: Device to use (cuda or cpu) (default: 'cuda' if available, otherwise 'cpu')

## Examples

### Training Example

```bash
# Basic training with default parameters
python swin_gan.py train --lesion_dir ./data/hie_lesions --lesion_atlas ./data/lesion_atlas.nii

# Training with custom parameters and checkpoint intervals
python swin_gan.py train \
    --lesion_dir ./data/hie_lesions \
    --lesion_atlas ./data/lesion_atlas.nii \
    --output_dir ./my_model_output \
    --batch_size 4 \
    --epochs 200 \
    --feature_size 32 \
    --lambda_focal 15.0 \
    --save_interval 10 \
    --generator_save_interval 4
```

### Generation Example

```bash
# Generate using a full model checkpoint
python swin_gan.py generate \
    --model_checkpoint ./swin_gan_output/swin_gan_epoch_100.pt \
    --lesion_atlas ./data/lesion_atlas.nii \
    --output_file ./generated_lesion.nii

# Generate using a generator-only checkpoint
python swin_gan.py generate \
    --model_checkpoint ./swin_gan_output/generator_checkpoints/generator_epoch_96.pt \
    --lesion_atlas ./data/lesion_atlas.nii \
    --output_file ./generated_lesion.nii \
    --threshold 0.3
```

## Python API

You can also use the model programmatically. See below for examples:

### Training the Model

```python
from swin_gan import HieLesionDataset, SwinGAN, SwinGANTrainer
import torch
from torch.utils.data import DataLoader

# Set paths
lesion_dir = "/path/to/lesion/data"
lesion_atlas_path = "/path/to/lesion_atlas.nii"
output_dir = "./swin_gan_output"

# Create dataset
train_dataset = HieLesionDataset(
    lesion_dir=lesion_dir,
    lesion_atlas_path=lesion_atlas_path,
    filter_empty=True,
    min_non_zero_percentage=0.001
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=2,  # Adjust based on GPU memory
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Create model
model = SwinGAN(
    in_channels=1,
    out_channels=1,
    feature_size=24,
    dropout_rate=0.0,
    lambda_focal=10.0,
    lambda_l1=5.0,
    focal_alpha=0.75,
    focal_gamma=2.0
)

# Create trainer
trainer = SwinGANTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=None,  # Add validation loader if available
    lr_g=0.0002,
    lr_d=0.0001,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir=output_dir,
    use_amp=True
)

# Train model
trainer.train(epochs=100, save_interval=5)
```

### Generating Lesions

```python
import torch
import nibabel as nib
import numpy as np
from swin_gan import SwinGAN

# Load model
model = SwinGAN()
checkpoint = torch.load("path/to/checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.to("cuda")

# Load atlas
atlas_nii = nib.load("path/to/lesion_atlas.nii")
atlas = atlas_nii.get_fdata()
atlas = atlas / np.max(atlas)  # Normalize to [0, 1]

# Convert to tensor
atlas_tensor = torch.from_numpy(atlas).float().unsqueeze(0).unsqueeze(0).to("cuda")

# Generate lesions
model.eval()
with torch.no_grad():
    fake_lesion = model.generator(atlas_tensor)
    binary_lesion = (fake_lesion > 0.5).float()

    # Constrain to atlas regions
    constraint_mask = (atlas_tensor > 0).float()
    binary_lesion = binary_lesion * constraint_mask

# Convert to numpy and save
lesion_np = binary_lesion.cpu().squeeze().numpy()
lesion_nii = nib.Nifti1Image(lesion_np.astype(np.uint8), atlas_nii.affine)
nib.save(lesion_nii, "generated_lesion.nii")
```

## Model Architecture

- **Generator**: SwinUNETR-based architecture that takes a lesion atlas as input and outputs a lesion probability map
- **Discriminator**: 3D convolutional network that differentiates between real and generated lesions
- **Training**: Adversarial training with Focal Loss and spatial constraints

## Evaluation

The quality of generated lesions can be evaluated based on:

1. Realism (visual inspection and discriminator score)
2. Adherence to atlas guidance (lesions should occur in high-probability regions)
3. Sparsity patterns (should match the distribution of real lesions)
4. Number of distinct lesions per volume
