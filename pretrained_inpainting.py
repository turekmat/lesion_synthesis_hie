import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import nibabel as nib
from glob import glob
import torch.nn.functional as F
import re
import shutil
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

class BrainLesionDataset(Dataset):
    def __init__(self, pseudo_healthy_dir, adc_dir, label_dir, transform=None):
        """
        Dataset for brain lesion inpainting with variable-sized inputs.
        
        Args:
            pseudo_healthy_dir: Directory with pseudo-healthy ADC maps
            adc_dir: Directory with ADC target maps
            label_dir: Directory with lesion masks
            transform: Optional transforms to apply
        """
        self.pseudo_healthy_dir = pseudo_healthy_dir
        self.adc_dir = adc_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # Find all files in each directory
        self.pseudo_healthy_files = sorted(glob(os.path.join(pseudo_healthy_dir, "*.mha")))
        self.adc_files = sorted(glob(os.path.join(adc_dir, "*.mha")))
        self.label_files = sorted(glob(os.path.join(label_dir, "*.mha")))
        
        # Match files by extracting patient IDs
        self.matched_files = self._match_files()
        
    def _match_files(self):
        """Match pseudo-healthy, ADC, and label files by patient ID"""
        matched = []
        
        # Extract IDs from pseudo-healthy files
        ph_ids = {}
        for ph_file in self.pseudo_healthy_files:
            # Extract identifier from filename
            basename = os.path.basename(ph_file)
            # Assuming filenames contain some form of patient ID
            # Adjust this pattern based on your actual filename format
            match = re.search(r'([\w-]+)', basename)
            if match:
                patient_id = match.group(1)
                ph_ids[patient_id] = ph_file
        
        # Match with ADC and label files
        for adc_file in self.adc_files:
            adc_basename = os.path.basename(adc_file)
            match = re.search(r'([\w-]+)', adc_basename)
            
            if match:
                patient_id = match.group(1)
                if patient_id in ph_ids:
                    # Find corresponding label file
                    label_pattern = os.path.join(self.label_dir, f"*{patient_id}*.mha")
                    label_files = glob(label_pattern)
                    
                    if label_files:
                        matched.append({
                            'pseudo_healthy': ph_ids[patient_id],
                            'adc': adc_file,
                            'label': label_files[0],
                            'patient_id': patient_id
                        })
        
        print(f"Found {len(matched)} matched file sets")
        return matched
    
    def __len__(self):
        return len(self.matched_files)
    
    def __getitem__(self, idx):
        # Get matched file paths
        file_set = self.matched_files[idx]
        
        # Load all files
        pseudo_healthy = self._load_mha(file_set['pseudo_healthy'])
        adc_target = self._load_mha(file_set['adc'])
        lesion_mask = self._load_mha(file_set['label'])
        
        # Ensure they all have the same dimensions (they should since they're matched)
        assert pseudo_healthy.shape == adc_target.shape == lesion_mask.shape, \
            f"Shape mismatch: {pseudo_healthy.shape}, {adc_target.shape}, {lesion_mask.shape}"
        
        # Convert to torch tensors and add channel dimension
        pseudo_healthy = torch.from_numpy(pseudo_healthy).float().unsqueeze(0)
        adc_target = torch.from_numpy(adc_target).float().unsqueeze(0)
        lesion_mask = torch.from_numpy(lesion_mask).float().unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            pseudo_healthy = self.transform(pseudo_healthy)
            adc_target = self.transform(adc_target)
            lesion_mask = self.transform(lesion_mask)
        
        return {
            'input': pseudo_healthy,
            'target': adc_target,
            'mask': lesion_mask,
            'patient_id': file_set['patient_id'],
            'shape': pseudo_healthy.shape[1:],  # Original shape without channel dim
            'input_path': file_set['pseudo_healthy'],
            'target_path': file_set['adc'],
            'mask_path': file_set['label']
        }
    
    def _load_mha(self, file_path):
        """Load an MHA file and normalize it with clipping of negative values"""
        img = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(img)
        
        # First clip negative values to 0 to remove noise
        data = np.clip(data, 0, None)
        
        # Then normalize intensity to range [0,1]
        if data.max() > 0:  # Ensure we don't divide by zero
            data = data / data.max()
        
        return data


class AdaptivePoolingModel(nn.Module):
    """
    A model that can handle variable-sized inputs by using adaptive pooling
    to standardize the latent space dimensions.
    """
    def __init__(self, brain2vec_model, target_shape=(80, 96, 80)):
        super(AdaptivePoolingModel, self).__init__()
        self.target_shape = target_shape
        
        # Use encoder from brain2vec
        self.encoder = brain2vec_model.encoder
        
        # Freeze encoder weights (optional)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Use decoder from brain2vec with modifications
        self.decoder = brain2vec_model.decoder
        
        # Add adaptive pooling layers
        self.adaptive_pool = nn.AdaptiveAvgPool3d(target_shape)
        
    def forward(self, x, mask):
        # Resize input to target shape for the encoder
        x_resized = F.interpolate(x, size=self.target_shape, mode='trilinear', align_corners=False)
        mask_resized = F.interpolate(mask, size=self.target_shape, mode='nearest')
        
        # Get latent representation
        latent = self.encoder(x_resized)
        
        # Feed to decoder
        decoded = self.decoder(latent)  # This outputs target_shape
        
        # Resize back to original input shape
        original_shape = x.shape[2:]  # Remove batch and channel dims
        output_resized = F.interpolate(decoded, size=original_shape, mode='trilinear', align_corners=False)
        
        # Apply the mask to focus on lesion areas
        # Start with the input and replace only the masked areas
        final_output = x * (1 - mask) + output_resized * mask
        
        return final_output


class DynamicResizeBatch:
    """
    Custom collate function for DataLoader that resizes all samples 
    in a batch to the same dimensions (the largest in the batch).
    """
    def __call__(self, batch):
        # Find max dimensions in the batch
        max_shape = [0, 0, 0]
        for item in batch:
            shape = item['input'].shape[1:]  # [C, D, H, W]
            for i in range(3):
                max_shape[i] = max(max_shape[i], shape[i])
                
        # Resize all items to the max shape
        for item in batch:
            input_shape = item['input'].shape[1:]
            if input_shape != tuple(max_shape):
                item['input'] = F.pad(item['input'], 
                                     (0, max_shape[2] - input_shape[2],
                                      0, max_shape[1] - input_shape[1],
                                      0, max_shape[0] - input_shape[0]))
                item['target'] = F.pad(item['target'], 
                                      (0, max_shape[2] - input_shape[2],
                                       0, max_shape[1] - input_shape[1],
                                       0, max_shape[0] - input_shape[0]))
                item['mask'] = F.pad(item['mask'], 
                                    (0, max_shape[2] - input_shape[2],
                                     0, max_shape[1] - input_shape[1],
                                     0, max_shape[0] - input_shape[0]))
        
        # Combine into a batch
        inputs = torch.stack([item['input'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        
        patient_ids = [item['patient_id'] for item in batch]
        shapes = [item['shape'] for item in batch]
        
        return {
            'input': inputs,
            'target': targets,
            'mask': masks,
            'patient_id': patient_ids,
            'shape': shapes
        }


def load_brain2vec_model(model_path):
    """Load pretrained brain2vec model"""
    print(f"Loading brain2vec model from {model_path}")
    try:
        model = torch.load(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # If loading fails, we could create a mock model for development/testing
        return None


def train_inpainting_model(model, train_loader, val_loader, num_epochs, device, 
                          output_dir, lr=1e-4, log_dir='./logs'):
    """Train the model for lesion inpainting"""
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Define loss functions
    l1_loss = nn.L1Loss()
    
    # Define optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for i, batch in enumerate(train_loader):
            # Get data
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(inputs, masks)
            
            # Compute loss
            loss = l1_loss(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log metrics
            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Compute average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # Validation
        val_loss = validate_model(model, val_loader, device)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(output_dir, f'model_epoch{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f'Training completed. Final model saved to {final_path}')
    
    # Close tensorboard writer
    writer.close()
    
    return final_path


def validate_model(model, val_loader, device):
    """Validate the model on the validation set"""
    model.eval()
    total_loss = 0
    l1_loss = nn.L1Loss()
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(inputs, masks)
            loss = l1_loss(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def download_brain2vec_model(output_dir):
    """Download brain2vec model from Hugging Face using the transformers library"""
    try:
        from huggingface_hub import hf_hub_download
        
        print("Downloading brain2vec model from Hugging Face...")
        model_path = hf_hub_download(repo_id="radiata-ai/brain2vec", filename="model.pth")
        
        # Copy the model to the output directory
        os.makedirs(output_dir, exist_ok=True)
        destination = os.path.join(output_dir, "brain2vec_model.pth")
        shutil.copy(model_path, destination)
        
        print(f"Model downloaded and saved to {destination}")
        return destination
    
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please ensure you have installed the huggingface_hub package: pip install huggingface_hub")
        return None


def main():
    parser = argparse.ArgumentParser(description='Fine-tune brain2vec for lesion inpainting in Kaggle environment')
    
    # Kaggle directories
    parser.add_argument('--pseudo_healthy_dir', type=str, 
                       default='/kaggle/input/pseudohealthy-masked/masked_pseudohealthy',
                       help='Directory with pseudo-healthy ADC maps')
    parser.add_argument('--adc_dir', type=str, 
                       default='/kaggle/input/bonbid-2023-train/BONBID2023_Train/1ADC_ss',
                       help='Directory with ADC target maps')
    parser.add_argument('--label_dir', type=str, 
                       default='/kaggle/input/bonbid-2023-train/BONBID2023_Train/3LABEL',
                       help='Directory with lesion masks')
    
    # Output directories
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/output',
                       help='Directory to save model outputs')
    parser.add_argument('--log_dir', type=str, default='/kaggle/working/logs',
                       help='Directory for TensorBoard logs')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--download_model', action='store_true', help='Download brain2vec model from Hugging Face')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to pretrained brain2vec model (if already downloaded)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get brain2vec model
    if args.download_model:
        model_path = download_brain2vec_model(args.output_dir)
    else:
        model_path = args.model_path
    
    if not model_path:
        print("No model path provided and download failed. Exiting.")
        return
    
    # Create dataset
    print("Creating dataset...")
    dataset = BrainLesionDataset(
        args.pseudo_healthy_dir,
        args.adc_dir, 
        args.label_dir
    )
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=DynamicResizeBatch(),
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=DynamicResizeBatch(),
        num_workers=2
    )
    
    # Initialize model
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain2vec = load_brain2vec_model(model_path)
    
    if brain2vec:
        model = AdaptivePoolingModel(brain2vec).to(device)
        
        # Train model
        print(f"Starting training on {device}...")
        train_inpainting_model(
            model, 
            train_loader, 
            val_loader, 
            args.num_epochs, 
            device, 
            args.output_dir,
            args.lr, 
            args.log_dir
        )
    else:
        print("Failed to load brain2vec model. Exiting.")


if __name__ == "__main__":
    main()
