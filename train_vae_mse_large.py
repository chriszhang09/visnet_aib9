import argparse
import logging
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import pathlib

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from aib9_lib import aib9_tools as aib9

def pairwise_distance_loss(true_coords, pred_coords, p=2):
    device = torch.device('cuda')
    if pred_coords.device != true_coords.device:
        pred_coords = pred_coords.to(device)
        true_coords = true_coords.to(device)
    true_distances = torch.pdist(true_coords, p=p)
    pred_distances = torch.pdist(pred_coords, p=p)
    loss = F.mse_loss(pred_distances, true_distances)
    return loss


def simple_mse_loss(pred_coords, target_coords):
    device = torch.device('cuda')
    """
    Simple MSE loss on coordinates with E(3) invariance via centering.
    Much faster and more stable than pairwise distance loss.
    All computations on GPU for maximum speed.
    
    Args:
        pred_coords: Predicted coordinates [total_atoms, 3] (PyG format)
        target_coords: Target coordinates [total_atoms, 3] (PyG format)
    
    Returns:
        MSE loss on centered coordinates
    """
    # Ensure both tensors are on the same device and float32
    pred_coords = pred_coords.float()
    target_coords = target_coords.float()
    
    # Ensure both are on the same device (GPU if available)
    if pred_coords.device != target_coords.device:
        pred_coords = pred_coords.to(device)
        target_coords = target_coords.to(device)
    
    # Center both coordinate sets to make them translation invariant
    # All operations stay on GPU
    pred_centered = pred_coords - pred_coords.mean(dim=0, keepdim=True)
    target_centered = target_coords - target_coords.mean(dim=0, keepdim=True)
    
    # Simple MSE loss on centered coordinates (GPU computation)
    mse_loss = F.mse_loss(pred_centered, target_centered)
    
    # Scale down to reasonable range (coordinates are typically 0.1-2.0 nm)
    return mse_loss


from mse_training.visnet_vae_encoder_mse import ViSNetEncoderMSE
from mse_training.vae_model_mse import MolecularVAEMSE, vae_loss_function_mse
from vae_utils import validate_and_sample, visualize_molecule_3d, compute_bond_lengths

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from pathlib import Path
import sys
import wandb
from torch.cuda.amp import autocast, GradScaler

def main():
    seed = 11
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Training parameters
    ATOM_COUNT = 58
    COORD_DIM = 3
    ORIGINAL_DIM = ATOM_COUNT * COORD_DIM  
    LATENT_DIM = 128 
    EPOCHS = 110
    VISNET_HIDDEN_CHANNELS = 256
    ENCODER_NUM_LAYERS = 9
    DECODER_HIDDEN_DIM = 256
    DECODER_NUM_LAYERS = 11
    BATCH_SIZE = 64
    LEARNING_RATE = 4e-5  
    NUM_WORKERS = 2  # Parallel data loading

    train_data_np = np.load(aib9.FULL_DATA)
    train_data_np = train_data_np.reshape(-1, 58, 3)
    train_data_np, val_data_np = train_test_split(train_data_np, test_size=0.1, random_state=seed)
  

        # Initialize Weights & Biases
    wandb.init(
        project="aib9-vae-pairwise",  # Different project name
        config={
            "atom_count": ATOM_COUNT,
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "encoder_hidden": VISNET_HIDDEN_CHANNELS,
            "encoder_layers": ENCODER_NUM_LAYERS,
            "decoder_hidden": DECODER_HIDDEN_DIM,
            "decoder_layers": DECODER_NUM_LAYERS,
            "loss_type": "Pairwise_Distance",  # Track loss type
        },
        save_code=True
    )
        # Create checkpoint directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA device: {torch.cuda.get_device_name()}')
        print(f'CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'Using MPS device: {torch.backends.mps.get_device()}')
        print(f'MPS memory: {torch.backends.mps.get_device().total_memory / 1e9:.1f} GB')
    else:
        device = torch.device('cpu')
        print('CUDA not available, using CPU')

    project_path = pathlib.Path(__file__).resolve().parent
    TOPO_FILE = (
        project_path / "aib9_lib/aib9_atom_info.npy"
    )  


    ATOMICNUMBER_MAPPING = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    }

    ATOMIC_NUMBERS = []
    topo = np.load(TOPO_FILE)
    #print(f"Loaded topology with shape: {topo.shape}", topo)
    for i in range(topo.shape[0]):
        atom_name = topo[i, 0][0]
        if atom_name in ATOMICNUMBER_MAPPING:
            ATOMIC_NUMBERS.append(ATOMICNUMBER_MAPPING[atom_name])
        else:
            raise ValueError(f"Unknown atom name: {atom_name}")

    # Create all tensors directly on the selected device to avoid device mismatch
    z = torch.tensor(ATOMIC_NUMBERS, dtype=torch.long, device='cpu')

    # Create a list of Data objects, one for each molecule
    # Use cutoff-based edge identification instead of predefined covalent edges
    train_data_list = []
    for i in range(train_data_np.shape[0]):
        pos = torch.from_numpy(train_data_np[i]).float().to('cpu')
        # No edge_index - let ViSNet use cutoff-based edge identification
        data = Data(z=z, pos=pos)
        train_data_list.append(data)
    train_loader = DataLoader(
        train_data_list,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # Only 10 unique atomic types in the dataset
    atom_feature_dim = 54 # Should be 10
    
    visnet_params = {
        'hidden_channels': VISNET_HIDDEN_CHANNELS,
        'num_layers': ENCODER_NUM_LAYERS,
        'num_rbf': 32,
        'cutoff': 5.0,  # Used for cutoff-based edge identification
        'max_z': max(ATOMIC_NUMBERS) + 1,
    }

    model = MolecularVAEMSE(
        latent_dim=LATENT_DIM, 
        num_atoms=ATOM_COUNT, 
        atom_feature_dim=atom_feature_dim,
        visnet_hidden_channels=VISNET_HIDDEN_CHANNELS,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,      
        decoder_num_layers=DECODER_NUM_LAYERS,         
        # No edge_index_template needed - PyG handles batching automatically
        visnet_kwargs=visnet_params,
        cutoff=5.0
    ).to(device)

    parser = argparse.ArgumentParser(description='Train VAE with MSE loss')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    start_epoch = 1

       # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=30, verbose=True)
       # Mixed precision training
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None

    if args.resume:
        print(f"Resuming training from: {args.resume}")
        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"Warning: could not load optimizer state: {e}")
            if 'epoch' in ckpt:
                start_epoch = int(ckpt['epoch']) + 1
            if 'scheduler_state_dict' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception as e:
                    print(f"Warning: could not load scheduler state: {e}")
            if 'scaler_state_dict' in ckpt:
                try:
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
                except Exception as e:
                    print(f"Warning: could not load scaler state: {e}")
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

 

 
   

    # Log model to wandb
    wandb.watch(model, log='all', log_freq=200)  # Reduced log frequency
    
    # Keep a fixed validation sample for visualization
    val_sample = train_data_list[0]  
    
    print(f"\n{'='*60}")
    print(f"Starting Training - {len(train_data_list)} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss type: MSE (coordinate-based)")
    print(f"{'='*60}\n")

    # Training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        # Linear warmup for first 10 epochs
        if epoch <= 10:
            current_lr = LEARNING_RATE * (epoch / 10.0)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        for batch_idx, data in enumerate(train_loader):
            molecules = data.to(device)
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Mixed precision forward pass
            with autocast(enabled=use_amp):
                recon_batch, mu, log_var = model(molecules)
                # KL divergence for non-centered isotropic Gaussian: 0.5 * (||μ||² + σ² - log(σ²) - 1)
                kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            # Use simple MSE loss with centering for E(3) invariance
            recon_loss = pairwise_distance_loss(recon_batch, molecules.pos, 2)
            kl_div = torch.clamp(kl_div, max = 2000)
                # Debug KL components every 100 batches
            if batch_idx % 500 == 0:
                mu_norm = torch.mean(mu.pow(2)).item()
                log_var_mean = torch.mean(log_var).item()
                exp_log_var_mean = torch.mean(torch.exp(log_var)).item()
                print(f"  Debug - kl: {kl_div:.4f}, recon_loss: {recon_loss:.4f}")
                print(f"  Debug - μ²: {mu_norm:.4f}, log_var: {log_var_mean:.4f}, exp(log_var): {exp_log_var_mean:.4f}")

            # Clamp reconstruction loss to prevent explosion
            recon_loss = torch.clamp(recon_loss, max = 1000)  # Lower clamp for MSE

            kl_weight = min(1, epoch / 10)
            recon_loss = recon_loss.to(device)

            loss = recon_loss + kl_weight*kl_div 
            
                
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
              
                optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_div.item()
        
        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        
        # Step scheduler
        scheduler.step(avg_loss)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_recon_loss': avg_recon_loss,
            'train_kl_loss': avg_kl_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'gradient_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        })
        
        print(f'Epoch {epoch:3d}: Loss={avg_loss:.4f} (Recon={avg_recon_loss:.4f}, KL={avg_kl_loss:.4f}) LR={optimizer.param_groups[0]["lr"]:.2e}')
        
        # Validation and sampling every 10 epochs or at start (reduced frequency for speed)
        if epoch == 1 or epoch % 2 == 0:
            print(f"  → Generating samples and visualizations...")
            metrics, figures = validate_and_sample(
                model, val_sample.clone(), device, z, None, epoch
            )
            
            # Log metrics
            wandb.log(metrics)
            
            # Log figures
            for fig_name, fig in figures.items():
                wandb.log({fig_name: wandb.Image(fig)})
                plt.close(fig)
        
        # Save checkpoint every 10 epochs with timestamp
        if epoch % 2 == 0:
            checkpoint_path = f'checkpoints/vae_model_pairwise_epoch{epoch}_large_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'config': {
                    'latent_dim': LATENT_DIM,
                    'num_atoms': ATOM_COUNT,
                    'atom_feature_dim': atom_feature_dim,
                    'visnet_hidden_channels': VISNET_HIDDEN_CHANNELS,
                    'decoder_hidden_dim': DECODER_HIDDEN_DIM,
                    'decoder_num_layers': DECODER_NUM_LAYERS,
                    'loss_type': 'MSE'
                }
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")

    # Final validation
    print(f"\n{'='*60}")
    print("Training complete! Generating final samples...")
    print(f"{'='*60}\n")
    
    metrics, figures = validate_and_sample(
        model, val_sample.clone(), device, z, None, EPOCHS
    )
    wandb.log(metrics)
    for fig_name, fig in figures.items():
        wandb.log({fig_name: wandb.Image(fig)})
        plt.close(fig)
    
    # Save final model with pairwise suffix
    final_model_path = 'checkpoints/vae_model_pairwise_final_large_model.pth'
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': avg_loss,
        'config': {
            'latent_dim': LATENT_DIM,
            'num_atoms': ATOM_COUNT,
            'atom_feature_dim': atom_feature_dim,
            'visnet_hidden_channels': VISNET_HIDDEN_CHANNELS,
            'decoder_hidden_dim': DECODER_HIDDEN_DIM,
            'decoder_num_layers': DECODER_NUM_LAYERS,
            'loss_type': 'MSE'
        }
    }, final_model_path)
    
    print(f"Final model saved to: {final_model_path}")
    wandb.finish()

if __name__ == "__main__":
    main()
