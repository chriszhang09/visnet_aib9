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

from aib9_lib import aib9_tools as aib9
from vae_model import MolecularVAE, vae_loss_function


def mse_loss_function(pred_coords, target_coords):
    """
    Simple MSE loss function for coordinate reconstruction.
    
    Args:
        pred_coords: Predicted coordinates [total_atoms, 3] (PyG format)
        target_coords: Target coordinates [total_atoms, 3] (PyG format)
    
    Returns:
        MSE loss between predicted and target coordinates
    """
    # Convert to float32
    pred_coords = pred_coords.float()
    target_coords = target_coords.float()
    
    # Handle both batch format [batch_size, num_atoms, 3] and PyG format [total_atoms, 3]
    if pred_coords.dim() == 3:
        # Batch format - flatten to PyG format
        batch_size, num_atoms, _ = pred_coords.shape
        pred_coords = pred_coords.view(-1, 3)  # [total_atoms, 3]
    
    # Simple MSE loss
    loss = F.mse_loss(pred_coords, target_coords)
    
    return loss


from visnet_vae_encoder import ViSNetEncoder

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from pathlib import Path
import sys
import wandb
from torch.cuda.amp import autocast, GradScaler
from vae_utils import validate_and_sample, visualize_molecule_3d, compute_bond_lengths

def main():
    # Training parameters
    ATOM_COUNT = 58
    COORD_DIM = 3
    ORIGINAL_DIM = ATOM_COUNT * COORD_DIM  
    LATENT_DIM = 36 
    EPOCHS = 50
    VISNET_HIDDEN_CHANNELS = 256
    ENCODER_NUM_LAYERS = 3
    DECODER_HIDDEN_DIM = 256
    DECODER_NUM_LAYERS = 2
    BATCH_SIZE = 512  # Increased from 128 (V100 can handle much more!)
    LEARNING_RATE = 5e-5  # Reduced to prevent gradient explosion
    NUM_WORKERS = 2  # Parallel data loading

    train_data_np = np.load(aib9.FULL_DATA)
    train_data_np = train_data_np.reshape(-1, 58, 3)

        # Initialize Weights & Biases
    wandb.init(
        project="aib9-vae-mse",  # Different project name
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
            "loss_type": "MSE",  # Track loss type
        }
    )

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA device: {torch.cuda.get_device_name()}')
        print(f'CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB')
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
    edges = aib9.identify_all_covalent_edges(topo)
    # edges is already in shape [2, num_edges], no need to transpose
    edge_index = torch.tensor(edges, dtype=torch.long, device='cpu').contiguous()

    train_data_list = []
    for i in range(train_data_np.shape[0]):
        pos = torch.from_numpy(train_data_np[i]).float().to('cpu')
        data = Data(z=z, pos=pos, edge_index=edge_index)
        train_data_list.append(data)
    train_loader = DataLoader(
        train_data_list,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    # Only 10 unique atomic types in the dataset
    atom_feature_dim = 54 # Should be 10
    
    visnet_params = {
        'hidden_channels': VISNET_HIDDEN_CHANNELS,
        'num_layers': ENCODER_NUM_LAYERS,
        'num_rbf': 32,
        'cutoff': 5.0,  # Kept for compatibility but not used with edge_index
        'max_z': max(ATOMIC_NUMBERS) + 1,
    }

    model = MolecularVAE(
        latent_dim=LATENT_DIM, 
        num_atoms=ATOM_COUNT, 
        atom_feature_dim=atom_feature_dim,
        visnet_hidden_channels=VISNET_HIDDEN_CHANNELS,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,      
        decoder_num_layers=DECODER_NUM_LAYERS,         
        # No edge_index_template needed - PyG handles batching automatically
        visnet_kwargs=visnet_params
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

    # Mixed precision training
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None

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
    for epoch in range(1, EPOCHS + 1):
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
                kl_div = 0.5 * torch.sum(mu.pow(2) + torch.exp(log_var) - log_var - 1)
                
                # Debug KL components every 100 batches
                if batch_idx % 100 == 0:
                    mu_norm = torch.mean(mu.pow(2)).item()
                    log_var_mean = torch.mean(log_var).item()
                    exp_log_var_mean = torch.mean(torch.exp(log_var)).item()
                    print(f"  Debug - μ²: {mu_norm:.4f}, log_var: {log_var_mean:.4f}, exp(log_var): {exp_log_var_mean:.4f}")

            # Use MSE loss instead of E(3) invariant loss
            recon_loss = mse_loss_function(recon_batch, molecules.pos)
            
            # Clamp reconstruction loss to prevent explosion
            recon_loss = torch.clamp(recon_loss, max=10.0)
            # Don't clamp KL divergence - let it grow naturally
            # kl_div = torch.clamp(kl_div, max=5.0)  # Removed clamping
            
            # Use higher KL weight to encourage latent space usage
            if kl_div < 50:
                kl_weight = min(1.0 + 0.1 * epoch, 2.0)  # scaling with time
            else:
                kl_weight = 0.1  # Increased from 0.01
            loss = recon_loss + kl_weight * kl_div
            # Check for numerical issues
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected in loss at epoch {epoch}, skipping batch...")
                continue
                
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
            
            # Check for gradient explosion
            if grad_norm > 10.0:
                print(f"Warning: Large gradient norm {grad_norm:.4f}")
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_div.item()
            
            # Early stopping for exploding loss
            if loss.item() > 100:
                print(f"Loss explosion detected: {loss.item():.4f}, stopping training...")
                return
        
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
        if epoch == 1 or epoch % 10 == 0:
            print(f"  → Generating samples and visualizations...")
            metrics, figures = validate_and_sample(
                model, val_sample, device, z, val_sample.edge_index, epoch
            )
            
            # Log metrics
            wandb.log(metrics)
            
            # Log figures
            for fig_name, fig in figures.items():
                wandb.log({fig_name: wandb.Image(fig)})
                plt.close(fig)
        
        # Save checkpoint every 10 epochs with timestamp
        if epoch % 10 == 0:
            checkpoint_path = f'vae_model_mse_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")

    # Final validation
    print(f"\n{'='*60}")
    print("Training complete! Generating final samples...")
    print(f"{'='*60}\n")
    
    metrics, figures = validate_and_sample(
        model, val_sample, device, z, val_sample.edge_index, EPOCHS
    )
    wandb.log(metrics)
    for fig_name, fig in figures.items():
        wandb.log({fig_name: wandb.Image(fig)})
        plt.close(fig)
    
    # Save final model with MSE suffix
    final_model_path = 'vae_model_mse_final.pth'
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
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
    # Add argument parser for resume functionality
    parser = argparse.ArgumentParser(description='Train VAE with MSE loss')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        # Load checkpoint logic would go here
        # For now, just run normal training
    
    main()
