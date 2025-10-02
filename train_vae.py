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


def e3_invariant_loss_bonds(pred_coords, target_coords, edge_index):
    """
    E(3) invariant loss using only covalent bond distances.
    Much more stable and meaningful than full distance matrices.
    
    Args:
        pred_coords: Predicted coordinates [batch_size, num_atoms, 3] or [total_atoms, 3] (PyG format)
        target_coords: Target coordinates [total_atoms, 3] (PyG format)
        edge_index: Edge connectivity [2, num_bonds] for bonded atoms (already batched by PyG)
    """
    # Convert to float32
    pred_coords = pred_coords.float()
    target_coords = target_coords.float()
    
    # Handle both batch format [batch_size, num_atoms, 3] and PyG format [total_atoms, 3]
    if pred_coords.dim() == 3:
        # Batch format - flatten to PyG format
        batch_size, num_atoms, _ = pred_coords.shape
        pred_coords = pred_coords.view(-1, 3)  # [total_atoms, 3]
    
    # Get bond indices
    row, col = edge_index[0], edge_index[1]
    
    # Compute bond distances for predicted coordinates
    pred_bond_distances = torch.norm(
        pred_coords[row] - pred_coords[col], 
        dim=-1
    )  # [num_bonds]
    
    # Compute bond distances for target coordinates
    target_bond_distances = torch.norm(
        target_coords[row] - target_coords[col], 
        dim=-1
    )  # [num_bonds]
    
    # MSE on bond distances only
    loss = F.mse_loss(pred_bond_distances, target_bond_distances)
    
    return loss
from visnet_vae_encoder import ViSNetEncoder

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from pathlib import Path
import sys
import wandb
from aib9_lib import aib9_tools as aib9
from torch.cuda.amp import autocast, GradScaler
from vae_utils import validate_and_sample, visualize_molecule_3d, compute_bond_lengths




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')


    #Training parameters
    ATOM_COUNT = 58
    COORD_DIM = 3
    ORIGINAL_DIM = ATOM_COUNT * COORD_DIM  
    LATENT_DIM = 64 
    EPOCHS = 50
    VISNET_HIDDEN_CHANNELS = 64
    ENCODER_NUM_LAYERS = 6
    DECODER_HIDDEN_DIM = 64
    DECODER_NUM_LAYERS = 6
    BATCH_SIZE = 512  # Increased from 128 (V100 can handle much more!)
    LEARNING_RATE = 5e-5  # Reduced to prevent gradient explosion
    NUM_WORKERS = 2  # Parallel data loading

    train_data_np = np.load(aib9.FULL_DATA)
    train_data_np = train_data_np.reshape(-1, 58, 3)

        # Initialize Weights & Biases
    wandb.init(
        project="aib9-vae",
        config={
            "atom_count": ATOM_COUNT,
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "encoder_hidden": VISNET_HIDDEN_CHANNELS,
            "encoder_layers": DECODER_NUM_LAYERS,
            "decoder_hidden": DECODER_HIDDEN_DIM,
            "decoder_layers": DECODER_NUM_LAYERS,
        }
    )

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
    z = torch.tensor(ATOMIC_NUMBERS, dtype=torch.long, device=device)

    # Create a list of Data objects, one for each molecule
    edges = aib9.identify_all_covalent_edges(topo)
    # edges is already in shape [2, num_edges], no need to transpose
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).contiguous()
    
    train_data_list = []
    for i in range(train_data_np.shape[0]):
        pos = torch.from_numpy(train_data_np[i]).float().to(device)
        data = Data(z=z, pos=pos, edge_index=edge_index).to(device)
        train_data_list.append(data)
    train_loader = DataLoader(
        train_data_list,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    # Only 10 unique atomic types in the dataset
    atom_feature_dim = 53 # Should be 10
    
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
        visnet_kwargs=visnet_params
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  

    # Optionally resume from checkpoint
    start_epoch = 1
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
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    scaler = GradScaler()
    use_amp = torch.cuda.is_available()  # Use AMP if CUDA available
    
    # Scheduler (used after warmup)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Linear warmup config
    WARMUP_EPOCHS = 10
    WARMUP_START_FACTOR = 0.1  
    BASE_LR = LEARNING_RATE
    
    # Watch model with wandb
    wandb.watch(model, log='all', log_freq=200)  # Reduced log frequency
    
    # Keep a fixed validation sample for visualization
    val_sample = train_data_list[0]  
    
    print(f"\n{'='*60}")
    print(f"Starting Training - {len(train_data_list)} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, EPOCHS + 1):
        # Apply linear warmup for the first WARMUP_EPOCHS
        if epoch <= WARMUP_EPOCHS:
            warmup_ratio = epoch / float(WARMUP_EPOCHS)
            current_lr = BASE_LR * (WARMUP_START_FACTOR + (1.0 - WARMUP_START_FACTOR) * warmup_ratio)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
        else:
            for pg in optimizer.param_groups:
                if 'initial_lr' in pg:
                    pg['lr'] = pg.get('lr', BASE_LR)
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            molecules = data.to(device)
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Mixed precision forward pass
            with autocast(enabled=use_amp):
                recon_batch, mu, log_var = model(molecules)
                # KL divergence for non-centered isotropic Gaussian: 0.5 * (||μ||² + σ² - log(σ²) - 1)
                kl_div = 0.5 * torch.sum(mu.pow(2) + torch.exp(log_var) - log_var - 1)
            
            # Use the batched edge_index from PyTorch Geometric (already properly offset for batching)
            recon_loss = e3_invariant_loss_bonds(recon_batch, molecules.pos, molecules.edge_index)
            
            # Clamp reconstruction loss to prevent explosion
            recon_loss = torch.clamp(recon_loss, max=10.0)
            kl_div = torch.clamp(kl_div, max=5.0)
            if recon_loss + kl_div < 10:
                kl_weight = 0.25
            else:
                kl_weight = 0.01
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
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_div.item()
        
        # Compute average losses
        avg_loss = train_loss / len(train_loader.dataset)
        avg_recon_loss = train_recon_loss / len(train_loader.dataset)
        avg_kl_loss = train_kl_loss / len(train_loader.dataset)
        
        # Step LR scheduler only after warmup
        if epoch > WARMUP_EPOCHS:
            scheduler.step(avg_loss)
        
        # Log to wandb
        wandb.log({
            'train/total_loss': avg_loss,
            'train/reconstruction_loss': avg_recon_loss,
            'train/kl_divergence': avg_kl_loss,
            'train/learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch
        })
        
        print(f'Epoch {epoch:3d}: Loss={avg_loss:.4f} (Recon={avg_recon_loss:.4f}, KL={avg_kl_loss:.4f}) LR={optimizer.param_groups[0]["lr"]:.2e}')
        
        # Validation and sampling every 10 epochs or at start (reduced frequency for speed)
        if epoch == 1 or epoch % 10 == 0:
            print(f"  → Generating samples and visualizations...")
            metrics, figures = validate_and_sample(
                model, val_sample, device, z, edge_index, epoch
            )
            
            # Log metrics
            wandb.log(metrics)
            
            # Log figures
            for fig_name, fig in figures.items():
                wandb.log({fig_name: wandb.Image(fig)})
                plt.close(fig)  # Close to free memory
            
            print(f"  → BondLoss: {metrics['val/bond_loss']:.4f}, "
                  f"Bond length (mean): {metrics['val/mean_bond_length_reconstructed']:.3f} Å")
        
        # Periodic checkpointing every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = f'vae_model_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'latent_dim': LATENT_DIM,
                    'num_atoms': ATOM_COUNT,
                    'atom_feature_dim': atom_feature_dim,
                    'visnet_hidden_channels': 128,
                    'decoder_hidden_dim': 256,
                    'decoder_num_layers': 8,
                }
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
    
    # Final validation
    print(f"\n{'='*60}")
    print("Training complete! Generating final samples...")
    print(f"{'='*60}\n")
    
    metrics, figures = validate_and_sample(
        model, val_sample, device, z, edge_index, EPOCHS
    )
    wandb.log(metrics)
    for fig_name, fig in figures.items():
        wandb.log({fig_name: wandb.Image(fig)})
        plt.close(fig)
    
    # Save final model
    model_save_path = 'vae_model.pth'
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'latent_dim': LATENT_DIM,
            'num_atoms': ATOM_COUNT,
            'atom_feature_dim': atom_feature_dim,
            'visnet_hidden_channels': VISNET_HIDDEN_CHANNELS,
            'decoder_hidden_dim': DECODER_HIDDEN_DIM,
            'decoder_num_layers': DECODER_NUM_LAYERS,
        }
    }, model_save_path)
    wandb.save(model_save_path)
    
    print(f"\n✅ Model saved to: {model_save_path}")
    print(f"   To load: model.load_state_dict(torch.load('{model_save_path}')['model_state_dict'])")
    
    wandb.finish()
    print("\nTraining finished! Check wandb for visualizations.")

if __name__ == "__main__":
    main()
