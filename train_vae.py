import argparse
import logging
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import pathlib


from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.data import Data

from aib9_lib import aib9_tools as aib9
from vae_decoder import EGNNDecoder
from vae_model import MolecularVAE, vae_loss_function
from visnet_vae_encoder import ViSNetEncoder

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from pathlib import Path
import sys
import wandb
from aib9_lib import aib9_tools as aib9
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.cuda.amp import autocast, GradScaler


def visualize_molecule_3d(coords, atomic_numbers, title="Molecule"):
    """Create a 3D visualization of a molecule."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for different atoms
    atom_colors = {1: 'white', 6: 'gray', 7: 'blue', 8: 'red', 
                   15: 'orange', 16: 'yellow', 17: 'green'}
    atom_sizes = {1: 20, 6: 70, 7: 70, 8: 70, 15: 100, 16: 100, 17: 100}
    
    # Plot atoms
    for i, (pos, z) in enumerate(zip(coords, atomic_numbers)):
        color = atom_colors.get(z, 'purple')
        size = atom_sizes.get(z, 50)
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=0.8, edgecolors='black')
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([coords[:, 0].max()-coords[:, 0].min(),
                          coords[:, 1].max()-coords[:, 1].min(),
                          coords[:, 2].max()-coords[:, 2].min()]).max() / 2.0
    mid_x = (coords[:, 0].max()+coords[:, 0].min()) * 0.5
    mid_y = (coords[:, 1].max()+coords[:, 1].min()) * 0.5
    mid_z = (coords[:, 2].max()+coords[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return fig


def compute_bond_lengths(coords, edge_index):
    """Compute bond lengths for a molecule."""
    bond_lengths = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < dst:  # Only count each bond once
            dist = torch.norm(coords[src] - coords[dst]).item()
            bond_lengths.append(dist)
    return bond_lengths


def validate_and_sample(model, val_data, device, atomic_numbers, edge_index, epoch):
    """Generate samples and visualizations for validation."""
    model.eval()
    
    with torch.no_grad():
        # 1. Reconstruction: encode and decode a real molecule
        val_molecule = val_data.to(device)
        mu, log_var = model.encoder(val_molecule)
        z = model.reparameterize(mu, log_var)
        
        atom_types_one_hot = F.one_hot(val_data.z.long(), 
                                       num_classes=model.decoder.atom_feature_dim).float().to(device)
        reconstructed = model.decoder(z, atom_types_one_hot)
        
        # Move to CPU for visualization
        original_coords = val_data.pos.cpu().numpy()
        reconstructed_coords = reconstructed[0].cpu().numpy()
        atomic_nums = atomic_numbers.cpu().numpy()
        
        # 2. Generate from random latent code
        random_z = torch.randn_like(z[:1])
        generated = model.decoder(random_z, atom_types_one_hot[:model.decoder.num_atoms])
        generated_coords = generated[0].cpu().numpy()
        
        # 3. Compute metrics
        rmsd = np.sqrt(np.mean((original_coords - reconstructed_coords)**2))
        
        # Compute bond lengths
        original_bonds = compute_bond_lengths(torch.from_numpy(original_coords), edge_index)
        recon_bonds = compute_bond_lengths(torch.from_numpy(reconstructed_coords), edge_index)
        gen_bonds = compute_bond_lengths(torch.from_numpy(generated_coords), edge_index)
        
        # 4. Create visualizations
        fig_orig = visualize_molecule_3d(original_coords, atomic_nums, 
                                         f"Original (Epoch {epoch})")
        fig_recon = visualize_molecule_3d(reconstructed_coords, atomic_nums, 
                                          f"Reconstructed (Epoch {epoch}, RMSD={rmsd:.3f}Å)")
        fig_gen = visualize_molecule_3d(generated_coords, atomic_nums, 
                                        f"Generated from Random Z (Epoch {epoch})")
        
        # 5. Bond length comparison plot
        fig_bonds, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(original_bonds, bins=20, alpha=0.5, label='Original', color='blue')
        ax1.hist(recon_bonds, bins=20, alpha=0.5, label='Reconstructed', color='red')
        ax1.set_xlabel('Bond Length (Å)')
        ax1.set_ylabel('Count')
        ax1.set_title('Reconstruction Bond Lengths')
        ax1.legend()
        ax1.axvline(np.mean(original_bonds), color='blue', linestyle='--', linewidth=2)
        ax1.axvline(np.mean(recon_bonds), color='red', linestyle='--', linewidth=2)
        
        ax2.hist(original_bonds, bins=20, alpha=0.5, label='Original', color='blue')
        ax2.hist(gen_bonds, bins=20, alpha=0.5, label='Generated', color='green')
        ax2.set_xlabel('Bond Length (Å)')
        ax2.set_ylabel('Count')
        ax2.set_title('Generated Bond Lengths')
        ax2.legend()
        ax2.axvline(np.mean(original_bonds), color='blue', linestyle='--', linewidth=2)
        ax2.axvline(np.mean(gen_bonds), color='green', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        
        metrics = {
            'val/rmsd': rmsd,
            'val/mean_bond_length_original': np.mean(original_bonds),
            'val/mean_bond_length_reconstructed': np.mean(recon_bonds),
            'val/mean_bond_length_generated': np.mean(gen_bonds),
            'val/std_bond_length_original': np.std(original_bonds),
            'val/std_bond_length_reconstructed': np.std(recon_bonds),
            'val/std_bond_length_generated': np.std(gen_bonds),
            'val/latent_mean': mu.mean().item(),
            'val/latent_std': mu.std().item(),
        }
        
        figures = {
            'molecules/original': fig_orig,
            'molecules/reconstructed': fig_recon,
            'molecules/generated': fig_gen,
            'molecules/bond_lengths': fig_bonds,
        }
        
    model.train()
    return metrics, figures


def main():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

    # Initialize Weights & Biases
    wandb.init(
        project="aib9-vae",
        config={
            "atom_count": 58,
            "latent_dim": 30,
            "epochs": 50,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "encoder_hidden": 256,
            "encoder_layers": 9,
            "decoder_hidden": 256,
            "decoder_layers": 8,
        }
    )

    ATOM_COUNT = 58
    COORD_DIM = 3
    ORIGINAL_DIM = ATOM_COUNT * COORD_DIM  
    LATENT_DIM = 30 
    EPOCHS = 50
    BATCH_SIZE = 256  # Increased from 128 (V100 can handle much more!)
    LEARNING_RATE = 1e-3  # Scale learning rate with batch size
    NUM_WORKERS = 4  # Parallel data loading

    train_data_np = np.load(aib9.FULL_DATA)
    train_data_np = train_data_np.reshape(-1, 58, 3)

    
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

    z = torch.tensor(ATOMIC_NUMBERS, dtype=torch.long)

    # Create a list of Data objects, one for each molecule


    
    edges = aib9.identify_all_covalent_edges(topo)
    # edges is already in shape [2, num_edges], no need to transpose
    edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
    train_data_list = []
    for i in range(train_data_np.shape[0]):
        pos = torch.from_numpy(train_data_np[i]).float()
        # Add edge_index to each data object
        data = Data(z=z, pos=pos, edge_index=edge_index) 
        train_data_list.append(data)
    train_loader = DataLoader(
        train_data_list, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,  # Parallel data loading
        pin_memory=True,  # Faster CPU-GPU transfer
        persistent_workers=True if NUM_WORKERS > 0 else False  # Keep workers alive
    )
    
    # Determine the maximum atomic number to set the correct atom_feature_dim for one-hot encoding
    max_atomic_number = max(ATOMIC_NUMBERS)  # Should be 53 for Iodine
    atom_feature_dim = max_atomic_number + 1  # Need +1 for zero-indexing (0 to max_z)
    
    # Create the encoder with proper ViSNet parameters
    # Note: cutoff is no longer used since we provide explicit edge_index (covalent bonds)
    # The model will use the edges defined by identify_all_covalent_edges()
    visnet_params = {
        'hidden_channels': 256,
        'num_layers': 9,
        'num_rbf': 32,
        'cutoff': 5.0,  # Kept for compatibility but not used with edge_index
        'max_z': max_atomic_number + 1,
    }
    
    # Enhanced decoder configuration
    model = MolecularVAE(
        latent_dim=LATENT_DIM, 
        num_atoms=ATOM_COUNT, 
        atom_feature_dim=atom_feature_dim,
        visnet_hidden_channels=128,
        decoder_hidden_dim=256,      # Larger hidden dimension for decoder
        decoder_num_layers=8,         # More layers for better reconstruction
        edge_index_template=edge_index,  # Use covalent bonds in decoder
        visnet_kwargs=visnet_params
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Mixed precision training - 2-3x speedup on V100!
    scaler = GradScaler('cuda')
    use_amp = torch.cuda.is_available()  # Use AMP if CUDA available
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Watch model with wandb
    wandb.watch(model, log='all', log_freq=200)  # Reduced log frequency
    
    # Keep a fixed validation sample for visualization
    val_sample = train_data_list[0]  # Use first sample for consistent visualization
    
    print(f"\n{'='*60}")
    print(f"Starting Training - {len(train_data_list)} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            molecules = data.to(device, non_blocking=True)  # Async GPU transfer
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Mixed precision forward pass
            with autocast(enabled=use_amp):
                recon_batch, mu, logvar = model(molecules)
                
                # Separate reconstruction and KL losses
                recon_loss = F.mse_loss(recon_batch.view(-1, 3), molecules.pos)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_div
            
            # Mixed precision backward pass
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_div.item()

        # Compute average losses
        avg_loss = train_loss / len(train_loader.dataset)
        avg_recon_loss = train_recon_loss / len(train_loader.dataset)
        avg_kl_loss = train_kl_loss / len(train_loader.dataset)
        
        # Step learning rate scheduler
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
            
            print(f"  → RMSD: {metrics['val/rmsd']:.3f} Å, "
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
            'visnet_hidden_channels': 128,
            'decoder_hidden_dim': 256,
            'decoder_num_layers': 8,
        }
    }, model_save_path)
    wandb.save(model_save_path)
    
    print(f"\n✅ Model saved to: {model_save_path}")
    print(f"   To load: model.load_state_dict(torch.load('{model_save_path}')['model_state_dict'])")
    
    wandb.finish()
    print("\nTraining finished! Check wandb for visualizations.")

if __name__ == "__main__":
    main()
