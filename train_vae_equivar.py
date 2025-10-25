import argparse
import logging
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import pathlib
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
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
  

from mse_training.vae_model_new import MolecularVAEMSE
from vae_utils import validate_and_sample

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
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
    LATENT_DIM = 32 
    EPOCHS = 1
    VISNET_HIDDEN_CHANNELS = 256
    ENCODER_NUM_LAYERS = 3
    DECODER_HIDDEN_DIM = 256
    DECODER_NUM_LAYERS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5  # Reduced learning rate for stability  
    NUM_WORKERS = 2  # Parallel data loading

    data = np.load(aib9.FULL_DATA).reshape(-1, 58, 3)
    data = data[0:10000]
    print(f"Original data shape: {data.shape}")
    
    # Check for NaN in input data
    if np.isnan(data).any():
        print("Warning: NaN detected in input data")
        data = np.nan_to_num(data, nan=0.0)
        print("Replaced NaN values with 0.0")
    
    cv = aib9.kai_calculator(data)
    mask = cv[:, 0] > 0
    filtered_data = data[mask]
    filtered_cv = cv[mask]
    print(f"CV shape: {cv.shape}")
    
    # Check for NaN in filtered data
    if np.isnan(filtered_data).any():
        print("Warning: NaN detected in filtered data")
        filtered_data = np.nan_to_num(filtered_data, nan=0.0)
        print("Replaced NaN values in filtered data with 0.0")

    train_data_np, test_data_np = train_test_split(filtered_data, test_size=0.2, random_state=42)
    
    # Set wandb to offline mode to avoid API key requirement
    import os
    os.environ["WANDB_MODE"] = "offline"
    
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

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA device: {torch.cuda.get_device_name()}')
        print(f'CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'Using MPS device: {device}')
        print('MPS memory: Not available via PyTorch API')
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
        # Check for NaN in this molecule's coordinates
        if np.isnan(train_data_np[i]).any():
            print(f"Warning: NaN detected in molecule {i}")
            train_data_np[i] = np.nan_to_num(train_data_np[i], nan=0.0)
            
        edge_index = radius_graph(torch.from_numpy(train_data_np[i]), r=5.0, batch=torch.zeros(train_data_np[i].shape[0], dtype=torch.long, device='cpu'))
        pos = torch.from_numpy(train_data_np[i]).float().to('cpu')
        
        # Final check for NaN in tensor
        if torch.isnan(pos).any():
            print(f"Warning: NaN detected in tensor for molecule {i}")
            pos = torch.nan_to_num(pos, nan=0.0)
            
        # No edge_index - let ViSNet use cutoff-based edge identification
        data = Data(z=z, pos=pos, edge_index=edge_index)
        train_data_list.append(data)
    train_loader = DataLoader(
        train_data_list,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to avoid macOS issues
        pin_memory=False  # Disable pin_memory when num_workers=0
    )
    
    # Only 10 unique atomic types in the dataset
    atom_feature_dim = 54 # Should be 10
    
    visnet_params = {
        'hidden_channels': VISNET_HIDDEN_CHANNELS,
        'num_layers': ENCODER_NUM_LAYERS,
        'num_rbf': 32,
        'cutoff': 5.0,  # Used for cutoff-based edge identification
        'max_z': max(ATOMIC_NUMBERS) + 1,
        'lmax':1
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
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

    # Test model with dummy data to check for NaN
    print("Testing model with dummy data...")
    dummy_data = Data(
        z=torch.tensor(ATOMIC_NUMBERS, dtype=torch.long).to(device),
        pos=torch.randn(58, 3).to(device),
        edge_index=torch.empty(2, 0, dtype=torch.long).to(device)
    )
    try:
        with torch.no_grad():
            test_output = model(dummy_data)
        print("Model test passed - no NaN in dummy forward pass")
    except Exception as e:
        print(f"Model test failed: {e}")
        print("Model has fundamental issues - reinitializing...")
        model.reset_parameters()
    
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
    avg_loss = 0
    avg_recon_loss = 0
    avg_kl_loss = 0
    grad_norm = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        if epoch >= 50:
            current_lr = LEARNING_RATE * 1/2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
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
            if batch_idx % 1 == 0:
                mu_norm = torch.mean(mu.pow(2)).item()
                log_var_mean = torch.mean(log_var).item()
                exp_log_var_mean = torch.mean(torch.exp(log_var)).item()
                print(f"  Debug - kl: {kl_div:.4f}, recon_loss: {recon_loss:.4f}")
                print(f"  Debug - μ²: {mu_norm:.4f}, log_var: {log_var_mean:.4f}, exp(log_var): {exp_log_var_mean:.4f}")

            # Clamp reconstruction loss to prevent explosion
            recon_loss = torch.clamp(recon_loss, max = 1000)  # Lower clamp for MSE
            kl_weight = 1

            loss = recon_loss + kl_weight*kl_div 
            
            
            # Check for NaN in loss before backpropagation
            if torch.isnan(loss):
                print("Warning: NaN detected in loss, skipping this batch")
                continue
                
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
              
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
        if epoch == 1 or epoch % 5 == 0:
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
        if epoch % 5 == 0:
            checkpoint_path = f'vae_model_pairwise_epoch{epoch}_small_model.pth'
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
    final_model_path = 'vae_model_pairwise_final_small_model_model.pth'
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
