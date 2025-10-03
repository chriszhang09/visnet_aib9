import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_bond_lengths(coords, edge_index):
    """Compute bond lengths for a molecule."""
    bond_lengths = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < dst:  # Only count each bond once
            dist = torch.norm(coords[src] - coords[dst]).item()
            bond_lengths.append(dist)
    return bond_lengths


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
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=0.8)
    
    # Plot bonds
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < dst:  # Only plot each bond once
            start = coords[src]
            end = coords[dst]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   'k-', alpha=0.6, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return fig


def validate_and_sample(model, val_data, device, atomic_numbers, edge_index, epoch):
    """
    MSE-specific validation and sampling function.
    Uses pairwise distance loss for validation metrics.
    """
    model.eval()
    
    with torch.no_grad():
        # Move data to device
        val_data = val_data.to(device)
        
        # Encode the validation sample
        mu, log_var = model.encoder(val_data)
        
        # Sample from the learned distribution
        z = model.reparameterize(mu, log_var)
        
        # Decode to get reconstructed coordinates
        atom_types_one_hot = F.one_hot(val_data.z.long(), num_classes=model.decoder.atom_feature_dim).float()
        reconstructed = model.decoder(z, atom_types_one_hot, val_data.edge_index, val_data.batch)
        
        # Generate from prior
        z_prior = model.sample_prior(1, device)
        generated = model.decoder(z_prior, atom_types_one_hot, val_data.edge_index, val_data.batch)
        
        # Convert to numpy for visualization
        reconstructed_coords = reconstructed.cpu().numpy()
        atomic_nums = atomic_numbers.cpu().numpy() if isinstance(atomic_numbers, torch.Tensor) else np.array(atomic_numbers)
        
        # Generate from prior
        generated_coords = generated.cpu().numpy()
        
        # Compute validation metrics using pairwise distance loss
        from train_vae_mse import pairwise_distance_loss
        
        # Compute pairwise distance loss for validation
        val_loss = pairwise_distance_loss(reconstructed, val_data.pos)
        
        # Compute bond-based E(3) invariant loss for comparison
        edge_index_cpu = edge_index.cpu()
        row, col = edge_index_cpu[0], edge_index_cpu[1]
        
        pred_bond_dist = torch.norm(reconstructed[row] - reconstructed[col], dim=-1)
        target_bond_dist = torch.norm(val_data.pos[row] - val_data.pos[col], dim=-1)
        bond_loss = F.mse_loss(pred_bond_dist, target_bond_dist)
        
        metrics = {
            'val_pairwise_loss': val_loss.item(),
            'val_bond_loss': bond_loss.item(),
            'mu_norm': torch.mean(mu.pow(2)).item(),
            'log_var_mean': torch.mean(log_var).item(),
            'exp_log_var_mean': torch.mean(torch.exp(log_var)).item()
        }
        
        # Create visualizations
        figures = {}
        
        # Reconstructed molecule
        fig_recon = visualize_molecule_3d(
            reconstructed_coords, atomic_nums, 
            f"Reconstructed (Epoch {epoch})"
        )
        figures['reconstructed_molecule'] = fig_recon
        
        # Generated molecule
        fig_gen = visualize_molecule_3d(
            generated_coords, atomic_nums,
            f"Generated from Prior (Epoch {epoch})"
        )
        figures['generated_molecule'] = fig_gen
        
        # Bond length comparison
        target_bonds = compute_bond_lengths(val_data.pos.cpu(), edge_index_cpu)
        recon_bonds = compute_bond_lengths(reconstructed.cpu(), edge_index_cpu)
        
        fig_bonds = plt.figure(figsize=(10, 6))
        plt.hist(target_bonds, bins=20, alpha=0.7, label='Target', color='blue')
        plt.hist(recon_bonds, bins=20, alpha=0.7, label='Reconstructed', color='red')
        plt.xlabel('Bond Length (Å)')
        plt.ylabel('Frequency')
        plt.title(f'Bond Length Distribution (Epoch {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        figures['bond_distribution'] = fig_bonds
        
        # Loss components over time
        fig_loss = plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.bar(['Pairwise Loss', 'Bond Loss'], [val_loss.item(), bond_loss.item()])
        plt.title('Validation Losses')
        plt.ylabel('Loss Value')
        
        plt.subplot(1, 2, 2)
        plt.bar(['μ²', 'log_var', 'exp(log_var)'], 
                [metrics['mu_norm'], metrics['log_var_mean'], metrics['exp_log_var_mean']])
        plt.title('Latent Space Statistics')
        plt.ylabel('Value')
        
        plt.tight_layout()
        figures['loss_analysis'] = fig_loss
        
    model.train()
    return metrics, figures
