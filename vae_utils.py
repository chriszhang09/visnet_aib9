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
        
        # Create batch tensor for single molecule
        batch = torch.zeros(val_data.z.size(0), dtype=torch.long, device=device)
        reconstructed = model.decoder(z, atom_types_one_hot, val_data.edge_index, batch)
        
        # Move to CPU for visualization
        original_coords = val_data.pos.cpu().numpy()
        # PyG decoder returns [num_atoms, 3] directly, not [batch_size, num_atoms, 3]
        reconstructed_coords = reconstructed.cpu().numpy()
        atomic_nums = atomic_numbers.cpu().numpy()
        
        # 2. Generate from isotropic Gaussian prior
        random_z = model.sample_prior(1, device)
        generated = model.decoder(random_z, atom_types_one_hot[:model.decoder.num_atoms], 
                                  val_data.edge_index, batch)
        # PyG decoder returns [num_atoms, 3] directly
        generated_coords = generated.cpu().numpy()
        
        # 3. Compute metrics (E(3)-invariant bond-distance MSE)
        edge_index_cpu = edge_index.cpu()
        row, col = edge_index_cpu[0], edge_index_cpu[1]
        pred_coords_t = torch.from_numpy(reconstructed_coords)
        target_coords_t = torch.from_numpy(original_coords)
        pred_bond_dist = torch.norm(pred_coords_t[row] - pred_coords_t[col], dim=-1)
        target_bond_dist = torch.norm(target_coords_t[row] - target_coords_t[col], dim=-1)
        bond_loss = F.mse_loss(pred_bond_dist, target_bond_dist).item()
        
        # Compute bond lengths
        original_bonds = compute_bond_lengths(torch.from_numpy(original_coords), edge_index_cpu)
        recon_bonds = compute_bond_lengths(torch.from_numpy(reconstructed_coords), edge_index_cpu)
        gen_bonds = compute_bond_lengths(torch.from_numpy(generated_coords), edge_index_cpu)
        
        # 4. Create visualizations
        fig_orig = visualize_molecule_3d(original_coords, atomic_nums, 
                                         f"Original (Epoch {epoch})")
        fig_recon = visualize_molecule_3d(reconstructed_coords, atomic_nums, 
                                          f"Reconstructed (Epoch {epoch}, BondLoss={bond_loss:.4f})")
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
            'val/bond_loss': bond_loss,
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
