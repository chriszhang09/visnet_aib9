"""
Test visualization functions with aib9 dataset.
This script tests the molecular visualization without needing a trained model.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pathlib

from aib9_lib import aib9_tools as aib9
from torch_geometric.data import Data


def visualize_molecule_3d(coords, atomic_numbers, title="Molecule", save_path=None):
    """Create a 3D visualization of a molecule."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for different atoms
    atom_colors = {1: 'lightgray', 6: 'black', 7: 'blue', 8: 'red', 
                   15: 'orange', 16: 'yellow', 17: 'green'}
    atom_sizes = {1: 30, 6: 100, 7: 100, 8: 100, 15: 120, 16: 120, 17: 120}
    atom_names = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 15: 'P', 16: 'S', 17: 'Cl'}
    
    # Count atoms by type
    atom_counts = {}
    for z in atomic_numbers:
        atom_counts[z] = atom_counts.get(z, 0) + 1
    
    # Plot atoms
    for i, (pos, z) in enumerate(zip(coords, atomic_numbers)):
        color = atom_colors.get(z, 'purple')
        size = atom_sizes.get(z, 50)
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=0.9, 
                  edgecolors='black', linewidths=1.5)
    
    # Add legend with atom counts
    legend_text = ", ".join([f"{atom_names.get(z, f'Z={z}')}: {count}" 
                            for z, count in sorted(atom_counts.items())])
    ax.text2D(0.05, 0.95, legend_text, transform=ax.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
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
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def compute_bond_lengths(coords, edge_index):
    """Compute bond lengths for a molecule."""
    bond_lengths = []
    bond_pairs = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < dst:  # Only count each bond once
            dist = torch.norm(coords[src] - coords[dst]).item()
            bond_lengths.append(dist)
            bond_pairs.append((src.item(), dst.item()))
    return bond_lengths, bond_pairs


def test_visualization():
    """Test visualization with aib9 dataset."""
    print("\n" + "="*70)
    print("Testing AIB9 Molecular Visualization")
    print("="*70 + "\n")
    
    # 1. Load data
    print("1. Loading AIB9 dataset...")
    project_path = pathlib.Path(__file__).resolve().parent
    TOPO_FILE = project_path / "aib9_lib/aib9_atom_info.npy"
    
    train_data_np = np.load(aib9.FULL_DATA)
    train_data_np = train_data_np.reshape(-1, 58, 3)
    print(f"   ✓ Loaded {train_data_np.shape[0]} conformations")
    print(f"   ✓ Each has {train_data_np.shape[1]} atoms")
    
    # 2. Load topology and create atomic numbers
    print("\n2. Loading topology and atomic numbers...")
    ATOMICNUMBER_MAPPING = {
        "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
        "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
    }
    
    ATOMIC_NUMBERS = []
    topo = np.load(TOPO_FILE)
    for i in range(topo.shape[0]):
        atom_name = topo[i, 0][0]
        if atom_name in ATOMICNUMBER_MAPPING:
            ATOMIC_NUMBERS.append(ATOMICNUMBER_MAPPING[atom_name])
        else:
            raise ValueError(f"Unknown atom name: {atom_name}")
    
    atomic_numbers = np.array(ATOMIC_NUMBERS)
    print(f"   ✓ Atomic composition: {dict(zip(*np.unique(atomic_numbers, return_counts=True)))}")
    
    # 3. Create edge index
    print("\n3. Creating covalent bond connectivity...")
    edges = aib9.identify_all_covalent_edges(topo)
    edge_index = torch.tensor(edges, dtype=torch.long)
    num_bonds = edge_index.shape[1] // 2  # Divide by 2 because edges are bidirectional
    print(f"   ✓ Created {num_bonds} covalent bonds ({edge_index.shape[1]} directed edges)")
    
    # 4. Visualize multiple conformations
    print("\n4. Creating visualizations...")
    output_dir = pathlib.Path("test_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Sample different conformations
    sample_indices = [0, len(train_data_np)//4, len(train_data_np)//2, 
                     3*len(train_data_np)//4, -1]
    
    all_bond_lengths = []
    
    for idx, sample_idx in enumerate(sample_indices):
        coords = train_data_np[sample_idx]
        
        # Visualize
        fig = visualize_molecule_3d(
            coords, atomic_numbers, 
            title=f"AIB9 Peptide - Conformation {sample_idx}",
            save_path=output_dir / f"molecule_sample_{idx}.png"
        )
        plt.close(fig)
        
        # Compute bond lengths
        bond_lengths, bond_pairs = compute_bond_lengths(
            torch.from_numpy(coords), edge_index
        )
        all_bond_lengths.extend(bond_lengths)
    
    print(f"   ✓ Created {len(sample_indices)} molecular visualizations")
    
    # 5. Bond length analysis
    print("\n5. Analyzing bond lengths...")
    all_bond_lengths = np.array(all_bond_lengths)
    
    print(f"   Bond length statistics:")
    print(f"   - Mean: {np.mean(all_bond_lengths):.3f} Å")
    print(f"   - Std:  {np.std(all_bond_lengths):.3f} Å")
    print(f"   - Min:  {np.min(all_bond_lengths):.3f} Å")
    print(f"   - Max:  {np.max(all_bond_lengths):.3f} Å")
    print(f"   - Median: {np.median(all_bond_lengths):.3f} Å")
    
    # Check if bonds are reasonable
    reasonable_bonds = (all_bond_lengths > 0.8) & (all_bond_lengths < 2.0)
    print(f"\n   ✓ {np.sum(reasonable_bonds)/len(all_bond_lengths)*100:.1f}% of bonds "
          f"in reasonable range (0.8-2.0 Å)")
    
    if np.sum(~reasonable_bonds) > 0:
        print(f"   ⚠ Warning: {np.sum(~reasonable_bonds)} bonds outside typical range")
    
    # 6. Create bond length histogram
    print("\n6. Creating bond length histogram...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(all_bond_lengths, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(all_bond_lengths), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(all_bond_lengths):.3f} Å')
    ax.set_xlabel('Bond Length (Å)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('AIB9 Peptide Bond Length Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "bond_length_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved bond length histogram")
    
    # 7. Create comparison visualization
    print("\n7. Creating multi-conformation comparison...")
    fig = plt.figure(figsize=(15, 5))
    
    for idx, sample_idx in enumerate(sample_indices[:3]):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        coords = train_data_np[sample_idx]
        
        atom_colors = {1: 'lightgray', 6: 'black', 7: 'blue', 8: 'red'}
        atom_sizes = {1: 20, 6: 70, 7: 70, 8: 70}
        
        for i, (pos, z) in enumerate(zip(coords, atomic_numbers)):
            color = atom_colors.get(z, 'purple')
            size = atom_sizes.get(z, 50)
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=0.8, 
                      edgecolors='black', linewidths=1)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Conf. {sample_idx}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "multi_conformation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved multi-conformation comparison")
    
    # 8. Summary
    print("\n" + "="*70)
    print("✅ Visualization Test Complete!")
    print("="*70)
    print(f"\nOutput files saved in: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - molecule_sample_*.png (5 individual conformations)")
    print("  - bond_length_distribution.png")
    print("  - multi_conformation_comparison.png")
    print("\nYou can now visually inspect these to verify:")
    print("  ✓ Molecules are rendered correctly")
    print("  ✓ Atom colors match types (C=black, N=blue, O=red, H=gray)")
    print("  ✓ Bond lengths are reasonable (~1.0-1.5 Å)")
    print("  ✓ Different conformations show structural variation")
    print("\nIf everything looks good, you're ready to run train_vae.py!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_visualization()

