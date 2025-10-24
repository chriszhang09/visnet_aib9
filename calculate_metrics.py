import numpy as np
import torch

# Set device for CUDA acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def _total_variation_distance(hist1, hist2):
    """
    Calculate the Total Variation (TV) distance between two normalized histograms.
    """
    return 0.5 * np.sum(np.abs(hist1 - hist2))


def identify_all_covalent_edges(topo):
    """
    Infers a complete set of covalent bonds (edges) for the AIB9 peptide
    by applying chemical rules to its topology.

    Args:
        topo (np.ndarray): The topology array with shape (num_atoms, 2),
                           where columns are [atom_name, residue_index].

    Returns:
        np.ndarray: An array of shape (2, num_edges) representing the
                    graph's covalent connectivity (edge_index).
    """
    atom_names = topo[:, 0]
    residue_indices = topo[:, 1].astype(int)
    
    # Create a map for easy lookup: {res_idx: {atom_name: original_idx}}
    atom_map = {}
    for i, (name, res_idx) in enumerate(zip(atom_names, residue_indices)):
        if res_idx not in atom_map:
            atom_map[res_idx] = {}
        atom_map[res_idx][name] = i

    edge_list = []
    sorted_residues = sorted(atom_map.keys())

    for res_idx in sorted_residues:
        res_atoms = atom_map[res_idx]
        
        # Rule for N-terminal ACE cap (Residue 1)
        if res_idx == 1 and all(k in res_atoms for k in ['CH3', 'C', 'O']):
            edge_list.append([res_atoms['CH3'], res_atoms['C']])
            edge_list.append([res_atoms['C'], res_atoms['O']])
            edge_list.append([res_atoms['C'], res_atoms['CH3']])
            edge_list.append([res_atoms['O'], res_atoms['C']])
            continue # Go to next residue

        # Rules for standard amino acid residues (AIB in this case)
        if all(k in res_atoms for k in ['N', 'CA', 'C']):
            edge_list.append([res_atoms['N'], res_atoms['CA']]) # N-CA bond
            edge_list.append([res_atoms['CA'], res_atoms['C']])  # CA-C bond
            edge_list.append([res_atoms['CA'], res_atoms['N']]) # N-CA bond
            edge_list.append([res_atoms['C'], res_atoms['CA']])  # CA-C bond
        
        if all(k in res_atoms for k in ['C', 'O']):
            edge_list.append([res_atoms['C'], res_atoms['O']])   # C=O bond
            edge_list.append([res_atoms['O'], res_atoms['C']])   # C=O bond
        if all(k in res_atoms for k in ['CA', 'CB1']):
            edge_list.append([res_atoms['CA'], res_atoms['CB1']])# Side chain
            edge_list.append([res_atoms['CB1'], res_atoms['CA']])# Side chain
        if all(k in res_atoms for k in ['CA', 'CB2']):
            edge_list.append([res_atoms['CA'], res_atoms['CB2']])# Side chain
            edge_list.append([res_atoms['CB2'], res_atoms['CA']])# Side chain
        # C-terminal capping: CA-CA2 bond (for residue 10)
        if all(k in res_atoms for k in ['CA', 'CA2']):
            edge_list.append([res_atoms['CA'], res_atoms['CA2']])# C-terminal cap
            edge_list.append([res_atoms['CA2'], res_atoms['CA']])# C-terminal cap
    # 2. Inter-Residue Peptide Bonds (C_i -> N_{i+1})
    for i in range(len(sorted_residues) - 1):
        res_idx1 = sorted_residues[i]
        res_idx2 = sorted_residues[i+1]
        
        if "C" in atom_map[res_idx1] and "N" in atom_map[res_idx2]:
            edge_list.append([atom_map[res_idx1]["C"], atom_map[res_idx2]["N"]])
            edge_list.append([atom_map[res_idx2]["N"], atom_map[res_idx1]["C"]])
    # Convert to NumPy array and make undirected
    edge_index = np.array(edge_list, dtype=int).T
    undirected_edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    
    return undirected_edge_index

def get_all_bond_lengths(coords_batch):
    """
    Calculates all pairwise bond lengths for a batch of molecules.
    This assumes a "completely connected" graph.
    
    Args:
        coords_batch (np.ndarray or torch.Tensor): Shape (num_molecules, num_atoms, 3)
        
    Returns:
        np.ndarray: A flat array of all pairwise distances.
    """
    # Convert to torch tensor if needed
    if isinstance(coords_batch, np.ndarray):
        coords_tensor = torch.from_numpy(coords_batch).to(device)
    else:
        coords_tensor = coords_batch.to(device)

    num_molecules, num_atoms, _ = coords_tensor.shape
    
    # Process all molecules at once on GPU for maximum speed
    if device.type == 'cuda':
        # Reshape to process all molecules simultaneously
        coords_flat = coords_tensor.view(-1, 3)  # [num_molecules * num_atoms, 3]
        
        # Calculate all pairwise distances at once
        all_dists = torch.pdist(coords_flat)
        
        # Convert back to numpy
        return all_dists.cpu().numpy()
    else:
        # CPU fallback - process molecules one by one
        all_lengths = []
        for i in range(num_molecules):
            dists = torch.pdist(coords_tensor[i])
            all_lengths.append(dists.cpu().numpy())
        return np.concatenate(all_lengths)


def get_all_bond_angles(coords_batch, topo_file_path):
    """
    Calculates all covalent bond angles for a batch of molecules.
    Uses the TOPO file to determine connectivity.
    
    Args:
        coords_batch (np.ndarray or torch.Tensor): Shape (num_molecules, num_atoms, 3)
        topo_file_path (str): Path to 'aib9_atom_info.npy'
        
    Returns:
        np.ndarray: A flat array of all bond angles in degrees.
    """
    # Convert to torch tensor if needed
    if isinstance(coords_batch, np.ndarray):
        coords_tensor = torch.from_numpy(coords_batch).to(device)
    else:
        coords_tensor = coords_batch.to(device)

    covalent_bonds = identify_all_covalent_edges(np.load(topo_file_path))

    # Build an adjacency list to find angles (i-j-k)
    adj = {i: [] for i in range(coords_batch.shape[1])}
    for i, j in covalent_bonds.T:
        adj[i].append(j)
        
    all_angles = []
    
    # Process each molecule
    for mol_idx in range(coords_batch.shape[0]):
        mol_coords = coords_tensor[mol_idx]  # Keep on GPU
        
        # Find all angle triplets (i, j, k) where i-j and j-k are bonds
        for j in range(mol_coords.shape[0]): # j is the central atom
            neighbors = adj[j]
            if len(neighbors) < 2:
                continue
                
            # Get all combinations of two distinct neighbors
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    i = neighbors[idx1]
                    k = neighbors[idx2]
                    
                    # Get coordinates (keep on GPU)
                    v_i = mol_coords[i]
                    v_j = mol_coords[j]
                    v_k = mol_coords[k]
                    
                    # Calculate vectors from center atom j
                    v_ji = v_i - v_j
                    v_jk = v_k - v_j
                    
                    # Normalize vectors
                    norm_ji = torch.norm(v_ji)
                    norm_jk = torch.norm(v_jk)
                    
                    if norm_ji == 0 or norm_jk == 0:
                        continue
                        
                    v_ji_norm = v_ji / norm_ji
                    v_jk_norm = v_jk / norm_jk
                    
                    # Calculate dot product (clip for numerical stability)
                    dot_prod = torch.clamp(torch.dot(v_ji_norm, v_jk_norm), -1.0, 1.0)
                    
                    # Get angle in degrees
                    angle = torch.degrees(torch.acos(dot_prod))
                    all_angles.append(angle.cpu().item())
                    
    return np.array(all_angles)


def calculate_tv_metrics(real_coords, gen_coords, topo_file,
                         bond_bins=100, bond_range=(0.5, 5.0),
                         angle_bins=100, angle_range=(0, 180)):
    """
    Calculates the Bond TV and Angle TV metrics.
    
    Args:
        real_coords (np.ndarray): Batch of real molecules (N_real, N_atoms, 3)
        gen_coords (np.ndarray): Batch of generated molecules (N_gen, N_atoms, 3)
        topo_file (str): Path to 'aib9_atom_info.npy'
        bond_bins (int): Number of bins for bond length histogram.
        bond_range (tuple): (min, max) for bond length histogram.
        angle_bins (int): Number of bins for angle histogram.
        angle_range (tuple): (min, max) for angle histogram.
        
    Returns:
        dict: A dictionary containing 'bond_tv' and 'angle_tv'.
    """
    
    # --- 1. Bond TV (All Pairwise) ---
    print("Calculating Bond TV (all pairwise)...")
    real_bonds = get_all_bond_lengths(real_coords)
    gen_bonds = get_all_bond_lengths(gen_coords)
    
    # Create normalized histograms (use torch for GPU acceleration if available)
    if device.type == 'cuda':
        real_bonds_tensor = torch.from_numpy(real_bonds).to(device)
        gen_bonds_tensor = torch.from_numpy(gen_bonds).to(device)
        
        # Use torch.histogram for GPU acceleration
        real_bond_hist, _ = torch.histogram(real_bonds_tensor, bins=bond_bins, range=bond_range, density=True)
        gen_bond_hist, _ = torch.histogram(gen_bonds_tensor, bins=bond_bins, range=bond_range, density=True)
        
        # Convert back to numpy for compatibility
        real_bond_hist = real_bond_hist.cpu().numpy()
        gen_bond_hist = gen_bond_hist.cpu().numpy()
    else:
        # CPU fallback
        real_bond_hist, _ = np.histogram(real_bonds, bins=bond_bins, range=bond_range, density=True)
        gen_bond_hist, _ = np.histogram(gen_bonds, bins=bond_bins, range=bond_range, density=True)
    
    # Normalize histograms to be probability distributions
    real_bond_hist = real_bond_hist / np.sum(real_bond_hist)
    gen_bond_hist = gen_bond_hist / np.sum(gen_bond_hist)
    
    bond_tv = _total_variation_distance(real_bond_hist, gen_bond_hist)
    
    
    # --- 2. Angle TV (Covalent) ---
    print("Calculating Angle TV (covalent)...")
    real_angles = get_all_bond_angles(real_coords, topo_file)
    gen_angles = get_all_bond_angles(gen_coords, topo_file)
    
    if len(real_angles) == 0 or len(gen_angles) == 0:
        print("Warning: Could not calculate angles. Skipping Angle TV.")
        angle_tv = np.nan
    else:
        # Create normalized histograms (use torch for GPU acceleration if available)
        if device.type == 'cuda':
            real_angles_tensor = torch.from_numpy(real_angles).to(device)
            gen_angles_tensor = torch.from_numpy(gen_angles).to(device)
            
            # Use torch.histogram for GPU acceleration
            real_angle_hist, _ = torch.histogram(real_angles_tensor, bins=angle_bins, range=angle_range, density=True)
            gen_angle_hist, _ = torch.histogram(gen_angles_tensor, bins=angle_bins, range=angle_range, density=True)
            
            # Convert back to numpy for compatibility
            real_angle_hist = real_angle_hist.cpu().numpy()
            gen_angle_hist = gen_angle_hist.cpu().numpy()
        else:
            # CPU fallback
            real_angle_hist, _ = np.histogram(real_angles, bins=angle_bins, range=angle_range, density=True)
            gen_angle_hist, _ = np.histogram(gen_angles, bins=angle_bins, range=angle_range, density=True)
        
        # Normalize histograms
        real_angle_hist = real_angle_hist / np.sum(real_angle_hist)
        gen_angle_hist = gen_angle_hist / np.sum(gen_angle_hist)
        
        angle_tv = _total_variation_distance(real_angle_hist, gen_angle_hist)

    return {
        "bond_tv": bond_tv,
        "angle_tv": angle_tv
    }


if __name__ == '__main__':
    # --- Example Usage ---
    # Create dummy data for demonstration
    
    # Assume aib9_atom_info.npy is in a subfolder 'aib9_lib'
    # NOTE: You MUST change this path to point to your actual file.
    TOPO_FILE_PATH = 'aib9_lib/aib9_atom_info.npy' 
    
    # Check if the file exists before running
    import os
    if not os.path.exists(TOPO_FILE_PATH):
        print(f"Warning: Dummy TOPO file not found at {TOPO_FILE_PATH}")
        print("Please update the TOPO_FILE_PATH variable to run the example.")
        
    else:
        print("--- Running TV Metrics Calculation Example ---")
        
        # Create two batches of "molecules"
        # In a real case, you'd load your test set and a batch of generated samples.
        from aib9_lib import aib9_tools as aib9
        print("Loading data...")
        real_data = np.load(aib9.FULL_DATA).reshape(-1, 58, 3)
        gen_data = np.load('molecule-coords_orig_large.npy').reshape(-1, 58, 3)
        
        print(f"Real data shape: {real_data.shape}")
        print(f"Generated data shape: {gen_data.shape}")
        print(f"Using device: {device}")
        
        # Add timing
        import time
        start_time = time.time()
        
        metrics = calculate_tv_metrics(real_data, gen_data, TOPO_FILE_PATH)
        
        end_time = time.time()
        print(f"\nCalculation completed in {end_time - start_time:.2f} seconds")
        
        print("\n--- Results ---")
        print(f"Bond TV (all-pairs):   {metrics['bond_tv']:.4f}")
        print(f"Angle TV (covalent): {metrics['angle_tv']:.4f}")
