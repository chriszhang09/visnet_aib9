#!/usr/bin/env python3
"""
Generate molecular coordinate samples from a trained VAE.

Loads a checkpoint (model.pth) and decodes coordinates from the prior.
Defaults are aligned with the MSE training setup.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
from aib9_lib import aib9_tools as aib9

try:
    from mse_training.vae_model_mse import MolecularVAEMSE as VAEModule
    DEFAULT_MODEL = "mse"
except Exception:
    # Fallback if mse module not available
    from vae_model import MolecularVAE as VAEModule  # type: ignore
    DEFAULT_MODEL = "original"



def build_atomic_numbers_from_topology(topo: np.ndarray) -> torch.Tensor:
    atomicnumber_mapping = {
        "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
        "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
    }
    atomic_numbers = []
    for i in range(topo.shape[0]):
        atom_name = topo[i, 0][0]
        if atom_name in atomicnumber_mapping:
            atomic_numbers.append(atomicnumber_mapping[atom_name])
        else:
            raise ValueError(f"Unknown atom name in topology: {atom_name}")
    return torch.tensor(atomic_numbers, dtype=torch.long)


def load_model(model_path: str,
               device: torch.device,
               latent_dim: int,
               num_atoms: int,
               atom_feature_dim: int,
               visnet_hidden_channels: int,
               encoder_num_layers: int,
               decoder_hidden_dim: int,
               decoder_num_layers: int,
               cutoff: float) -> VAEModule:
    visnet_kwargs = {
        'hidden_channels': visnet_hidden_channels,
        'num_layers': encoder_num_layers,
        'num_rbf': 32,
        'cutoff': cutoff,
        'max_z': 54,
    }

    model = VAEModule(
        latent_dim=latent_dim,
        num_atoms=num_atoms,
        atom_feature_dim=atom_feature_dim,
        visnet_hidden_channels=visnet_hidden_channels,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_num_layers=decoder_num_layers,
        visnet_kwargs=visnet_kwargs,
        cutoff=cutoff,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _build_rdkit_mol_from_coords(atomic_numbers: np.ndarray,
                                 coords: np.ndarray,
                                 distance_cutoff: float = 1.9):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdchem
    except Exception:
        return None

    num_atoms = len(atomic_numbers)
    rw = Chem.RWMol()
    for z in atomic_numbers:
        atom = Chem.Atom(int(z))
        rw.AddAtom(atom)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d <= distance_cutoff:
                try:
                    rw.AddBond(i, j, rdchem.BondType.SINGLE)
                except Exception:
                    pass

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        pass
    return mol


def _build_rdkit_mol_from_edges(atomic_numbers: np.ndarray,
                                coords: np.ndarray,
                                edge_index: np.ndarray):
    """Create an RDKit molecule using a provided covalent edge list.

    edge_index: shape [2, num_edges] with 0-based indices
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdchem
    except Exception:
        return None

    num_atoms = len(atomic_numbers)
    rw = Chem.RWMol()
    for z in atomic_numbers:
        atom = Chem.Atom(int(z))
        rw.AddAtom(atom)

    if edge_index is not None and edge_index.size > 0:
        rows, cols = edge_index
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i == j:
                continue
            try:
                rw.AddBond(int(i), int(j), rdchem.BondType.SINGLE)
            except Exception:
                pass

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        pass
    return mol


def compute_partial_charges_gasteiger(atomic_numbers: np.ndarray,
                                      coords: np.ndarray,
                                      edge_index: np.ndarray = None,
                                      distance_cutoff: float = 1.9) -> np.ndarray:
    try:
        from rdkit.Chem import AllChem
        if edge_index is not None:
            mol = _build_rdkit_mol_from_edges(atomic_numbers, coords, edge_index)
        else:
            mol = _build_rdkit_mol_from_coords(atomic_numbers, coords, distance_cutoff=distance_cutoff)
        if mol is None:
            return np.zeros(len(atomic_numbers), dtype=np.float32)
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for idx in range(mol.GetNumAtoms()):
            val = mol.GetAtomWithIdx(idx).GetProp('_GasteigerCharge')
            try:
                charges.append(float(val))
            except Exception:
                charges.append(0.0)
        return np.asarray(charges, dtype=np.float32)
    except Exception:
        return np.zeros(len(atomic_numbers), dtype=np.float32)


def save_charge_bar_plot(charges_batch: np.ndarray,
                         out_png: str,
                         title: str = 'Partial Charge Distribution') -> None:
    plt.figure(figsize=(8, 4))
    flat = charges_batch.reshape(-1)
    plt.hist(flat, bins=41, range=(-1.0, 1.0), color='#4C78A8', alpha=0.85)
    plt.xlabel('Partial charge (e)')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def wasserstein_w1_1d(samples_gen: np.ndarray,
                      samples_data: np.ndarray,
                      bin_width: float,
                      vmin: float,
                      vmax: float) -> float:
    """Compute W1 distance between two 1D distributions via CDF L1 on bins."""
    # Build bin edges
    if vmax <= vmin:
        vmax = vmin + bin_width
    num_bins = int(np.ceil((vmax - vmin) / bin_width))
    edges = vmin + np.arange(num_bins + 1) * bin_width

    hg, _ = np.histogram(samples_gen, bins=edges, range=(vmin, vmax), density=False)
    hd, _ = np.histogram(samples_data, bins=edges, range=(vmin, vmax), density=False)

    hg = hg.astype(np.float64)
    hd = hd.astype(np.float64)
    if hg.sum() > 0:
        hg /= hg.sum()
    if hd.sum() > 0:
        hd /= hd.sum()

    cdf_g = np.cumsum(hg)
    cdf_d = np.cumsum(hd)
    w1 = np.sum(np.abs(cdf_d - cdf_g)) * bin_width
    return float(w1)


def compute_bl_w1_for_batch(coords_batch: np.ndarray,
                            topo: np.ndarray,
                            data_coords_ref: np.ndarray,
                            bin_width: float = 0.01) -> float:
    """Compute BL (bond-length) W1 between generated batch and reference data."""
    edges = aib9.identify_all_covalent_edges(topo)
    row, col = edges[0], edges[1]

    # Generated bond lengths
    gen_lengths = []
    for i in range(coords_batch.shape[0]):
        g = coords_batch[i]
        d = np.linalg.norm(g[row] - g[col], axis=1)
        gen_lengths.append(d)
    gen_lengths = np.concatenate(gen_lengths, axis=0)

    # Reference bond lengths
    ref_lengths = []
    for i in range(data_coords_ref.shape[0]):
        r = data_coords_ref[i]
        d = np.linalg.norm(r[row] - r[col], axis=1)
        ref_lengths.append(d)
    ref_lengths = np.concatenate(ref_lengths, axis=0)

    vmin = float(min(np.min(gen_lengths), np.min(ref_lengths)))
    vmax = float(max(np.max(gen_lengths), np.max(ref_lengths)))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 5.0

    return wasserstein_w1_1d(gen_lengths, ref_lengths, bin_width=bin_width, vmin=vmin, vmax=vmax)


def compute_bond_lengths_flat(coords_array: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute all covalent bond lengths for an array of coords and flatten.

    coords_array: [K, N, 3] or [N, 3]
    edges: [2, E]
    Returns: [K*E] or [E]
    """
    row, col = edges[0], edges[1]
    if coords_array.ndim == 2:
        diffs = coords_array[row] - coords_array[col]
        return np.linalg.norm(diffs, axis=1)
    lens = []
    for i in range(coords_array.shape[0]):
        c = coords_array[i]
        diffs = c[row] - c[col]
        lens.append(np.linalg.norm(diffs, axis=1))
    return np.concatenate(lens, axis=0)


def save_bl_w1_plot(gen_lengths: np.ndarray,
                    ref_lengths: np.ndarray,
                    w1_value: float,
                    out_png: str,
                    bin_width: float = 0.01) -> None:
    """Save overlaid histograms of BL distributions with W1 in the title."""
    vmin = float(min(gen_lengths.min(), ref_lengths.min()))
    vmax = float(max(gen_lengths.max(), ref_lengths.max()))
    # Build common bins
    if vmax <= vmin:
        vmax = vmin + bin_width
    num_bins = int(np.ceil((vmax - vmin) / bin_width))
    edges = vmin + np.arange(num_bins + 1) * bin_width

    plt.figure(figsize=(7, 4))
    plt.hist(ref_lengths, bins=edges, color='#4C78A8', alpha=0.6, label='Data')
    plt.hist(gen_lengths, bins=edges, color='#F58518', alpha=0.6, label='Generated')
    plt.xlabel('Bond length (Å)')
    plt.ylabel('Count')
    plt.title(f'BL distributions — W1={w1_value:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate molecules from trained VAE")
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to .pth checkpoint')
    parser.add_argument('--output', type=str, default='generated_coords.npy', help='Output .npy path')
    parser.add_argument('--num_samples', type=int, default=100_000, help='Number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=128, help='Decode batch size (number of molecules)')
    parser.add_argument('--latent_dim', type=int, default=24)
    parser.add_argument('--num_atoms', type=int, default=58)
    parser.add_argument('--atom_feature_dim', type=int, default=10)
    parser.add_argument('--visnet_hidden_channels', type=int, default=128)
    parser.add_argument('--encoder_num_layers', type=int, default=3)
    parser.add_argument('--decoder_hidden_dim', type=int, default=256)
    parser.add_argument('--decoder_num_layers', type=int, default=3)
    parser.add_argument('--cutoff', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--save_charge_plots', action='store_true', help='Compute partial charges (RDKit) and save a histogram per batch')
    parser.add_argument('--charge_distance_cutoff', type=float, default=1.9, help='Distance cutoff (Å) for bond inference when computing charges')
    parser.add_argument('--use_covalent_edges', action='store_true', help='Use identify_all_covalent_edges(topo) for bonding when computing charges')
    parser.add_argument('--compute_w1', action='store_true', help='Compute BL (bond-length) Wasserstein-1 per batch against training data')
    parser.add_argument('--w1_ref_samples', type=int, default=1000, help='Number of reference molecules from data for W1')
    parser.add_argument('--w1_bin_width', type=float, default=0.01, help='Bin width (Å) for BL W1')
    args = parser.parse_args()

    
    # Training parameters
    ATOM_COUNT = 58
    COORD_DIM = 3
    ORIGINAL_DIM = ATOM_COUNT * COORD_DIM  
    LATENT_DIM = 128 
    EPOCHS = 110
    VISNET_HIDDEN_CHANNELS = 256
    ENCODER_NUM_LAYERS = 9
    DECODER_HIDDEN_DIM = 256
    DECODER_NUM_LAYERS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 4e-4  
    NUM_WORKERS = 2  # Parallel data loading
    # Device & seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load topology and atomic numbers for one-hot
    topo = np.load(aib9.TOPO_FILE)
    atomic_numbers = build_atomic_numbers_from_topology(topo)
    num_atoms = args.num_atoms
    if len(atomic_numbers) != num_atoms:
        raise ValueError(f"Topology has {len(atomic_numbers)} atoms, expected {num_atoms}")

    # Prepare constant inputs
    atomic_numbers_device = atomic_numbers.to(device)
    atom_types_one_hot = F.one_hot(atomic_numbers_device, num_classes=args.atom_feature_dim).float()
    batch_template = torch.zeros(num_atoms, dtype=torch.long, device=device)

    # Load model
    model = load_model(
        model_path=args.model_path,
        device=device,
        latent_dim=LATENT_DIM,
        num_atoms=ATOM_COUNT,
        atom_feature_dim=54,
        visnet_hidden_channels=VISNET_HIDDEN_CHANNELS,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,
        encoder_num_layers=ENCODER_NUM_LAYERS,
        decoder_num_layers=DECODER_NUM_LAYERS,
        cutoff=args.cutoff,
    )

    # Allocate output array (saved incrementally to avoid peak memory)
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Use memmap to avoid large RAM spikes
    mmap = np.memmap(out_path, dtype=np.float32, mode='w+', shape=(args.num_samples, num_atoms, 3))

    model.eval()
    with torch.no_grad():
        write_idx = 0
        remaining = args.num_samples

        # Preload reference data for W1 if requested
        data_coords_ref = None
        if args.compute_w1:
            data_np = np.load(aib9.FULL_DATA)
            data_np = data_np.reshape(-1, num_atoms, 3)
            m = min(args.w1_ref_samples, data_np.shape[0])
            data_coords_ref = data_np[:m]
        while remaining > 0:
            b = min(args.batch_size, remaining)

            # Sample z for each molecule in the batch (shape [b, latent_dim])
            z = torch.randn(b, args.latent_dim, device=device)

            # Decode each molecule; decoder expects per-molecule inputs
            batch_coords = []
            for i in range(b):
                coords = model.decoder(
                    z[i].unsqueeze(0),               # [1, latent_dim]
                    atom_types_one_hot[:num_atoms],   # [num_atoms, atom_feature_dim]
                    None,                             # use cutoff-based edges
                    batch_template                     # [num_atoms]
                )  # -> [num_atoms, 3]
                batch_coords.append(coords.unsqueeze(0))

            coords_tensor = torch.cat(batch_coords, dim=0)  # [b, num_atoms, 3]
            coords_np = coords_tensor.detach().cpu().numpy()

            mmap[write_idx:write_idx + b] = coords_np

            if args.save_charge_plots:
                charges_list = []
                z_np = atomic_numbers.detach().cpu().numpy()
                edge_index_np = None
                if args.use_covalent_edges:
                    # Use covalent edges from topology (0-based indices expected)
                    edges = aib9.identify_all_covalent_edges(topo)
                    edge_index_np = edges
                for i in range(b):
                    charges_i = compute_partial_charges_gasteiger(
                        z_np, coords_np[i], edge_index=edge_index_np, distance_cutoff=args.charge_distance_cutoff
                    )
                    charges_list.append(charges_i)
                charges_batch = np.stack(charges_list, axis=0)
                png_path = os.path.splitext(out_path)[0] + f"_charges_{write_idx}_{write_idx + b - 1}.png"
                save_charge_bar_plot(charges_batch, png_path, title=f'Partial Charge Distribution (molecules {write_idx}-{write_idx + b - 1})')

            if args.compute_w1 and data_coords_ref is not None:
                # Compute W1 and also save an overlaid histogram
                edges = aib9.identify_all_covalent_edges(topo)
                gen_lengths = compute_bond_lengths_flat(coords_np, edges)
                ref_lengths = compute_bond_lengths_flat(data_coords_ref, edges)
                w1_bl = wasserstein_w1_1d(
                    gen_lengths, ref_lengths,
                    bin_width=args.w1_bin_width,
                    vmin=float(min(gen_lengths.min(), ref_lengths.min())),
                    vmax=float(max(gen_lengths.max(), ref_lengths.max()))
                )
                print(f"Batch {write_idx}-{write_idx + b - 1}: BL W1 = {w1_bl:.4f}")
                w1_png = os.path.splitext(out_path)[0] + f"_w1_{write_idx}_{write_idx + b - 1}.png"
                save_bl_w1_plot(gen_lengths, ref_lengths, w1_bl, w1_png, bin_width=args.w1_bin_width)
            write_idx += b
            remaining -= b

    # Flush memmap
    mmap.flush()
    print(f"Saved {args.num_samples} molecules to {out_path}")


if __name__ == '__main__':
    main()


