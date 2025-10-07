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
    args = parser.parse_args()

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
        latent_dim=args.latent_dim,
        num_atoms=args.num_atoms,
        atom_feature_dim=args.atom_feature_dim,
        visnet_hidden_channels=args.visnet_hidden_channels,
        decoder_hidden_dim=args.decoder_hidden_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
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
            write_idx += b
            remaining -= b

    # Flush memmap
    mmap.flush()
    print(f"Saved {args.num_samples} molecules to {out_path}")


if __name__ == '__main__':
    main()


