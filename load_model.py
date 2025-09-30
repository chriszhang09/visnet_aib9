"""
Helper script to load a saved VAE model and generate samples.
Usage: python load_model.py [model_path]
"""
import torch
import numpy as np
import sys
from vae_model import MolecularVAE
import torch.nn.functional as F


def load_vae_model(model_path='vae_model.pth', device='cpu'):
    """
    Load a saved VAE model from checkpoint.
    
    Args:
        model_path (str): Path to saved model checkpoint
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded MolecularVAE model
        config: Model configuration dictionary
        epoch: Training epoch the model was saved at
    """
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration
    config = checkpoint['config']
    epoch = checkpoint.get('epoch', 'unknown')
    
    print(f"Model trained for {epoch} epochs")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    
    # Create model with same architecture
    model = MolecularVAE(
        latent_dim=config['latent_dim'],
        num_atoms=config['num_atoms'],
        atom_feature_dim=config['atom_feature_dim'],
        visnet_hidden_channels=config.get('visnet_hidden_channels', 128),
        decoder_hidden_dim=config.get('decoder_hidden_dim', 128),
        decoder_num_layers=config.get('decoder_num_layers', 6),
        edge_index_template=None,  # Will need to provide if using
        visnet_kwargs=config.get('visnet_kwargs', {
            'hidden_channels': config.get('visnet_hidden_channels', 128),
            'num_layers': 9,
            'num_rbf': 32,
            'cutoff': 5.0,
            'max_z': config['atom_feature_dim'],
        })
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    
    return model, config, epoch


def generate_samples(model, num_samples=5, device='cpu'):
    """
    Generate molecular conformations from random latent vectors.
    
    Args:
        model: Loaded MolecularVAE model
        num_samples (int): Number of samples to generate
        device (str): Device to run on
    
    Returns:
        numpy array of shape (num_samples, num_atoms, 3)
    """
    print(f"\nGenerating {num_samples} samples from random latent codes...")
    
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        latent_dim = model.encoder.latent_dim
        num_atoms = model.decoder.num_atoms
        atom_feature_dim = model.decoder.atom_feature_dim
        
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Create dummy atom types (you'll need actual atom types for real use)
        # For now, just use zeros as placeholder
        atom_types = torch.zeros(num_samples * num_atoms, atom_feature_dim).to(device)
        atom_types[:, 0] = 1  # Set first feature to 1
        
        # Generate coordinates
        generated_coords = model.decoder(z, atom_types)
        
        # Convert to numpy
        samples = generated_coords.cpu().numpy()
    
    print(f"✓ Generated {num_samples} conformations")
    print(f"  Shape: {samples.shape}")
    
    return samples


def reconstruct_molecule(model, coords, atomic_numbers, device='cpu'):
    """
    Encode and decode a molecule (reconstruction).
    
    Args:
        model: Loaded MolecularVAE model
        coords: Coordinates (num_atoms, 3)
        atomic_numbers: Atomic numbers for each atom
        device (str): Device to run on
    
    Returns:
        reconstructed coordinates, latent code (mu, log_var)
    """
    from torch_geometric.data import Data
    
    print("\nReconstructing molecule...")
    
    model.eval()
    with torch.no_grad():
        # Create data object
        pos = torch.from_numpy(coords).float().to(device)
        z = torch.from_numpy(atomic_numbers).long().to(device)
        data = Data(z=z, pos=pos)
        
        # Encode
        mu, log_var = model.encoder(data)
        print(f"  Latent code: mu shape={mu.shape}, log_var shape={log_var.shape}")
        
        # Decode
        z_sample = model.reparameterize(mu, log_var)
        atom_types_one_hot = F.one_hot(data.z.long(), 
                                       num_classes=model.decoder.atom_feature_dim).float()
        reconstructed = model.decoder(z_sample, atom_types_one_hot)
        
        reconstructed_coords = reconstructed[0].cpu().numpy()
        
        # Compute RMSD
        rmsd = np.sqrt(np.mean((coords - reconstructed_coords)**2))
        print(f"  Reconstruction RMSD: {rmsd:.4f} Å")
    
    return reconstructed_coords, (mu, log_var)


def main():
    """Example usage of loading and using the model."""
    
    # Parse command line arguments
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'vae_model.pth'
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model
    model, config, epoch = load_vae_model(model_path, device)
    
    # Generate samples
    samples = generate_samples(model, num_samples=10, device=device)
    
    # Save samples
    output_path = 'generated_samples.npy'
    np.save(output_path, samples)
    print(f"\n✓ Saved generated samples to: {output_path}")
    
    print("\n" + "="*60)
    print("Model loaded successfully!")
    print("="*60)
    print("\nTo use in your code:")
    print("```python")
    print("from load_model import load_vae_model, generate_samples")
    print()
    print("# Load model")
    print(f"model, config, epoch = load_vae_model('{model_path}')")
    print()
    print("# Generate new conformations")
    print("samples = generate_samples(model, num_samples=100)")
    print()
    print("# Encode existing molecule")
    print("from torch_geometric.data import Data")
    print("data = Data(z=atomic_numbers, pos=coords)")
    print("mu, log_var = model.encoder(data)")
    print("```")
    print("="*60)


if __name__ == "__main__":
    main()

