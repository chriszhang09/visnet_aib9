from tkinter import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from .visnet_vae_encoder_mse import ViSNetEncoderMSE
from .vae_decoder_new import EquivariantDecoder
class MolecularVAEMSE(nn.Module):
    """
    MSE-specific Molecular VAE with proper initialization for stable training.
    """
    def __init__(self, latent_dim, num_atoms, atom_feature_dim, 
                 visnet_hidden_channels=128, decoder_hidden_dim=128, 
                 decoder_num_layers=6, visnet_kwargs=None, cutoff: float = 3.0):
        """
        Args:
            latent_dim (int): Dimension of the latent space
            num_atoms (int): Number of atoms in each molecule
            atom_feature_dim (int): Dimension for one-hot encoding of atom types (max_atomic_number + 1)
            visnet_hidden_channels (int): Hidden dimension for ViSNet encoder
            decoder_hidden_dim (int): Hidden dimension for EGNN decoder
            decoder_num_layers (int): Number of EGNN layers in decoder
            visnet_kwargs (dict): Additional parameters for ViSNetBlock
        """
        super().__init__()
        if visnet_kwargs is None:
            visnet_kwargs = {}
        
        self.encoder = ViSNetEncoderMSE(
            latent_dim=latent_dim,
            visnet_hidden_channels=visnet_hidden_channels,
            **visnet_kwargs
        )
        self.decoder = EquivariantDecoder(
            scalar_lat_dim=atom_feature_dim,
            vector_lat_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            k_neighbors=num_atoms-1
        )
        self.latent_dim = latent_dim
        self.atom_feature_dim = atom_feature_dim
        self.num_atoms = num_atoms

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for non-centered isotropic Gaussian.
        z ~ N(μ, σ²I) where σ² is a scalar per-sample.
        Handles log_var shapes [batch], [batch, 1], or scalar.
        Uses soft bounds to prevent numerical issues while allowing learning.
        """
        # Ensure log_var can broadcast to mu: make it [batch, 1]
        if log_var.dim() == 1:
            log_var = log_var.unsqueeze(1)
        
        # Apply soft bounds only during sampling (not for KL computation)
        log_var_clamped = torch.clamp(log_var, min=-5, max=2)
        std = torch.exp(0.5 * log_var_clamped)
        eps = torch.randn_like(mu)
        return mu + eps * std
    
    def sample_prior(self, device):
        """Sample from 3D isotropic Gaussian prior N(0, I) with shape (num_atoms, H, 3)"""
        return torch.randn(self.num_atoms, self.latent_dim, 3, device=device)

    def forward(self, data):
        # Encode
        v = self.encoder(data)
        log_var = v[:, :, self.latent_dim:]
        mu = v[:, :, :self.latent_dim]

        latent_vector = self.reparameterize(mu, log_var)
        latent_vector = latent_vector.transpose(1, 2)
        
        # Check for NaN in latent vector
        if torch.isnan(latent_vector).any():
            print("Warning: NaN detected in latent_vector")
            latent_vector = torch.nan_to_num(latent_vector, nan=0.0)
      
        # We need the one-hot atom types for the decoder
        # This assumes data.z contains integer atom types
        atom_types_one_hot = F.one_hot(data.z.long(), num_classes=self.atom_feature_dim).float()

        # Force cutoff-based edges inside the decoder (edge_index=None)
        # Use smaller random positions for stability
        pos_rand = torch.randn(data.num_nodes, 3, device=data.pos.device) * 0.1
       
        reconstructed_pos = self.decoder(z_v = latent_vector, z_s = atom_types_one_hot, pos_rand = pos_rand)
        
        # Check for NaN in reconstruction
        if torch.isnan(reconstructed_pos).any():
            print("Warning: NaN detected in reconstructed_pos")
            reconstructed_pos = torch.nan_to_num(reconstructed_pos, nan=0.0)
        
        return reconstructed_pos, mu, log_var

def pairwise_distance_loss_mse(pred_coords, target_coords):
    """
    E(3) invariant loss using ALL pairwise distances between atoms.
    Much more informative than bond-only loss, still rotationally invariant.
    """
    # Convert to float32
    pred_coords = pred_coords.float()
    target_coords = target_coords.float()
    
    # Handle both batch format [batch_size, num_atoms, 3] and PyG format [total_atoms, 3]
    if pred_coords.dim() == 3:
        # Batch format - flatten to PyG format for processing
        batch_size, num_atoms, _ = pred_coords.shape
        pred_flat = pred_coords.view(-1, 3)  # [total_atoms, 3]
        target_flat = target_coords.view(-1, 3)
        
        # Process each molecule in the batch separately
        total_loss = 0
        for i in range(batch_size):
            start_idx = i * num_atoms
            end_idx = (i + 1) * num_atoms
            
            pred_mol = pred_flat[start_idx:end_idx]  # [num_atoms, 3]
            target_mol = target_flat[start_idx:end_idx]  # [num_atoms, 3]
            
            # Compute pairwise distances for this molecule
            pred_dists = torch.cdist(pred_mol, pred_mol)  # [num_atoms, num_atoms]
            target_dists = torch.cdist(target_mol, target_mol)  # [num_atoms, num_atoms]
            
            # Only use upper triangular part (avoid diagonal and duplicates)
            mask = torch.triu(torch.ones_like(pred_dists), diagonal=1).bool()
            pred_dists_upper = pred_dists[mask]  # [num_pairs]
            target_dists_upper = target_dists[mask]  # [num_pairs]
            
            # MSE on pairwise distances
            mol_loss = F.mse_loss(pred_dists_upper, target_dists_upper)
            total_loss += mol_loss
        
        return total_loss / batch_size
    
    else:
        # PyG format - single molecule or already flattened batch
        # Assume this is a single molecule for now
        pred_dists = torch.cdist(pred_coords, pred_coords)  # [num_atoms, num_atoms]
        target_dists = torch.cdist(target_coords, target_coords)  # [num_atoms, num_atoms]
        
        # Only use upper triangular part (avoid diagonal and duplicates)
        mask = torch.triu(torch.ones_like(pred_dists), diagonal=1).bool()
        pred_dists_upper = pred_dists[mask]  # [num_pairs]
        target_dists_upper = target_dists[mask]  # [num_pairs]
        
        # MSE on pairwise distances
        loss = F.mse_loss(pred_dists_upper, target_dists_upper)
        
        return loss


def vae_loss_function_mse(reconstructed_pos, original_pos, mu, log_var):
    """
    Loss function for MSE training with non-centered isotropic Gaussian VAE.
    Uses pairwise distance loss for E(3) invariance.
    """
    # Use pairwise distance loss for E(3) invariance
    recon_loss = pairwise_distance_loss_mse(reconstructed_pos, original_pos)
    
    # KL divergence for non-centered isotropic Gaussian: KL(N(μ, σ²I) || N(0, I))
    kl_div = 0.5 * torch.sum(mu.pow(2) + torch.exp(log_var) - log_var - 1)
    
    return recon_loss + kl_div
