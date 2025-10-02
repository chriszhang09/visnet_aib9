import torch
import torch.nn as nn
import torch.nn.functional as F
from .visnet_vae_encoder_mse import ViSNetEncoderMSE
from .vae_decoder_mse import PyGEGNNDecoderMSE
class MolecularVAEMSE(nn.Module):
    """
    MSE-specific Molecular VAE with proper initialization for stable training.
    """
    def __init__(self, latent_dim, num_atoms, atom_feature_dim, 
                 visnet_hidden_channels=128, decoder_hidden_dim=128, 
                 decoder_num_layers=6, visnet_kwargs=None):
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
        self.decoder = PyGEGNNDecoderMSE(
            latent_dim=latent_dim, 
            num_atoms=num_atoms, 
            atom_feature_dim=atom_feature_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers
        )

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
    
    def sample_prior(self, batch_size, device):
        """Sample from isotropic Gaussian prior N(0, I)"""
        return torch.randn(batch_size, self.decoder.latent_dim, device=device)

    def forward(self, data):
        # Encode
        mu, log_var = self.encoder(data)
        
        z = self.reparameterize(mu, log_var)
        
        # Decode
        # We need the one-hot atom types for the decoder
        # This assumes data.z contains integer atom types
        atom_types_one_hot = F.one_hot(data.z.long(), num_classes=self.decoder.atom_feature_dim).float()
        
        # PyG decoder needs edge_index and batch information
        reconstructed_pos = self.decoder(z, atom_types_one_hot, data.edge_index, data.batch)
        
        return reconstructed_pos, mu, log_var

def simple_mse_loss_mse(pred_coords, target_coords):
    """
    Simple MSE loss on coordinates with E(3) invariance via centering.
    Much faster and more stable than pairwise distance loss.
    All computations on GPU for maximum speed.
    """
    # Ensure both tensors are on the same device and float32
    pred_coords = pred_coords.float()
    target_coords = target_coords.float()
    
    # Ensure both are on the same device (GPU if available)
    if pred_coords.device != target_coords.device:
        target_coords = target_coords.to(pred_coords.device)
    
    # Center both coordinate sets to make them translation invariant
    # All operations stay on GPU
    pred_centered = pred_coords - pred_coords.mean(dim=0, keepdim=True)
    target_centered = target_coords - target_coords.mean(dim=0, keepdim=True)
    
    # Simple MSE loss on centered coordinates (GPU computation)
    mse_loss = F.mse_loss(pred_centered, target_centered)
    
    # Scale down to reasonable range (coordinates are typically 0.1-2.0 nm)
    return mse_loss * 0.1


def vae_loss_function_mse(reconstructed_pos, original_pos, mu, log_var):
    """
    Loss function for MSE training with non-centered isotropic Gaussian VAE.
    Uses pairwise distance loss for E(3) invariance.
    """
    # Use simple MSE loss for E(3) invariance
    recon_loss = simple_mse_loss_mse(reconstructed_pos, original_pos)
    
    # KL divergence for non-centered isotropic Gaussian: KL(N(μ, σ²I) || N(0, I))
    kl_div = 0.5 * torch.sum(mu.pow(2) + torch.exp(log_var) - log_var - 1)
    
    return recon_loss + kl_div
