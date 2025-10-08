import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Batch

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

class PyGEGNNLayerMSE(MessagePassing):
    """MSE-specific PyTorch Geometric EGNN layer with proper message passing."""
    
    def __init__(self, hidden_dim, activation='silu'):
        # NOTE: The aggregation is now done manually, so we can set aggr=None
        super().__init__(aggr=None) 
        act_fn = nn.SiLU() if activation == 'silu' else nn.ReLU()
        
        # Edge MLP for computing messages from edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), # Input: [h_i, h_j, ||x_i - x_j||^2]
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # Node MLP for updating node features from aggregated messages
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Input: [h_i, m_i]
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate MLP for computing coordinate weights from edge messages
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False) # Output a single scalar weight
        )
        
        # Layer normalization for stability
        self.node_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, pos, edge_index):
        """
        Forward pass of the EGNN layer.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            pos: Node positions [num_nodes, 3]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Updated node features and positions
        """
        # Safety checks for numerical stability
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf detected in input features x")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(pos).any() or torch.isinf(pos).any():
            print("Warning: NaN/Inf detected in input positions")
            pos = torch.nan_to_num(pos, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Move edge_index to CPU for safety checks to prevent CUDA errors
        edge_index_cpu = edge_index.cpu()
        row, col = edge_index_cpu[0], edge_index_cpu[1]
        
        # Check for invalid edge indices
        if (row < 0).any() or (col < 0).any():
            print("Warning: Negative edge indices detected")
            return x, pos
        
        if row.max() >= x.size(0) or col.max() >= x.size(0):
            print(f"Warning: Edge indices out of bounds. Max: {row.max()}, {col.max()}, Size: {x.size(0)}")
            return x, pos
        
        # Move back to device
        row, col = row.to(x.device), col.to(x.device)
        
        # Compute messages
        messages = self.message(x, pos, row, col)
        
        # Aggregate messages
        agg_messages = self.aggregate(messages, row, x.size(0))
        
        # Update node features
        x_new = self.node_mlp(torch.cat([x, agg_messages], dim=-1))
        x_new = self.node_norm(x_new)
        
        # Update coordinates
        pos_new = self.update_coords(x, pos, row, col, messages)
        
        return x_new, pos_new
    
    def message(self, x, pos, row, col):
        """Compute messages between connected nodes."""
        # Get features of source and target nodes
        x_i, x_j = x[row], x[col]
        pos_i, pos_j = pos[row], pos[col]
        
        # Compute relative positions and distances
        rel_pos = pos_i - pos_j
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)
        
        # Clamp distances for numerical stability
        dist_sq = torch.clamp(dist_sq, min=1e-6, max=1e6)
        
        # Check for NaN/Inf in distances
        if torch.isnan(dist_sq).any() or torch.isinf(dist_sq).any():
            print("Warning: NaN/Inf in distances, using fallback")
            dist_sq = torch.ones_like(dist_sq) * 1.0
        
        # Concatenate features for edge MLP
        edge_features = torch.cat([x_i, x_j, dist_sq], dim=-1)
        
        # Compute edge messages
        messages = self.edge_mlp(edge_features)
        
        # Check for NaN/Inf in messages
        if torch.isnan(messages).any() or torch.isinf(messages).any():
            print("Warning: NaN/Inf in messages, using fallback")
            messages = torch.zeros_like(messages)
        
        return messages
    
    def aggregate(self, messages, row, num_nodes):
        """Aggregate messages to target nodes."""
        # Use scatter_add to aggregate messages
        agg_messages = scatter_add(messages, row, dim=0, dim_size=num_nodes)
        
        # Check for NaN/Inf in aggregated messages
        if torch.isnan(agg_messages).any() or torch.isinf(agg_messages).any():
            print("Warning: NaN/Inf in aggregated messages, using fallback")
            agg_messages = torch.zeros_like(agg_messages)
        
        return agg_messages
    
    def update_coords(self, x, pos, row, col, messages):
        """Update coordinates based on edge messages."""
        # Compute coordinate weights from edge messages
        coord_weights = self.coord_mlp(messages)
        
        # Clamp weights for stability
        coord_weights = torch.clamp(coord_weights, min=-1.0, max=1.0)
        
        # Compute relative positions
        rel_pos = pos[row] - pos[col]
        
        # Compute weighted relative positions
        weighted_rel_pos = coord_weights * rel_pos
        
        # Initialize coordinate updates
        coord_update = torch.zeros_like(pos)
        
        # Use dimension-wise scatter to avoid shape issues
        if row.max() >= pos.size(0):
            print(f"Warning: Edge index out of bounds. Max row: {row.max()}, pos size: {pos.size(0)}")
            pos_new = pos  # Skip coordinate update
        else:
            for dim in range(3):
                coord_update[:, dim].scatter_add_(0, row, weighted_rel_pos[:, dim])
            
            # Add a normalization factor for stability (optional but recommended)
            num_neighbors = torch.bincount(row, minlength=pos.size(0)).float().unsqueeze(1)
            pos_new = pos + coord_update / (num_neighbors + 1e-6)
        
        return pos_new


class PyGEGNNDecoderMSE(nn.Module):
    """MSE-specific PyTorch Geometric EGNN decoder for molecular coordinates."""
    
    def __init__(self, latent_dim, hidden_dim=256, num_layers=2, num_atoms=58, atom_feature_dim=10, cutoff: float = 3.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_atoms = num_atoms
        self.atom_feature_dim = atom_feature_dim
        self.cutoff = float(cutoff)
        
        # Project latent vector to initial node features
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Project atom features to hidden dimension
        self.atom_proj = nn.Linear(atom_feature_dim, hidden_dim)
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            PyGEGNNLayerMSE(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final layer to output coordinates
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Initialize coordinates with small random noise for translation equivariance
        self.register_buffer('init_coords', torch.randn(num_atoms, 3) * 0.1)
        
    def forward(self, z, atom_types, edge_index, batch=None):
        """
        Forward pass of the decoder.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
            atom_types: One-hot encoded atom types [batch_size * num_atoms, atom_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch indices for PyG format
        
        Returns:
            Reconstructed coordinates [total_atoms, 3]
        """
        batch_size = z.size(0)
        total_atoms = batch_size * self.num_atoms
        
        # Project latent vector to node features
        node_features = self.latent_proj(z)  # [batch_size, hidden_dim]
        
        # Expand to all atoms in the batch
        node_features = node_features.unsqueeze(1).expand(-1, self.num_atoms, -1)  # [batch_size, num_atoms, hidden_dim]
        node_features = node_features.reshape(total_atoms, self.hidden_dim)  # [total_atoms, hidden_dim]
        
        # Add atom type information
        atom_features = self.atom_proj(atom_types)  # [total_atoms, hidden_dim]
        node_features = node_features + atom_features
        
        # Initialize coordinates with small random noise
        coords = self.init_coords.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_atoms, 3]
        coords = coords.reshape(total_atoms, 3)  # [total_atoms, 3]
        
        # Build cutoff-based edges ONCE per forward to avoid excessive memory use
        if edge_index is None:
            from torch_cluster import radius_graph
            # Cap the number of neighbors to bound memory; tune as needed
            edge_index = radius_graph(
                coords, r=self.cutoff, batch=batch, loop=False, max_num_neighbors=64
            )
        
        # Apply EGNN layers with the same connectivity
        for layer in self.egnn_layers:
            node_features, coords = layer(node_features, coords, edge_index)
        
        # Final coordinate prediction
        coord_deltas = self.coord_head(node_features)
        coords = coords + coord_deltas
        
        return coords


class EGNNDecoderMSE(nn.Module):
    """MSE-specific EGNN decoder (legacy interface for compatibility)."""
    
    def __init__(self, latent_dim, hidden_dim=256, num_layers=2, num_atoms=58, atom_feature_dim=10, cutoff: float = 3.0):
        super().__init__()
        self.pyg_decoder = PyGEGNNDecoderMSE(latent_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_atoms=num_atoms, atom_feature_dim=atom_feature_dim, cutoff=cutoff)
        # Expose commonly used attributes for external code compatibility
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_atoms = num_atoms
        self.atom_feature_dim = atom_feature_dim
        self.cutoff = float(cutoff)
    
    def forward(self, z, atom_types, edge_index, batch=None):
        return self.pyg_decoder(z, atom_types, edge_index, batch)
