import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Batch

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

class PyGEGNNLayer(MessagePassing):
    """PyTorch Geometric EGNN layer with proper message passing."""
    
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
        Args:
            x (Tensor): Node features [num_nodes, hidden_dim]
            pos (Tensor): Node positions [num_nodes, 3]
            edge_index (LongTensor): Edge connectivity [2, num_edges]
        """
        row, col = edge_index
        
        # Safety checks to prevent CUDA assertions
        num_nodes = x.size(0)
        
        # Check for empty edges
        if edge_index.size(1) == 0:
            print("Warning: No edges provided, returning unchanged")
            return x, pos
            
        # Check for negative indices
        if (row < 0).any() or (col < 0).any():
            print("Error: Negative edge indices detected")
            return x, pos
            
        if row.max() >= num_nodes or col.max() >= num_nodes:
            print(f"Error: Edge indices out of bounds. Max node: {num_nodes-1}, Max edge indices: {row.max()}, {col.max()}")
            return x, pos
        
        if torch.isnan(pos).any() or torch.isinf(pos).any():
            print("Error: NaN/Inf detected in positions")
            return x, pos
            
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Error: NaN/Inf detected in node features")
            return x, pos
        
        # 1. Compute un-aggregated messages for each edge
        # We manually call the message function
        try:
            edge_messages = self.message(x[row], x[col], pos[row], pos[col])
        except RuntimeError as e:
            print(f"Error in message computation: {e}")
            return x, pos
        
        # 2. Update coordinates (Equivariant Step)
        # Use per-edge messages to get per-edge weights
        # Ensure edge_messages are in float32 to avoid precision issues
        edge_messages_f32 = edge_messages.float()
        coord_weights = self.coord_mlp(edge_messages_f32)
        rel_pos = pos[row] - pos[col]
        
        # Check for NaN/Inf values that could cause cuBLAS errors
        if torch.isnan(edge_messages_f32).any() or torch.isinf(edge_messages_f32).any():
            print("Warning: NaN/Inf detected in edge_messages, skipping coordinate update")
            pos_new = pos
        else:
            # Aggregate coordinate updates using scatter_add
            coord_update = torch.zeros_like(pos)
            weighted_rel_pos = coord_weights * rel_pos
            
            # Use safer scatter operation
            for dim in range(3):
                coord_update[:, dim].scatter_add_(0, row, weighted_rel_pos[:, dim])
            
            # Add a normalization factor for stability
            num_neighbors = torch.bincount(row, minlength=pos.size(0)).float().unsqueeze(1)
            pos_new = pos + coord_update / (num_neighbors + 1e-6)
        
        # 3. Update node features (Invariant Step)
        # Manually aggregate edge messages to get per-node messages
        aggregated_messages = scatter_add(edge_messages_f32, row, dim=0, dim_size=x.size(0))
        
        # Check for numerical issues in aggregated messages
        if torch.isnan(aggregated_messages).any() or torch.isinf(aggregated_messages).any():
            print("Warning: NaN/Inf detected in aggregated_messages, using identity update")
            x_new = x
        else:
            # Update node features using original features and aggregated messages
            node_mlp_input = torch.cat([x.float(), aggregated_messages], dim=1)
            node_update = self.node_mlp(node_mlp_input)
            
            # Add residual connection and layer normalization
            x_new = self.node_norm(x.float() + node_update)
        
        return x_new, pos_new
    
    def message(self, x_i, x_j, pos_i, pos_j):
        """
        Computes messages from node j to node i for each edge.
        This is now called manually in the forward pass.
        """
        # Safety checks
        if torch.isnan(pos_i).any() or torch.isnan(pos_j).any():
            print("Warning: NaN in positions during message computation")
            # Return zero messages
            return torch.zeros(x_i.size(0), x_i.size(1), device=x_i.device, dtype=torch.float32)
        
        # Compute squared distance (an invariant feature) with safety
        try:
            rel_pos = pos_i - pos_j
            dist_sq = torch.sum(rel_pos ** 2, dim=1, keepdim=True)
            
            # Replace any NaN/Inf with safe values
            dist_sq = torch.where(torch.isnan(dist_sq) | torch.isinf(dist_sq), 
                                torch.ones_like(dist_sq), dist_sq)
            
            # Clamp distance to prevent numerical issues
            dist_sq = torch.clamp(dist_sq, min=1e-6, max=1e6)
            
        except RuntimeError:
            print("Warning: Error computing distances, using default")
            dist_sq = torch.ones(x_i.size(0), 1, device=x_i.device, dtype=torch.float32)
        
        # Create edge features, ensure float32 precision
        edge_features = torch.cat([x_i.float(), x_j.float(), dist_sq.float()], dim=1)
        
        # Compute messages using the edge MLP
        try:
            messages = self.edge_mlp(edge_features)
            # Clamp messages to prevent explosion
            messages = torch.clamp(messages, min=-10.0, max=10.0)
        except RuntimeError:
            print("Warning: Error in edge MLP, returning zero messages")
            messages = torch.zeros(x_i.size(0), x_i.size(1), device=x_i.device, dtype=torch.float32)
        
        return messages


class PyGEGNNDecoder(nn.Module):
    """
    PyTorch Geometric-based EGNN Decoder that properly handles batching.
    """
    
    def __init__(self, latent_dim, num_atoms, atom_feature_dim, hidden_dim=128, 
                 num_layers=6, activation='silu'):
        """
        Args:
            latent_dim (int): Dimension of latent space
            num_atoms (int): Number of atoms in molecule
            atom_feature_dim (int): Dimension of atom type one-hot encoding
            hidden_dim (int): Hidden dimension for EGNN layers
            num_layers (int): Number of EGNN layers
            activation (str): Activation function ('silu' or 'relu')
        """
        super().__init__()
        self.num_atoms = num_atoms
        self.latent_dim = latent_dim
        self.atom_feature_dim = atom_feature_dim
        self.hidden_dim = hidden_dim
        
        # Latent injection network
        act_fn = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.latent_injector = nn.Sequential(
            nn.Linear(latent_dim + atom_feature_dim, hidden_dim * 2),
            act_fn,
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Stack of EGNN layers
        self.layers = nn.ModuleList([
            PyGEGNNLayer(hidden_dim, activation=activation) 
            for _ in range(num_layers)
        ])
        
    def forward(self, z, atom_types, edge_index, batch):
        """
        Args:
            z (Tensor): Latent vectors, shape (batch_size, latent_dim)
            atom_types (Tensor): One-hot encoded atom types, shape (batch_size * num_atoms, atom_feature_dim)
            edge_index (Tensor): Edge connectivity, shape (2, num_edges)
            batch (Tensor): Batch assignment for each node, shape (batch_size * num_atoms,)
        
        Returns:
            Tensor: Reconstructed positions, shape (batch_size * num_atoms, 3)
        """
        batch_size = z.size(0)
        
        # 1. Expand latent codes to all atoms in the batch
        z_expanded = z.repeat_interleave(self.num_atoms, dim=0)
        
        # 2. Initialize node features by combining atom types and latent codes
        h = torch.cat([atom_types, z_expanded], dim=1)
        h = self.latent_injector(h)
        
        # 3. Initialize coordinates with small random noise
        pos = torch.randn(batch_size * self.num_atoms, 3, device=z.device) * 0.1
        
        # 4. Iteratively refine structure through EGNN layers
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        
        return pos


# For backward compatibility, create an alias
EGNNDecoder = PyGEGNNDecoder