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
        
        # 1. Compute un-aggregated messages for each edge
        # We manually call the message function
        edge_messages = self.message(x[row], x[col], pos[row], pos[col])
        
        # 2. Update coordinates (Equivariant Step)
        # Use per-edge messages to get per-edge weights
        coord_weights = self.coord_mlp(edge_messages)
        rel_pos = pos[row] - pos[col]
        
        # Aggregate coordinate updates using scatter_add
        coord_update = torch.zeros_like(pos)
        weighted_rel_pos = coord_weights * rel_pos
        coord_update.scatter_add_(0, row.unsqueeze(1).expand_as(rel_pos), weighted_rel_pos)
        
        # Add a normalization factor for stability (optional but recommended)
        num_neighbors = torch.bincount(row, minlength=pos.size(0)).float().unsqueeze(1)
        pos_new = pos + coord_update / (num_neighbors + 1e-6)

        
        aggregated_messages = scatter_add(edge_messages, row, dim=0, dim_size=x.size(0))
        
        # Update node features using original features and aggregated messages
        node_mlp_input = torch.cat([x, aggregated_messages], dim=1)
        node_update = self.node_mlp(node_mlp_input)
        
        # Add residual connection and layer normalization
        x_new = self.node_norm(x + node_update)
        
        return x_new, pos_new
    
    def message(self, x_i, x_j, pos_i, pos_j):
        """
        Computes messages from node j to node i for each edge.
        This is now called manually in the forward pass.
        """
        # Compute squared distance (an invariant feature)
        dist_sq = torch.sum((pos_i - pos_j) ** 2, dim=1, keepdim=True)
        
        # Create edge features
        edge_features = torch.cat([x_i, x_j, dist_sq], dim=1)
        
        # Compute messages using the edge MLP
        return self.edge_mlp(edge_features)