import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch_geometric.nn import global_add_pool
from visnet.models.visnet_block import ViSNetBlock
from visnet.models.output_modules import EquivariantEncoder, OutputModel
from torch_geometric.data import Data
from pytorch_lightning.utilities import rank_zero_warn
from torch.autograd import grad
from torch import Tensor
from torch_scatter import scatter
from visnet.models.output_modules import EquivariantEncoder

class ViSNetEncoderMSE(nn.Module):
    """
    MSE-specific ViSNet encoder with proper initialization to prevent variance explosion.
    This version includes initialization fixes for stable MSE training.
    """
    
    def __init__(self, latent_dim, visnet_hidden_channels=128, prior_model=None, mean=None, std=None, **visnet_kwargs):
        """
        Args:
            latent_dim (int): The dimension of the latent space.
            visnet_hidden_channels (int): The hidden dimension for the ViSNet model.
            **visnet_kwargs: Additional arguments to pass to the ViSNet model constructor.
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        # Store the representation model (ViSNetBlock) directly
        # We need access to atom-level features before they get pooled
        self.representation_model = ViSNetBlock(**visnet_kwargs)
        
        # Get the actual hidden_channels used by ViSNetBlock (may differ from default)
        actual_hidden_channels = visnet_kwargs.get('hidden_channels', visnet_hidden_channels)
        
        # EquivariantEncoder must use the same hidden_channels as ViSNetBlock
        # We need latent_dim for mu + 1 for log_var (scalar)
        self.output_model = EquivariantEncoder(actual_hidden_channels, output_channels=latent_dim + 1)
        
        # Initialize the log_var output to be small (prevent variance explosion)
        with torch.no_grad():
            # The EquivariantEncoder uses GatedEquivariantBlocks, initialize the final layer
            if hasattr(self.output_model, 'output_network') and len(self.output_model.output_network) > 0:
                final_block = self.output_model.output_network[-1]
                if hasattr(final_block, 'scalar_linear'):
                    # Initialize bias for log_var (last output channel) to be closer to 0
                    final_block.scalar_linear.bias.data[-1] = -1.0  # Start with reasonable variance
                    # Scale down weights for log_var output
                    final_block.scalar_linear.weight.data[-1] *= 0.1

    
    def forward(self, data):
        # Get atom-level features from ViSNetBlock
        x, v = self.representation_model(data)
        
        x = self.output_model.pre_reduce(x, v, data.z, data.pos, data.batch)
        
        global_features = global_add_pool(x, data.batch)
        
        mu = global_features[:, :self.latent_dim]
        # Return log_var as shape [batch, 1] to ensure stable broadcasting
        log_var = global_features[:, self.latent_dim:self.latent_dim + 1]
        
        # Clamp log_var to prevent variance explosion
        log_var = torch.clamp(log_var, min=-10, max=2)
        
        return mu, log_var


class ViSNet(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
    ):
        super(ViSNet, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                "Prior model was given but the output model does "
                "not allow prior models. Dropping the prior model."
            )

        self.reduce_op = reduce_op
        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, data: Data) -> Tuple[Tensor, Optional[Tensor]]:
        
        if self.derivative:
            data.pos.requires_grad_(True)

        x, v = self.representation_model(data)
        x = self.output_model.pre_reduce(x, v, data.z, data.pos, data.batch)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, data.z)

        out = scatter(x, data.batch, dim=0, reduce=self.reduce_op)
        out = self.output_model.post_reduce(out)
        
        out = out + self.mean

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [data.pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, -dy
        return out, None
