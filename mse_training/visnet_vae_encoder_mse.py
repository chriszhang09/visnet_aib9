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
        
        self.representation_model = ViSNetBlock(**visnet_kwargs)
        
        actual_hidden_channels = visnet_kwargs.get('hidden_channels', visnet_hidden_channels)
        
        self.output_model = EquivariantEncoder(actual_hidden_channels, output_channels=latent_dim + 1)
        
                # In your Encoder's __init__ method
        with torch.no_grad():
            # Target the final linear layer in the EquivariantEncoder
            final_linear_layer = self.output_model.output_network[-1].update_net[-1]

            # The log_var is the LAST output channel. Its index is `latent_dim`.
            log_var_index = self.latent_dim

            # Set its bias to a negative value to start variance near zero.
            final_linear_layer.bias.data[log_var_index] = -3.0

            # Scale down its weights to make the initial output more stable.
            final_linear_layer.weight.data[log_var_index] *= 0.1

            print(f"Successfully initialized bias for log_var output to {final_linear_layer.bias.data[log_var_index]:.2f}")
            # --- END OF FIX ---
    
    def forward(self, data):
        # Get atom-level features from ViSNetBlock
        x, v = self.representation_model(data)
        
        x = self.output_model.pre_reduce(x, v, data.z, data.pos, data.batch)
        
        global_features = global_add_pool(x, data.batch)
        
        mu = global_features[:, :self.latent_dim]
        # Return log_var as shape [batch, 1] to ensure stable broadcasting
        log_var = global_features[:, self.latent_dim:self.latent_dim + 1]
        
        # Add initialization bias to log_var
        log_var = log_var + self.log_var_bias

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
