import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Module, ReLU, Sequential, init
import einops
import math
import torch.nn.functional as F
from enum import Enum


class TiedBiasPosition(str, Enum):
    """Tied Bias Position."""

    PRE_ENCODER = "pre_encoder"
    POST_DECODER = "post_decoder"

class TiedBias(nn.Module):

    def __init__(self, bias: Float[Tensor, "input_activations"], position: TiedBiasPosition) -> None:
        super().__init__()

        self.bias = bias
        self.position = position

    def forward(self, x: Float[Tensor, "*batch input_activations"]) -> Float[Tensor, "*batch input_activations"]:
        if self.position == TiedBiasPosition.PRE_ENCODER:
            return x - self.bias
        elif self.position == TiedBiasPosition.POST_DECODER:
            return x + self.bias
        else:
            raise ValueError(f"Invalid tied bias position: {self.position}")

class ConstrainedUnitNormLinear(nn.Module):

    DIMENSION_CONSTRAIN_UNIT_NORM: int = -1

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # Register backward hook to remove any gradient information parallel to the dictionary
        # vectors (columns of the weight matrix) before applying the gradient step.
        self.weight.register_hook(self._weight_backward_hook)

    def reset_parameters(self) -> None:
        self.weight: Float[Tensor, "out_features in_features"] = init.normal_(
            self.weight, mean=0, std=1
        )

        # Scale so that each column has unit norm
        with torch.no_grad():
            torch.nn.functional.normalize(
                self.weight, dim=self.DIMENSION_CONSTRAIN_UNIT_NORM, out=self.weight
            )

        # Initialise the bias
        # This is the standard approach used in `torch.nn.Linear.reset_parameters`
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _weight_backward_hook(self, grad):
        dot_product: Float[Tensor, " out_features"] = einops.einsum(
            grad,
            self.weight,
            "out_features in_features, out_features in_features -> out_features",
        )

        normalized_weight: Float[Tensor, "out_features in_features"] = (
            self.weight
            / torch.norm(
                self.weight, dim=self.DIMENSION_CONSTRAIN_UNIT_NORM, keepdim=True
            )
        )

        projection = einops.einsum(
            dot_product,
            normalized_weight,
            "out_features, out_features in_features -> out_features in_features",
        )

        return grad - projection

    def constrain_weights_unit_norm(self) -> None:
        with torch.no_grad():
            torch.nn.functional.normalize(
                self.weight, dim=self.DIMENSION_CONSTRAIN_UNIT_NORM, out=self.weight
            )

    def forward(self, x):
        self.constrain_weights_unit_norm()
        return torch.nn.functional.linear(x, self.weight, self.bias)

    
class SparseAutoencoder(nn.Module):

    def __init__(self, n_input_features: int, n_learned_features: int, l1_coefficient: float):
        super(SparseAutoencoder, self).__init__()

        self.n_input_features = n_input_features
        self.n_learned_features = n_learned_features
        self.l1_coefficient = l1_coefficient

        # Tied bias
        self.geometric_median_dataset = torch.zeros(n_input_features)
        self.geometric_median_dataset.requires_grad = False
        self.tied_bias = Parameter(torch.empty(n_input_features))
        self.initialise_tied_parameters()
        

        # Encoder and decoder
        self.encoder = nn.Sequential(
            TiedBias(self.tied_bias, TiedBiasPosition.PRE_ENCODER),
            ConstrainedUnitNormLinear(n_input_features, n_learned_features),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            ConstrainedUnitNormLinear(n_learned_features, n_input_features, bias=False),
            TiedBias(self.tied_bias, TiedBiasPosition.POST_DECODER),
        )

    def forward(self, x):
        learned_activations = self.encoder(x)
        reconstructed_activations = self.decoder(learned_activations)
        loss, recon_loss = self.loss_fn(reconstructed_activations, learned_activations, x)
        return reconstructed_activations, loss, recon_loss
    
    def loss_fn(self, decoded_activations, learned_activations, resid_streams):

        # RECONSTRUCTION LOSS
        per_item_mse_loss = self.per_item_mse_loss_with_target_norm(decoded_activations, resid_streams)
        recon_loss = per_item_mse_loss.sum(dim=-1).mean()

        # SPARSITY LOSS
        sparsity_loss = learned_activations.norm(p=1, dim=-1).mean()

        # combine
        return recon_loss + (self.l1_coefficient * sparsity_loss), recon_loss

    def initialise_tied_parameters(self) -> None:
        self.tied_bias.data = self.geometric_median_dataset.clone() 

    def per_item_mse_loss_with_target_norm(self, preds, target):
        return torch.nn.functional.mse_loss(preds, target, reduction='none')