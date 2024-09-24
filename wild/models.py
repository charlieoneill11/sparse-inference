import torch
from torch import nn
from typing import Optional
import math

class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seed: int = 20240625
    ):
        """
        Initializes the Sparse Autoencoder.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Number of neurons in the hidden (encoded) layer.
            seed (int, optional): Random seed for weight initialization. Defaults to 20240625.
        """
        super(SparseAutoencoder, self).__init__()

        # Set the random seed for reproducibility
        torch.manual_seed(seed + 42)

        # Encoder: Linear layer followed by ReLU activation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        )

        # Decoder: Linear layer
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))

    def forward(self, X: torch.Tensor, norm_D: bool = True):
        """
        Forward pass of the autoencoder.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            norm_D (bool, optional): If True and `learn_decoder` is True, normalize decoder weights.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - S_: Encoded representations of shape (batch_size, hidden_dim).
                - X_: Reconstructed inputs of shape (batch_size, input_dim).
        """
        if norm_D:
            # Normalise decoder weights along the input_dim axis
            with torch.no_grad():
                self.decoder.weight /= torch.linalg.norm(self.decoder.weight, dim=1, keepdim=True) + 1e-8

        # Encode the input
        S_ = self.encoder(X)  # Shape: (batch_size, hidden_dim)

        # Decode the representation
        X_ = torch.matmul(S_, self.decoder.weight.T)  # Shape: (batch_size, input_dim)

        # Add bias if present
        if self.decoder.bias is not None:
            X_ += self.decoder.bias

        return S_, X_