"""
DCGAN architecture for Fashion-MNIST (28x28 grayscale).

Generator upsamples a latent vector z to a 28x28 image.
Discriminator classifies 28x28 images as real or fake.

Weight initialization follows the DCGAN paper convention:
all weights ~ N(0, 0.02), all biases = 0.
"""

import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """DCGAN weight initialization: Normal(0, 0.02) for conv/linear/bn weights."""
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """
    Generator for Fashion-MNIST.

    z (z_dim,) -> Linear -> Reshape (256,7,7)
               -> ConvT(256->128, 4x4, s2, p1) + BN + ReLU   -> (128,14,14)
               -> ConvT(128->64,  4x4, s2, p1) + BN + ReLU   -> (64,28,28)
               -> ConvT(64->1,    3x3, s1, p1) + Tanh         -> (1,28,28)
    """

    def __init__(self, z_dim: int = 64) -> None:
        super().__init__()
        self.z_dim = z_dim

        self.project = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            # (256, 7, 7) -> (128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # (128, 14, 14) -> (64, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (64, 28, 28) -> (1, 28, 28)
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)          # (B, 256*7*7)
        x = x.view(-1, 256, 7, 7)    # (B, 256, 7, 7)
        return self.conv_blocks(x)    # (B, 1, 28, 28)


class Discriminator(nn.Module):
    """
    Discriminator for Fashion-MNIST.

    (1, 28, 28) -> Conv(1->64,   4x4, s2, p1) + LeakyReLU + Dropout2d  -> (64,14,14)
                -> Conv(64->128, 4x4, s2, p1) + BN + LeakyReLU + Drop  -> (128,7,7)
                -> Flatten -> Linear(128*7*7, 1) + Sigmoid
    """

    def __init__(self, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate

        self.conv_blocks = nn.Sequential(
            # (1, 28, 28) -> (64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            # (64, 14, 14) -> (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_blocks(x)   # (B, 128, 7, 7)
        return self.classifier(features)  # (B, 1)


class DiscriminatorSpectralNorm(nn.Module):
    """
    Discriminator with SpectralNorm instead of BatchNorm.
    Used for Task B in the modularity experiment.
    Same architecture as Discriminator but with spectral normalization
    on conv and linear layers (no BatchNorm needed).
    """

    def __init__(self, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate

        self.conv_blocks = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(128 * 7 * 7, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_blocks(x)
        return self.classifier(features)
