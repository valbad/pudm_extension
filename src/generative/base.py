"""Abstract base class for generative strategies (DDPM, Flow Matching, etc.)."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class GenerativeStrategy(ABC):
    """Base interface for point cloud generative upsampling strategies.

    Each strategy defines:
    - How to compute diffusion/flow hyperparameters
    - How to compute training loss
    - How to sample (generate) dense point clouds from sparse conditions

    The neural backbone (PointNet2CloudCondition) is shared across strategies.
    """

    @abstractmethod
    def compute_hyperparams(self, **config) -> dict:
        """Compute strategy-specific hyperparameters from config.
        Returns a dict that will be passed to training_loss() and sampling().
        """
        pass

    @abstractmethod
    def training_loss(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        x0: torch.Tensor,
        hyperparams: dict,
        label: torch.Tensor = None,
        condition: torch.Tensor = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Compute the training loss for one batch.

        Args:
            net: The denoising/velocity network.
            loss_fn: Loss function (e.g., MSELoss).
            x0: Ground truth dense point cloud, (B, N, 3).
            hyperparams: Strategy-specific hyperparameters dict.
            label: Class labels, (B,).
            condition: Sparse input point cloud, (B, M, 3).
            alpha: Weight for condition reconstruction loss.

        Returns:
            Scalar loss tensor.
        """
        pass

    @abstractmethod
    def sample(
        self,
        net: nn.Module,
        size: tuple,
        hyperparams: dict,
        condition: torch.Tensor = None,
        label: torch.Tensor = None,
        R: int = 4,
        gamma: float = 0.5,
        **kwargs,
    ) -> tuple:
        """Generate dense point clouds.

        Args:
            net: The denoising/velocity network.
            size: Output shape (B, N, 3).
            hyperparams: Strategy-specific hyperparameters dict.
            condition: Sparse input, (B, M, 3).
            label: Class labels, (B,).
            R: Upsampling rate.
            gamma: Condition interpolation weight.

        Returns:
            Tuple of (generated_data, condition_prediction, initial_noise).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this strategy."""
        pass
