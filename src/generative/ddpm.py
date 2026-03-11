"""DDPM generative strategy for point cloud upsampling.

Implements the original PUDM diffusion process with both DDPM and DDIM sampling.
"""
import torch
import torch.nn as nn
import numpy as np

from src.generative.base import GenerativeStrategy
from src.utils.pc_utils import midpoint_interpolate, get_interpolate
from src.utils.misc import std_normal


class DDPMStrategy(GenerativeStrategy):
    """DDPM-based generative strategy (original PUDM baseline)."""

    @property
    def name(self) -> str:
        return "DDPM"

    def compute_hyperparams(self, T=1000, beta_0=0.0001, beta_T=0.02, **kwargs) -> dict:
        """Compute DDPM diffusion schedule hyperparameters.

        Returns dict with keys: T, Beta, Alpha, Alpha_bar, Sigma.
        """
        Beta = torch.linspace(beta_0, beta_T, T)
        Alpha = 1 - Beta
        Alpha_bar = Alpha.clone()
        Beta_tilde = Beta.clone()
        for t in range(1, T):
            Alpha_bar[t] *= Alpha_bar[t - 1]
            Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
        Sigma = torch.sqrt(Beta_tilde)

        return {
            "T": T,
            "Beta": Beta,
            "Alpha": Alpha,
            "Alpha_bar": Alpha_bar,
            "Sigma": Sigma,
        }

    def training_loss(
        self,
        net,
        loss_fn,
        x0,
        hyperparams,
        label=None,
        condition=None,
        alpha=1.0,
    ):
        """DDPM training loss: predict noise from noisy point cloud."""
        T = hyperparams["T"]
        Alpha_bar = hyperparams["Alpha_bar"]
        B, N, D = x0.shape
        device = x0.device

        # Sample random timestep for each batch element
        diffusion_steps = torch.randint(T, size=(B, 1, 1), device=device)
        z = std_normal(x0.shape, device=device)

        # Forward diffusion: q(x_t | x_0)
        xt = (
            torch.sqrt(Alpha_bar[diffusion_steps]) * x0
            + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
        )

        # Midpoint interpolation of condition
        i = midpoint_interpolate(condition.permute(0, 2, 1)).permute(0, 2, 1)

        # Concatenate noisy data with interpolated condition
        xt = torch.cat([xt, i], dim=-1)

        # Predict noise
        epsilon_theta = net(
            xt, condition, ts=diffusion_steps.view(B), label=label
        )

        if isinstance(epsilon_theta, tuple):
            noisy_pred, condition_pred = epsilon_theta
            mse_theta = loss_fn(noisy_pred, z)
            mse_psi = loss_fn(condition_pred, condition)
            loss = mse_theta + alpha * mse_psi
        else:
            loss = loss_fn(epsilon_theta, z)

        return loss

    def sample(
        self,
        net,
        size,
        hyperparams,
        condition=None,
        label=None,
        R=4,
        gamma=0.5,
        print_every_n_steps=100,
        **kwargs,
    ):
        """Full DDPM reverse sampling (T steps)."""
        print("---- DDPM Sampling ----")
        T = hyperparams["T"]
        Alpha = hyperparams["Alpha"]
        Alpha_bar = hyperparams["Alpha_bar"]
        Sigma = hyperparams["Sigma"]
        device = condition.device if condition is not None else 'cuda'

        print('---- begin sampling, total steps: %s ----' % T)
        z = std_normal(size, device=device)
        x = z

        if label is not None and isinstance(label, int):
            label = torch.ones(size[0], dtype=torch.long, device=device) * label

        # Hierarchical midpoint interpolation
        i = get_interpolate(condition, R)

        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                if t % print_every_n_steps == 0:
                    print('reverse step: %d' % t, flush=True)

                diffusion_steps = (t * torch.ones((size[0],), device=device))
                x_ = torch.cat([x, i], dim=-1)
                results = net(
                    x_, condition, ts=diffusion_steps, label=label,
                    use_retained_condition_feature=True
                )
                if isinstance(results, tuple):
                    epsilon_theta, condition_pre = results
                else:
                    epsilon_theta = results

                # Reverse step: x_{t-1} from x_t
                item_1 = (
                    (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta)
                    / torch.sqrt(Alpha[t])
                )
                item_2 = Sigma[t] * std_normal(size, device=device) if t > 0 else 0.0
                x = gamma * (item_1 + item_2 + i)

        if condition is not None:
            net.reset_cond_features()
        return x, condition_pre, z

    def sample_ddim(
        self,
        net,
        size,
        hyperparams,
        condition=None,
        label=None,
        R=4,
        gamma=0.5,
        step=30,
        print_every_n_steps=10,
        **kwargs,
    ):
        """Accelerated DDIM sampling (fewer steps)."""
        print("---- DDIM Sampling ----")
        T = hyperparams["T"]
        Alpha = hyperparams["Alpha"]
        Alpha_bar = hyperparams["Alpha_bar"]
        Sigma = hyperparams["Sigma"]
        device = condition.device if condition is not None else 'cuda'

        print('---- begin sampling, total steps: %s ----' % step)
        z = std_normal(size, device=device)
        x = z

        if label is not None and isinstance(label, int):
            label = torch.ones(size[0], dtype=torch.long, device=device) * label

        # Build DDIM timestep schedule
        ts1 = torch.linspace(T - 1, step // 2 + 1, step // 2, dtype=torch.int64)
        ts2 = torch.linspace(step // 2, 0, step // 2, dtype=torch.int64)
        ts = torch.cat([ts1, ts2], dim=0)
        steps = reversed(range(len(ts)))

        i = get_interpolate(condition, R)

        with torch.no_grad():
            for s, t in zip(steps, ts):
                if (s + 1) % print_every_n_steps == 0 or s == 0:
                    print('reverse step: %d' % (s + 1 if s > 0 else s), flush=True)

                diffusion_steps = (t * torch.ones((size[0],), device=device))
                x_ = torch.cat([x, i], dim=-1)
                results = net(
                    x_, condition, ts=diffusion_steps, label=label,
                    use_retained_condition_feature=True
                )
                if isinstance(results, tuple):
                    epsilon_theta, condition_pre = results
                else:
                    epsilon_theta = results

                # DDIM deterministic step
                x0 = (
                    (x - torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta)
                    / torch.sqrt(Alpha_bar[t])
                )
                if t > 0:
                    c_xs_1 = torch.sqrt(Alpha_bar[t - 1]) * x0
                    c_xs_2 = torch.sqrt(1 - Alpha_bar[t - 1]) * epsilon_theta
                    x = gamma * (c_xs_1 + c_xs_2 + i)
                else:
                    x = gamma * (x0 + i)

        if condition is not None:
            net.reset_cond_features()
        return x, condition_pre, z
