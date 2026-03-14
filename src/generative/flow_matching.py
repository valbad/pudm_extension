"""Flow Matching generative strategy for point cloud upsampling.

Implements Conditional Flow Matching (CFM) as an alternative to DDPM.
Instead of learning to denoise, the network learns a velocity field v(x_t, t)
that transports samples from a noise distribution to the data distribution
along straight-line paths (optimal transport).

Key differences from DDPM:
- Forward process: x_t = (1 - t) * x_0 + t * z  (linear interpolation, t in [0, 1])
- Training: predict velocity v = z - x_0  (instead of noise epsilon)
- Sampling: ODE integration from t=1 (noise) to t=0 (data) using Euler steps
- No variance schedule needed — simpler hyperparameters
- Deterministic sampling by default (ODE, not SDE)
"""
import torch
import torch.nn as nn

from src.generative.base import GenerativeStrategy
from src.utils.pc_utils import midpoint_interpolate, get_interpolate
from src.utils.misc import std_normal


class FlowMatchingStrategy(GenerativeStrategy):
    """Conditional Flow Matching strategy for point cloud upsampling.

    Uses the same backbone network as DDPM but trains it to predict
    the velocity field instead of noise. The timestep embedding is
    rescaled to [0, T-1] for compatibility with the existing network.
    """

    @property
    def name(self) -> str:
        return "FlowMatching"

    def compute_hyperparams(self, T=1000, num_steps=100, **kwargs) -> dict:
        """Flow Matching has minimal hyperparameters.

        T is used only for rescaling the continuous time to match
        the network's timestep embedding range.
        num_steps is the default number of Euler integration steps at inference.
        """
        return {"T": T, "num_steps": num_steps}

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
        """Flow Matching training loss: predict velocity field.

        The flow is defined as: x_t = (1 - t) * x_0 + t * z
        The target velocity is: v = z - x_0
        The network predicts v_theta(x_t, t), trained with MSE against v.
        """
        T = hyperparams["T"]
        B, N, D = x0.shape
        device = x0.device

        # Sample continuous time t ~ U(0, 1)
        t = torch.rand(B, 1, 1, device=device)

        # Sample noise
        z = std_normal(x0.shape, device=device)

        # Interpolate: x_t = (1 - t) * x_0 + t * z
        xt = (1 - t) * x0 + t * z

        # Target velocity: v = z - x_0
        v_target = z - x0

        # Midpoint interpolation of condition (same as DDPM)
        i = midpoint_interpolate(condition.permute(0, 2, 1)).permute(0, 2, 1)

        # Concatenate with interpolated condition
        xt_input = torch.cat([xt, i], dim=-1)

        # Rescale t to [0, T-1] for the network's timestep embedding
        t_scaled = (t.squeeze(-1).squeeze(-1) * (T - 1))

        # Predict velocity
        v_theta = net(
            xt_input, condition, ts=t_scaled, label=label
        )

        if isinstance(v_theta, tuple):
            v_pred, condition_pred = v_theta
            mse_v = loss_fn(v_pred, v_target)
            mse_cond = loss_fn(condition_pred, condition)
            loss = mse_v + alpha * mse_cond
        else:
            loss = loss_fn(v_theta, v_target)

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
        num_steps=100,
        print_every_n_steps=10,
        **kwargs,
    ):
        """Generate via Euler integration of the learned velocity field.

        Integrates from t=1 (noise) to t=0 (data) in `num_steps` steps.
        """
        print("---- Flow Matching Sampling ----")
        T = hyperparams["T"]
        device = condition.device if condition is not None else 'cuda'

        print('---- begin sampling, total steps: %s ----' % num_steps)
        z = std_normal(size, device=device)
        x = z

        if label is not None and isinstance(label, int):
            label = torch.ones(size[0], dtype=torch.long, device=device) * label

        # Hierarchical midpoint interpolation (same as DDPM)
        i = get_interpolate(condition, R)

        dt = 1.0 / num_steps
        condition_pre = None

        with torch.no_grad():
            for step in range(num_steps):
                # Current time: goes from 1.0 to dt
                t_current = 1.0 - step * dt

                if step % print_every_n_steps == 0:
                    print('step: %d / %d  (t=%.4f)' % (step, num_steps, t_current), flush=True)

                # Rescale to network's timestep range
                t_scaled = t_current * (T - 1) * torch.ones((size[0],), device=device)

                x_ = torch.cat([x, i], dim=-1)
                results = net(
                    x_, condition, ts=t_scaled, label=label,
                    use_retained_condition_feature=True
                )
                if isinstance(results, tuple):
                    v_theta, condition_pre = results
                else:
                    v_theta = results

                # Euler step: x_{t-dt} = x_t - dt * v_theta
                # (velocity points from data to noise, so we subtract)
                x = x - dt * v_theta

                # Apply condition interpolation guidance
                x = gamma * (x + i)

        if condition is not None:
            net.reset_cond_features()
        return x, condition_pre, z
