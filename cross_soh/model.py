"""SOH network combining Zhang's two branches with Wang's dynamics residual.

Intra-cell input
    Feature sequence of the target window minus that cell's first retained
    cycle. This is the cycle-level version of Zhang et al. (2025) intra-cell
    differences.

Inter-cell input
    Target window minus a reference cell's aligned window. The reference cell
    is drawn from the training conditions only.

Shared head
    Both encoders use their own TimesNet stack (Wu et al., 2023). A single
    linear layer maps both embeddings to a scalar, as in Zhang's equation (8).

    ``y_intra = w h_theta(x - x_first)``
    ``y_inter = w h_phi(x - x_ref) + y_ref``
    ``y = mix_alpha * y_intra + (1 - mix_alpha) * y_inter``

Physics
    Wang et al. (2024) still supply the training losses. The dynamics network
    reads the current absolute feature vector, the predicted SOH, and the
    partial derivatives of that prediction. Those derivatives are taken with
    respect to the last cycle of the target window.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import grad

from cross_soh.times_matrix import TimesBlock


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class DynamicsF(nn.Module):
    """Wang's dynamics network: 35 -> 60 -> 60 -> 1, sine, dropout 0.2.

    Input layout is ``[x_t (17), u (1), u_x (16), u_t (1)]``.
    """

    def __init__(self, hidden_dim: int = 60, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(35, hidden_dim),
            Sin(),
            nn.Linear(hidden_dim, hidden_dim),
            Sin(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimesEncoder(nn.Module):
    """Embed a difference sequence and read it with TimesBlocks."""

    def __init__(
        self,
        n_features: int = 17,
        d_model: int = 32,
        n_layers: int = 2,
        k: int = 3,
        n_kernels: int = 3,
    ):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(
            [TimesBlock(d_model, k=k, n_kernels=n_kernels) for _ in range(n_layers)]
        )
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(sequence)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden[:, -1, :]


class CrossConditionSOH(nn.Module):
    """Fused intra-cell / inter-cell SOH predictor."""

    def __init__(
        self,
        n_features: int = 17,
        d_model: int = 32,
        n_layers: int = 2,
        k: int = 3,
        n_kernels: int = 3,
        mix_alpha: float = 0.5,
        dynamics_hidden: int = 60,
        dynamics_dropout: float = 0.2,
    ):
        super().__init__()
        if not 0.0 <= mix_alpha <= 1.0:
            raise ValueError("mix_alpha must be in [0, 1]")
        self.n_features = n_features
        self.mix_alpha = mix_alpha
        self.intra_encoder = TimesEncoder(n_features, d_model, n_layers, k, n_kernels)
        self.inter_encoder = TimesEncoder(n_features, d_model, n_layers, k, n_kernels)
        self.head = nn.Linear(d_model, 1)
        self.dynamics = DynamicsF(dynamics_hidden, dynamics_dropout)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        window: torch.Tensor,
        ref_window: torch.Tensor,
        y_ref: torch.Tensor,
        x0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict SOH for the last cycle of ``window``.

        Shapes: windows are ``[batch, length, 17]``, ``y_ref`` is ``[batch, 1]``,
        and ``x0`` is the target cell's first retained cycle ``[batch, 17]``.
        """
        intra = window - x0.unsqueeze(1)
        inter = window - ref_window
        h_intra = self.intra_encoder(intra)
        h_inter = self.inter_encoder(inter)
        y_intra = self.head(h_intra)
        y_delta = self.head(h_inter)
        y_inter = y_delta + y_ref
        y_hat = self.mix_alpha * y_intra + (1.0 - self.mix_alpha) * y_inter
        return {
            "pred": y_hat,
            "pred_intra": y_intra,
            "pred_delta": y_delta,
            "pred_inter": y_inter,
        }

    def forward_with_residual(
        self,
        window: torch.Tensor,
        ref_window: torch.Tensor,
        y_ref: torch.Tensor,
        x0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward plus Wang's residual ``u_t - F`` at the current cycle.

        ``window`` is differentiated. Period indices stay discrete, matching
        TimesNet, so the gradient flows through the tensor values and the
        amplitude weights, not through the choice of period.
        """
        if not window.requires_grad:
            window = window.detach().requires_grad_(True)
        outputs = self.forward(window, ref_window, y_ref, x0)
        derivatives = grad(outputs["pred"].sum(), window, create_graph=True)[0]
        current_gradient = derivatives[:, -1, :]
        current_features = window[:, -1, :]
        u_x = current_gradient[:, :-1]
        u_t = current_gradient[:, -1:]
        dynamics_input = torch.cat(
            [current_features, outputs["pred"], u_x, u_t],
            dim=-1,
        )
        outputs["residual"] = u_t - self.dynamics(dynamics_input)
        outputs["u_t"] = u_t
        return outputs


def compute_soh_losses(
    pred_1: torch.Tensor,
    pred_2: torch.Tensor,
    y_1: torch.Tensor,
    y_2: torch.Tensor,
    delta_1: torch.Tensor,
    delta_2: torch.Tensor,
    y_ref_1: torch.Tensor,
    y_ref_2: torch.Tensor,
    residual_1: torch.Tensor,
    residual_2: torch.Tensor,
    alpha: float,
    beta: float,
    lambda_delta: float,
) -> dict[str, torch.Tensor]:
    """Wang's pair losses plus Zhang's inter-cell difference loss.

    ``L_data`` and ``L_PDE`` are means. ``L_physics`` is a sum, as written in
    the PINN training step, so its scale grows with the batch. ``L_delta``
    trains the shared head to predict ``SOH - SOH_ref``.
    """
    mse = nn.functional.mse_loss
    loss_data = 0.5 * mse(pred_1, y_1) + 0.5 * mse(pred_2, y_2)
    loss_delta = 0.5 * mse(delta_1, y_1 - y_ref_1) + 0.5 * mse(delta_2, y_2 - y_ref_2)
    loss_physics = torch.relu((pred_2 - pred_1) * (y_1 - y_2)).sum()
    loss_pde = 0.5 * torch.mean(residual_1 ** 2) + 0.5 * torch.mean(residual_2 ** 2)
    total = loss_data + alpha * loss_pde + beta * loss_physics + lambda_delta * loss_delta
    return {
        "total": total,
        "data": loss_data,
        "delta": loss_delta,
        "physics": loss_physics,
        "pde": loss_pde,
    }
