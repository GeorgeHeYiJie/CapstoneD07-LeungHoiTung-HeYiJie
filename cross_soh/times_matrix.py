"""Wu et al. 2023: turn a 1D multivariate series into period-based 2D matrices.

The paper writes the selected period as ``p_i = ceil(T / f_i)`` and places
that series in a ``p_i`` by ``f_i`` tensor. The working TimesNet procedure,
which is what we implement, is:

1. FFT along time, average the amplitude over channels, ignore the DC bin.
2. Keep the top-k frequency indices ``f``.
3. Period length ``p = T // f`` (at least 1, at most T).
4. Zero-pad time so its length is a multiple of ``p``.
5. Reshape to ``[n_periods, p, channels]``.

Rows are successive periods (inter-period variation). Columns are the phase
inside one period (intra-period variation). The same reshape is applied to
every channel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dominant_periods(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shared period lengths and per-sample amplitudes.

    Parameters
    ----------
    x:
        Series of shape ``[batch, time, channels]``.
    k:
        Number of non-DC frequencies to keep.

    Returns
    -------
    periods:
        Integer tensor ``[k_eff]``. One set of periods is estimated for the
        whole batch, by averaging amplitudes, as in TimesNet.
    amplitudes:
        Tensor ``[batch, k_eff]`` used to weight the period branches.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [batch, time, channels], got {tuple(x.shape)}")
    time_steps = x.shape[1]
    if time_steps < 2:
        raise ValueError("need at least 2 time steps to estimate a period")
    if k < 1:
        raise ValueError("k must be positive")

    spectrum = torch.fft.rfft(x, dim=1)
    # Mean over channels. Bin 0 is the constant offset, not a period.
    mean_amplitude = spectrum.abs().mean(dim=0).mean(dim=-1).clone()
    mean_amplitude[0] = 0
    usable = int((mean_amplitude[1:] > 0).sum().item()) if mean_amplitude.numel() > 1 else 0
    k_eff = min(k, max(mean_amplitude.numel() - 1, 1))
    if usable == 0:
        # Flat series: still return k_eff legal periods so the reshape is defined.
        k_eff = min(k, max(time_steps // 2, 1))
    _, top = torch.topk(mean_amplitude, k_eff)
    top = top.detach().clamp(min=1)
    periods = (time_steps // top).clamp(min=1, max=time_steps)
    sample_amplitude = spectrum.abs().mean(dim=-1)[:, top]
    return periods, sample_amplitude


def reshape_to_2d(x: torch.Tensor, period: int) -> torch.Tensor:
    """Pad ``x`` and reshape one period.

    Parameters
    ----------
    x:
        ``[batch, time, channels]``.
    period:
        Intra-period length. Must be in ``1 .. time``.

    Returns
    -------
    Tensor ``[batch, n_periods, period, channels]``.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [batch, time, channels], got {tuple(x.shape)}")
    time_steps = x.shape[1]
    if period < 1 or period > time_steps:
        raise ValueError(f"period {period} is outside 1..{time_steps}")
    if time_steps % period != 0:
        padded_length = (time_steps // period + 1) * period
        x = F.pad(x, (0, 0, 0, padded_length - time_steps))
    else:
        padded_length = time_steps
    n_periods = padded_length // period
    return x.reshape(x.shape[0], n_periods, period, x.shape[2])


def series_to_period_matrices(x: torch.Tensor, k: int) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Convert one batch of 1D series into Wu's set of 2D matrices.

    This is the data-side transform. The network calls the same two helpers
    again inside each TimesBlock, because later layers see a new sequence.
    """
    periods, amplitudes = dominant_periods(x, k)
    matrices = [reshape_to_2d(x, int(period.item())) for period in periods]
    return matrices, periods, amplitudes


class Inception2d(torch.nn.Module):
    """Parameter-efficient 2D inception used by TimesBlock.

    Kernel sizes are ``1, 3, ..., 2*n-1`` with matching padding, so the
    period grid keeps its shape. Outputs of the kernels are averaged.
    """

    def __init__(self, channels: int, n_kernels: int = 3):
        super().__init__()
        if n_kernels < 1:
            raise ValueError("n_kernels must be positive")
        self.kernels = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=2 * i + 1,
                    padding=i,
                )
                for i in range(n_kernels)
            ]
        )
        for conv in self.kernels:
            torch.nn.init.xavier_normal_(conv.weight)
            torch.nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, n_periods, period]
        stacked = torch.stack([kernel(x) for kernel in self.kernels], dim=-1)
        return stacked.mean(dim=-1)


class TimesBlock(torch.nn.Module):
    """One TimesNet block: 2D matrices, inception, amplitude-weighted sum."""

    def __init__(self, channels: int, k: int = 3, n_kernels: int = 3):
        super().__init__()
        self.k = k
        self.inception = Inception2d(channels, n_kernels=n_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, channels]
        periods, amplitudes = dominant_periods(x, self.k)
        branches = []
        for period in periods:
            matrix = reshape_to_2d(x, int(period.item()))
            # 2D kernels see channels first, then inter-period, then intra-period.
            grid = matrix.permute(0, 3, 1, 2).contiguous()
            mixed = self.inception(grid).permute(0, 2, 3, 1).contiguous()
            restored = mixed.reshape(x.shape[0], -1, x.shape[2])[:, : x.shape[1], :]
            branches.append(restored)
        stacked = torch.stack(branches, dim=-1)
        weights = torch.softmax(amplitudes, dim=-1)
        aggregated = (stacked * weights[:, None, None, :]).sum(dim=-1)
        return aggregated + x
