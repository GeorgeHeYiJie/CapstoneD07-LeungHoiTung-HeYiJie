"""Cross-condition SOH model.

Wu et al. (2023) turn each cycle-feature sequence into period-based 2D
matrices. Zhang et al. (2025) compare a target cell with a reference cell.
Wang et al. (2024) score those SOH predictions with a data loss, a dynamics
residual and a degradation-direction penalty.

This package is a new experiment. It does not modify the PINN baseline.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
