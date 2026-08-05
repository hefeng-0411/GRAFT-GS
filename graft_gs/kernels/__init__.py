"""Optional hardware kernels with numerically audited PyTorch fallbacks."""

from .fused_ssim import fused_ssim_loss, triton_ssim_available

__all__ = ["fused_ssim_loss", "triton_ssim_available"]
