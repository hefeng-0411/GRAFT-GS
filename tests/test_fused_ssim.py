"""Numerical and dispatch tests for the memory-constant SSIM kernel."""

from __future__ import annotations

import unittest

import torch

from graft_gs.engine.losses import (
    _recomputed_dense_loss,
    multiscale_perceptual_loss,
    robust_rgb,
    structural_similarity_loss,
)
from graft_gs.kernels.fused_ssim import _reference_ssim_loss, triton_ssim_available


class FusedSSIMTests(unittest.TestCase):
    def test_cpu_path_preserves_float64_reference(self) -> None:
        generator = torch.Generator().manual_seed(83)
        predicted = torch.rand(
            2, 3, 3, 7, 9, generator=generator, dtype=torch.float64
        ).requires_grad_()
        target = torch.rand(
            2, 3, 3, 7, 9, generator=generator, dtype=torch.float64
        )
        mask = torch.rand(
            2, 3, 1, 7, 9, generator=generator, dtype=torch.float64
        )
        actual = structural_similarity_loss(predicted, target, mask)
        expected = _reference_ssim_loss(predicted, target, mask)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_rejects_non_rgb_input(self) -> None:
        value = torch.zeros(1, 1, 4, 3, 3)
        with self.assertRaisesRegex(ValueError, "three color channels"):
            structural_similarity_loss(value, value)

    def test_dense_recomputation_preserves_value_and_adjoint(self) -> None:
        generator = torch.Generator().manual_seed(97)
        predicted = torch.rand(
            1, 2, 3, 12, 14, generator=generator, dtype=torch.float64
        ).requires_grad_()
        target = torch.rand_like(predicted)
        mask = torch.rand(
            1, 2, 1, 12, 14, generator=generator, dtype=torch.float64
        )

        def objective(argument: torch.Tensor, recompute: bool) -> torch.Tensor:
            invoke = (
                _recomputed_dense_loss
                if recompute
                else lambda function, left, right, weight: function(
                    left, right, weight
                )
            )
            return invoke(robust_rgb, argument, target, mask) + invoke(
                multiscale_perceptual_loss, argument, target, mask
            )

        expected = objective(predicted, False)
        expected_gradient = torch.autograd.grad(expected, predicted)[0]
        recomputed_input = predicted.detach().clone().requires_grad_()
        actual = objective(recomputed_input, True)
        actual_gradient = torch.autograd.grad(actual, recomputed_input)[0]
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=0.0, rtol=0.0
        )

    @unittest.skipUnless(
        triton_ssim_available(), "requires a CUDA device and Triton"
    )
    def test_cuda_forward_and_adjoint_match_eager_oracle(self) -> None:
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        generator = torch.Generator(device=device).manual_seed(109)
        predicted = torch.rand(
            1, 2, 3, 13, 17, generator=generator, device=device
        ).requires_grad_()
        target = torch.rand(
            1, 2, 3, 13, 17, generator=generator, device=device
        ).requires_grad_()
        mask = torch.rand(
            1, 2, 1, 13, 17, generator=generator, device=device
        )
        expected = _reference_ssim_loss(predicted, target, mask)
        expected_gradients = torch.autograd.grad(expected, (predicted, target))

        fused_predicted = predicted.detach().clone().requires_grad_()
        fused_target = target.detach().clone().requires_grad_()
        actual = structural_similarity_loss(fused_predicted, fused_target, mask)
        actual_gradients = torch.autograd.grad(
            actual, (fused_predicted, fused_target)
        )
        torch.testing.assert_close(actual, expected, atol=1.0e-7, rtol=2.0e-7)
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=8.0e-9,
                rtol=3.0e-5,
            )

    @unittest.skipUnless(
        triton_ssim_available(), "requires a CUDA device and Triton"
    )
    def test_cuda_dispatch_is_deterministic(self) -> None:
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        generator = torch.Generator(device=device).manual_seed(211)
        predicted = torch.rand(
            1, 1, 3, 23, 19, generator=generator, device=device
        )
        target = torch.rand_like(predicted)
        mask = torch.rand(
            1, 1, 1, 23, 19, generator=generator, device=device
        )
        values = []
        gradients = []
        for _ in range(3):
            argument = predicted.detach().clone().requires_grad_()
            value = structural_similarity_loss(argument, target, mask)
            value.backward()
            values.append(value.detach())
            gradients.append(argument.grad.detach())
        for value in values[1:]:
            torch.testing.assert_close(value, values[0], atol=0.0, rtol=0.0)
        for gradient in gradients[1:]:
            torch.testing.assert_close(
                gradient, gradients[0], atol=0.0, rtol=0.0
            )


if __name__ == "__main__":
    unittest.main()
