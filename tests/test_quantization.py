from __future__ import annotations

import unittest

import torch

from graft_gs.optimization.quantization import (
    BlockwiseFakeQuantizer,
    QuantizationConfig,
    attention_score_error_bound,
    topology_step_is_certified,
)


class QuantizationCertificateTest(unittest.TestCase):
    def test_block_error_and_topology_certificate(self) -> None:
        value = torch.linspace(-1.0, 1.0, 257, dtype=torch.float32)
        quantizer = BlockwiseFakeQuantizer(QuantizationConfig(bits=8, block_size=16, stochastic_rounding=False)).eval()
        quantized = quantizer(value)
        self.assertLess(float((quantized - value).abs().max()), 0.01)
        query_error = 0.01
        bound = attention_score_error_bound(query_error, 0.5)
        self.assertAlmostEqual(float(bound), (2 * query_error + query_error**2) / 0.5, places=7)
        self.assertTrue(bool(topology_step_is_certified(query_error, 0.5, 2.0, 0.1, 0.1)))
        self.assertFalse(bool(topology_step_is_certified(0.5, 0.1, 5.0, 0.5, 0.01)))

    def test_ste_gradient_is_finite(self) -> None:
        value = torch.linspace(-0.7, 0.9, 64, dtype=torch.float64, requires_grad=True)
        quantizer = BlockwiseFakeQuantizer(
            QuantizationConfig(bits=4, block_size=16, stochastic_rounding=False)
        ).double().eval()
        loss = quantizer(value).square().sum()
        loss.backward()
        torch.testing.assert_close(value.grad, 2.0 * quantizer(value).detach())

    def test_bounded_scale_adversary_changes_forward_and_has_gradient(self) -> None:
        value = torch.linspace(-0.7, 0.9, 64, dtype=torch.float64, requires_grad=True)
        quantizer = BlockwiseFakeQuantizer(
            QuantizationConfig(
                bits=4,
                block_size=16,
                stochastic_rounding=False,
                adversarial_log_scale_radius=0.1,
            )
        ).double().eval()
        baseline = quantizer(value)
        scale_gradient = torch.autograd.grad(
            baseline.square().sum(), quantizer.adversarial_log_scale
        )[0]
        self.assertTrue(bool(torch.isfinite(scale_gradient)))
        quantizer.set_worst_case_from_gradient(scale_gradient)
        adversarial = quantizer(value)
        self.assertFalse(torch.equal(baseline, adversarial))
        self.assertLessEqual(
            abs(float(quantizer.adversarial_log_scale)), 0.1 + 1.0e-12
        )
        quantizer.reset_adversary()
        self.assertEqual(float(quantizer.adversarial_log_scale), 0.0)


if __name__ == "__main__":
    unittest.main()
