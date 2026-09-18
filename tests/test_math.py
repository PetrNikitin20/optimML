from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimml.losses import angle_degrees, effective_sample_size, pairwise_loss_grad, pointwise_loss_grad


class LossTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.chosen = rng.normal(size=(9, 7))
        self.rejected = rng.normal(size=(9, 7))
        self.theta = rng.normal(size=7)

    def directional_check(self, function, gradient, direction, epsilon=1e-6):
        finite = (function(self.theta + epsilon * direction) - function(self.theta - epsilon * direction)) / (2 * epsilon)
        self.assertAlmostEqual(finite, float(gradient @ direction), places=7)

    def test_pairwise_gradient(self):
        x = self.chosen - self.rejected
        _, gradient, _, _ = pairwise_loss_grad(self.theta, x, 0.8, 0.01)
        direction = np.arange(1, 8, dtype=float)
        direction /= np.linalg.norm(direction)
        self.directional_check(lambda t: pairwise_loss_grad(t, x, 0.8, 0.01)[0], gradient, direction)

    def test_pointwise_gradient(self):
        _, gradient, _, _ = pointwise_loss_grad(self.theta, self.chosen, self.rejected, 1.2, 0.01)
        direction = np.arange(1, 8, dtype=float)
        direction /= np.linalg.norm(direction)
        self.directional_check(lambda t: pointwise_loss_grad(t, self.chosen, self.rejected, 1.2, 0.01)[0], gradient, direction)

    def test_ess_bounds_and_angles(self):
        weights = np.array([1.0, 2.0, 3.0])
        self.assertGreaterEqual(effective_sample_size(weights), 1.0)
        self.assertLessEqual(effective_sample_size(weights), 3.0)
        self.assertAlmostEqual(angle_degrees(np.ones(3), np.ones(3)), 0.0)
        self.assertAlmostEqual(angle_degrees(np.array([1.0, 0]), np.array([0, 1.0])), 90.0)


if __name__ == "__main__":
    unittest.main()
