"""Entry point executed under torchrun for multi-rank evidence validation."""

from __future__ import annotations

import unittest


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(
        "tests", pattern="test_distributed_evidence.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
