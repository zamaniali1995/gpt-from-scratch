#!/usr/bin/env python3
"""Train a from-scratch GPT model.

Thin CLI wrapper around :mod:`gpt_from_scratch.train`. Requires the
package to be installed (``pip install -e .`` or ``uv sync``).

Examples:
    python scripts/train.py
    python scripts/train.py --config configs/gpt2_small.yaml --epochs 3 --out-dir checkpoints
"""

from gpt_from_scratch.train import main

if __name__ == "__main__":
    main()
