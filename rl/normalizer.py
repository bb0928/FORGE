"""EmpiricalNormalization: running mean/std obs whitening.

Standard rsl-rl style. Normalizes per-dim using running statistics updated
during training. Eval mode freezes stats. Applied to the per-frame obs
(before mask/impute), so each of the 131 (actor) / 146 (critic) dims is whitened.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    def __init__(self, shape, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until  # stop updating after this many samples (None = always)
        self.register_buffer("_mean", torch.zeros(1, shape))
        self.register_buffer("_var", torch.ones(1, shape))
        self.register_buffer("_std", torch.ones(1, shape))
        self.register_buffer("_count", torch.zeros(1, dtype=torch.long))
        self._training = True

    def train(self, mode=True):
        self._training = mode
        return super().train(mode)

    def eval(self):
        return self.train(False)

    @property
    def mean(self):
        return self._mean.squeeze(0)

    @property
    def std(self):
        return self._std.squeeze(0)

    def forward(self, x):
        # x: (..., shape) — normalize last dim
        if self._training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.no_grad()
    def update(self, x):
        if self.until is not None and self._count.item() >= self.until:
            return
        # flatten leading dims
        flat = x.reshape(-1, x.shape[-1])
        n = flat.shape[0]
        if n == 0:
            return
        count = self._count.item()
        new_count = count + n
        batch_mean = flat.mean(dim=0, keepdim=True)
        batch_var = flat.var(dim=0, unbiased=False, keepdim=True)
        delta = batch_mean - self._mean
        # Welford-style parallel update
        self._mean += delta * (n / new_count)
        m_a = self._var * count
        m_b = batch_var * n
        M2 = m_a + m_b + delta.pow(2) * (count * n / new_count)
        self._var = M2 / new_count
        self._std = self._var.sqrt()
        self._count += n
