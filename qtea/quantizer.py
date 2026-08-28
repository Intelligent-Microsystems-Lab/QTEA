"""Group-wise ternary quantizer.

Every weight is approximated by a ternary code plus two per-row parameters that
are shared by all columns of one quantization group (128 columns by default):

    W_ij ~= beta_i + alpha_i * T_ij ,      T_ij in {-1, 0, +1}

`TernaryGroup.fit` estimates (alpha, beta) from the full-precision weights of a
single group; `qtea.py` then refines a per-column rescale factor on top of them.
"""

import torch


def ternary_assign(centered, delta):
    """Nearest ternary code for `centered` under a symmetric dead zone `delta`."""
    codes = torch.zeros_like(centered)
    codes[centered > delta] = 1.0
    codes[centered < -delta] = -1.0
    return codes


class TernaryGroup:
    """Per-row ternary parameters (alpha, beta, delta) for one column group."""

    def __init__(self, delta_coef=0.96, itf_iters=1):
        self.delta_coef = delta_coef
        self.itf_iters = itf_iters
        self.alpha = None   # (rows, 1) magnitude of the +-1 levels
        self.beta = None    # (rows, 1) quantization center
        self.delta = None   # (rows, 1) dead-zone threshold

    def fit(self, W):
        """Estimate the group parameters from `W` of shape (rows, group_size)."""
        beta = W.mean(dim=1, keepdim=True)
        centered = W - beta
        magnitude = centered.abs()

        # Initial guess: a dead zone proportional to the mean absolute deviation,
        # and a level magnitude equal to the mean of the entries that survive it.
        delta = self.delta_coef * magnitude.mean(dim=1, keepdim=True)
        alpha = torch.where(magnitude > delta, magnitude, torch.nan).nanmean(dim=1, keepdim=True)

        # Iterative ternary fitting: alternate a least-squares re-solve of
        # (alpha, beta) with a reassignment of the ternary codes.
        codes = ternary_assign(centered, delta)
        for _ in range(self.itf_iters):
            alpha, shift = self._least_squares(centered, codes, fallback=alpha)
            delta = 0.5 * alpha
            new_codes = ternary_assign(centered - shift, delta)
            if torch.equal(new_codes, codes):
                break
            codes = new_codes
        if self.itf_iters > 0:
            beta = beta + shift

        self.alpha, self.beta, self.delta = alpha, beta, delta
        return self

    @staticmethod
    def _least_squares(centered, codes, fallback):
        """Row-wise solution of  min_{a, s} || centered - s - a * codes ||^2."""
        n = centered.shape[1]
        cc = (codes * codes).sum(dim=1, keepdim=True)
        c1 = codes.sum(dim=1, keepdim=True)
        rc = (centered * codes).sum(dim=1, keepdim=True)
        r1 = centered.sum(dim=1, keepdim=True)

        det = cc * n - c1 * c1
        ok = det.abs() > 1e-8
        det = torch.where(ok, det, torch.ones_like(det))

        alpha = torch.where(ok, (n * rc - c1 * r1) / det, fallback)
        shift = torch.where(ok, (cc * r1 - c1 * rc) / det, torch.zeros_like(r1))
        return alpha.abs().clamp(min=1e-8), shift
