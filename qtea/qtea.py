"""QTEA: Quantized Ternary Error Adaptation for a single linear layer.

The layer weight is turned into a ternary base plus a sparse FP8 residual by a
GPTQ-style column-by-column sweep with three additions:

  * salient columns are picked first and carry a 1:4 semi-sparse FP8 residual
    that corrects the ternary error (`_select_salient`);
  * every column gets a scalar rescale factor v_j that is refined jointly with
    its ternary codes (`_rescale_refine`);
  * the GPTQ error propagation is damped along the block so that late columns,
    which have little capacity left to absorb error, are not over-compensated.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .quantizer import TernaryGroup, ternary_assign

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

# The sweep is a hard-threshold loop with error feedback, so reduced-precision
# matmuls change which codes get assigned. Keep the FP32 paths at full width.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@dataclass
class QTEAConfig:
    group_size: int = 128       # columns per quantization group (== GPTQ block size)
    salient_ratio: float = 0.05  # fraction of columns that carry an FP8 residual
    residual_group: int = 4      # 1:N semi-sparse pattern inside a salient column
    delta_coef: float = 0.96     # initial dead zone, as a multiple of the mean |W|
    itf_iters: int = 1           # iterative ternary fitting rounds per group
    rescale_iters: int = 2       # column-wise rescale refinement rounds (0 disables)
    decay: float = 1.0           # lambda of the GPTQ error decay (0 disables)
    damp: float = 0.007          # Hessian dampening, as a fraction of its mean diagonal


class QTEA:
    """Accumulates a calibration Hessian for one `nn.Linear`, then quantizes it."""

    def __init__(self, layer, config: QTEAConfig):
        self.layer = layer
        self.config = config
        self.device = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.hessian = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

        if self.columns % config.group_size:
            raise ValueError(f"{self.columns} columns is not a multiple of the group size")
        if self.rows % config.residual_group:
            raise ValueError(f"{self.rows} rows is not a multiple of the residual group")

    @torch.no_grad()
    def add_batch(self, inp):
        """Fold one calibration batch into the running Hessian H = 2/N * sum X X^T.

        As in GPTQ, N counts calibration *sequences* rather than tokens. The
        choice only rescales H, which the sweep is invariant to, but keeping it
        makes the arithmetic match the reference implementation bit for bit.
        """
        sequences = inp.shape[0] if inp.dim() == 3 else 1
        inp = inp.reshape(-1, inp.shape[-1]).t()
        self.hessian *= self.nsamples / (self.nsamples + sequences)
        self.nsamples += sequences
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.hessian += inp.matmul(inp.t())

    def free(self):
        self.hessian = None
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- ordering

    def _select_salient(self, W, hessian_diag):
        """Rank columns by max_i (W_ij^2 * H_jj^2) and keep the top ones (Eq. 3).

        The count is rounded to whole groups, so a salient column never splits a
        quantization group.
        """
        if self.config.salient_ratio <= 0:
            n_salient = 0
        else:
            blocks = self.columns // self.config.group_size
            n_salient = max(1, round(blocks * self.config.salient_ratio)) * self.config.group_size
            n_salient = min(n_salient, self.columns)

        score = torch.max(W ** 2 * hessian_diag.reshape(1, -1) ** 2, dim=0).values
        order = torch.argsort(score, descending=True)
        return order[:n_salient], order[n_salient:]

    def _ldl_order(self, H, columns):
        """Order non-salient columns by the pivots of a symmetric LDL^T of H.

        The pivot order puts well-conditioned columns first, which gives the
        GPTQ sweep a better sequence for compensating error.
        """
        if len(columns) == 0:
            return columns
        from scipy.linalg._decomp_ldl import _ldl_sanitize_ipiv
        from scipy.linalg.lapack import get_lapack_funcs

        sub = H[columns][:, columns].clone()
        diag = torch.arange(len(columns), device=sub.device)
        sub[diag, diag] += self.config.damp * torch.diag(sub).mean()
        # `sub` is symmetric, so its transpose is already Fortran-contiguous for LAPACK.
        matrix = sub.t().cpu().float().numpy()
        del sub

        try:
            sytrf, sytrf_lwork = get_lapack_funcs(("sytrf", "sytrf_lwork"), (matrix,))
            lwork = int(sytrf_lwork(matrix.shape[0], lower=True)[0])
            _, ipiv, info = sytrf(matrix, lwork=lwork, lower=True, overwrite_a=True)
            if info < 0:
                raise RuntimeError(f"LAPACK sytrf: illegal argument {-info}")
            swaps, _ = _ldl_sanitize_ipiv(ipiv, lower=True)
        except Exception as exc:  # keep the importance order if LAPACK is unhappy
            print(f"  [ldl] falling back to importance order ({exc})")
            return columns

        state = np.arange(len(columns))
        for i in range(len(columns) - 1, -1, -1):
            if swaps[i] != i:
                state[[swaps[i], i]] = state[[i, swaps[i]]]
        permutation = torch.from_numpy(np.argsort(state)).to(columns.device)
        return columns[permutation]

    def _inverse_cholesky(self, H):
        """Upper-triangular Cholesky factor of H^-1, retrying with more damping."""
        diag = torch.arange(self.columns, device=H.device)
        H[diag, diag] += self.config.damp * torch.diag(H).mean()
        for _ in range(5):
            try:
                inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))
                return torch.linalg.cholesky(inverse, upper=True)
            except torch.linalg.LinAlgError:
                H[diag, diag] += 0.01 * torch.diag(H).mean()
        print("  [chol] Hessian not positive definite, using its diagonal")
        return torch.diag(1.0 / torch.diag(H).sqrt())

    # ------------------------------------------------------------ column steps

    def _rescale_refine(self, centered, group):
        """Alternate the column rescale v_j with a ternary reassignment (Eq. 4-5)."""
        codes = ternary_assign(centered, group.delta)
        if self.config.rescale_iters == 0:
            return codes, 1.0

        alpha_mean, v = group.alpha.mean(), 1.0
        for _ in range(self.config.rescale_iters):
            nonzero = codes != 0
            if nonzero.any():  # Eq. 5
                v = (centered[nonzero].abs().mean() / alpha_mean).clamp(min=0.01).item()
            else:
                # An overshooting rescale can push every level past the data and
                # zero the whole column. Reset to v = 1 so the next round re-fits
                # from a live assignment instead of compounding the overshoot.
                v = 1.0
            level = group.alpha * v
            # Nearest of {-level, 0, +level}, ties resolved towards the outer levels.
            to_pos, to_neg, to_zero = (centered - level).abs(), (centered + level).abs(), centered.abs()
            codes = torch.zeros_like(centered)
            codes[(to_pos <= to_neg) & (to_pos <= to_zero)] = 1.0
            codes[(to_neg < to_pos) & (to_neg <= to_zero)] = -1.0
        return codes, v

    def _sparse_residual(self, residual):
        """1:N semi-sparse FP8 approximation of a salient column's residual.

        Keeps the largest entry of every group of N rows and rounds it to FP8
        (E4M3) under one scale per column, so that the peak of the column maps
        onto the largest finite FP8 value.
        """
        grouped = residual.view(-1, self.config.residual_group)
        index = grouped.abs().argmax(dim=1, keepdim=True)
        value = grouped.gather(1, index)

        scale = FP8_MAX / value.abs().max().clamp(min=1e-10)
        value = (value * scale).to(torch.float8_e4m3fn).to(residual.dtype) / scale

        dense = torch.zeros_like(grouped).scatter(1, index, value)
        return dense.view_as(residual), index.flatten().to(torch.uint8), value.flatten(), scale

    # ------------------------------------------------------------------ driver

    @torch.no_grad()
    def quantize(self):
        """Run the QTEA sweep. Writes the dequantized weight back into the layer
        and returns everything needed to store the layer in packed form."""
        cfg = self.config
        W = self.layer.weight.data.clone().float()
        H = self.hessian
        self.hessian = None

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Group parameters are fitted once, on the original weights.
        groups = [
            TernaryGroup(cfg.delta_coef, cfg.itf_iters).fit(W[:, i:i + cfg.group_size])
            for i in range(0, self.columns, cfg.group_size)
        ]

        # Salient columns are quantized first, the rest in pivoted LDL order.
        salient, rest = self._select_salient(W, torch.diag(H))
        n_salient = len(salient)
        order = torch.cat([salient, self._ldl_order(H, rest)])
        inverse_order = torch.argsort(order)

        W = W[:, order]
        Hinv = self._inverse_cholesky(H[order][:, order])
        del H
        torch.cuda.empty_cache()

        Q = torch.zeros_like(W)
        codes_all = torch.zeros(self.rows, self.columns, dtype=torch.int8, device=self.device)
        v_all = torch.ones(self.columns, device=self.device)
        residuals, residual_scales = [], []
        group_of_column = (order // cfg.group_size).tolist()
        error = torch.zeros((), device=self.device)

        for start in range(0, self.columns, cfg.group_size):
            stop = start + cfg.group_size
            block, block_Q = W[:, start:stop].clone(), torch.zeros(self.rows, cfg.group_size, device=self.device)
            block_err = torch.zeros_like(block_Q)
            block_Hinv = Hinv[start:stop, start:stop]
            hinv_mean = torch.diag(block_Hinv).mean()

            for i in range(cfg.group_size):
                column = block[:, i:i + 1]
                group = groups[group_of_column[start + i]]

                centered = column - group.beta
                codes, v = self._rescale_refine(centered, group)
                q = group.alpha * v * codes + group.beta

                if start + i < n_salient:
                    dense, index, value, scale = self._sparse_residual(column - q)
                    q = q + dense
                    residuals.append((start + i, index, value))
                    residual_scales.append(scale)

                block_Q[:, i] = q.flatten()
                codes_all[:, start + i] = codes.flatten().to(torch.int8)
                v_all[start + i] = v

                # GPTQ update, damped by gamma_i so that late columns in the
                # block propagate less error than early ones (Eq. 8-9).
                d = block_Hinv[i, i]
                residual_error = (column.flatten() - q.flatten()) / d
                error += (residual_error ** 2).sum() / 2
                curvature = (d / hinv_mean).clamp(0.3, 3.0).item()
                gamma = math.exp(-cfg.decay * curvature * i / cfg.group_size)
                block[:, i:] -= gamma * residual_error.unsqueeze(1).matmul(block_Hinv[i, i:].unsqueeze(0))
                block_err[:, i] = residual_error

            Q[:, start:stop] = block_Q
            W[:, stop:] -= block_err.matmul(Hinv[start:stop, stop:])

        print(f"  proxy loss {error.item():.4f}")

        self.layer.weight.data = Q[:, inverse_order].reshape(self.layer.weight.shape).to(
            self.layer.weight.dtype
        )
        return self._payload(groups, codes_all, v_all, order, inverse_order, residuals, residual_scales)

    def _payload(self, groups, codes, v, order, inverse_order, residuals, residual_scales):
        """Collect the quantized layer in original column order, ready for packing."""
        payload = {
            "rows": self.rows,
            "cols": self.columns,
            "group_size": self.config.group_size,
            "residual_group": self.config.residual_group,
            "ternary": codes[:, inverse_order].cpu(),
            "col_v": v[inverse_order].cpu(),
            "group_alpha": torch.cat([group.alpha for group in groups], dim=1).cpu(),
            "group_beta": torch.cat([group.beta for group in groups], dim=1).cpu(),
        }

        if not residuals:
            payload.update(
                salient_cols=torch.zeros(0, dtype=torch.int32),
                residual_index=torch.zeros(0, 0, dtype=torch.uint8),
                residual_value=torch.zeros(0, 0),
                residual_scale=torch.zeros(0),
            )
            return payload

        # `residuals` is indexed by position in the quantization order; store the
        # salient columns by their original index instead.
        positions = torch.tensor([position for position, _, _ in residuals], device=self.device)
        columns = order[positions]
        by_column = torch.argsort(columns).tolist()

        payload.update(
            salient_cols=columns[by_column].to(torch.int32).cpu(),
            residual_index=torch.stack([residuals[i][1] for i in by_column]).cpu(),
            residual_value=torch.stack([residuals[i][2] for i in by_column]).cpu(),
            residual_scale=torch.stack([residual_scales[i] for i in by_column]).cpu(),
        )
        return payload
