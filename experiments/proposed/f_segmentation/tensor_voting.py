"""WBBS denoising primitives vendored from compared_methods/methods/wbbs/wbbs.py.

Source SHA-256: 45271a147fccbcad0fbe40ad9fcfb29acbcb7aad18b873fbe63b98f65fab623b
Only _points, tensor_vote, tensor_saliencies and retained_points are copied.
No sibling repository is needed. sigma_d=9 is an implementation convention;
this supplies denoising, not the 2022 marching segmentation algorithm.
"""
from __future__ import annotations
import numpy as np


def _points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite (N, 2) array")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Voting sites must be unique")
    return points


def tensor_vote(points: np.ndarray, sigma_d: float = 9.0, block_size: int = 256) -> np.ndarray:
    """Exact Eq. (2), (6)-(8), with each voter initialized to I and no self-vote.

    For K_j=I, R(I-0.5 rr^T)R^T = I-0.5 rr^T, so this is the closed-form
    matrix expression, not an approximation by a density feature. All other
    defective points vote once; there is no radius cutoff or point sampling.
    Units in exp(-||xi-xj||^2/sigma_d) are raw pixel coordinates.
    """
    points = _points(points)
    if not np.isfinite(sigma_d) or sigma_d <= 0:
        raise ValueError("sigma_d must be finite and positive")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    tensors = np.zeros((len(points), 2, 2), dtype=np.float64)
    for start in range(0, len(points), block_size):
        delta = points[start:start + block_size, None, :] - points[None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", delta, delta)
        nonzero = distance2 > 0
        decay = np.exp(-distance2 / sigma_d) * nonzero
        # c * (I - .5 rr^T); zero-distance terms are excluded completely.
        scale = np.divide(0.5 * decay, distance2, out=np.zeros_like(decay), where=nonzero)
        tensors[start:start + len(delta)] = -np.einsum("ij,ija,ijb->iab", scale, delta, delta)
        total = decay.sum(axis=1)
        tensors[start:start + len(delta), 0, 0] += total
        tensors[start:start + len(delta), 1, 1] += total
    return tensors


def tensor_saliencies(tensors: np.ndarray) -> np.ndarray:
    """Eq. (9)-(10): columns are lambda_1-lambda_2 and lambda_2."""
    tensors = np.asarray(tensors, dtype=np.float64)
    if tensors.ndim != 3 or tensors.shape[1:] != (2, 2) or not np.isfinite(tensors).all():
        raise ValueError("Expected finite (N, 2, 2) tensors")
    if not np.allclose(tensors, tensors.swapaxes(1, 2), atol=1e-12, rtol=1e-12):
        raise ValueError("Tensor voting output must be symmetric")
    eigenvalues = np.linalg.eigvalsh(tensors)
    if np.any(eigenvalues < -1e-10):
        raise ValueError("Tensor voting output must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0)
    return np.column_stack((eigenvalues[:, 1] - eigenvalues[:, 0], eigenvalues[:, 0]))


def retained_points(saliencies: np.ndarray, threshold: float = 0.32) -> np.ndarray:
    """Section 2.1: remove only when BOTH stick and ball are strictly below t*max."""
    saliencies = np.asarray(saliencies, dtype=np.float64)
    if saliencies.ndim != 2 or saliencies.shape[1] != 2 or not np.isfinite(saliencies).all() or np.any(saliencies < 0):
        raise ValueError("Expected nonnegative finite (N, 2) saliencies")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    if len(saliencies) == 0:
        return np.empty(0, dtype=bool)
    return ~np.all(saliencies < threshold * saliencies.max(axis=0), axis=1)
