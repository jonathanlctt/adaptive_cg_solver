import torch
from scipy.linalg import cholesky as scipy_cholesky
from scipy.linalg import solve_triangular as scipy_solve_triangular
import numpy as np
from scipy.sparse import issparse
from ..sketching.sketches import sjlt


def get_reg_param_threshold(a):
    if isinstance(a, torch.Tensor):
        return np.sqrt(min(a.shape[0], a.shape[1]) * torch.finfo(a.dtype).eps)
    else:
        return np.sqrt(min(a.shape[0], a.shape[1]) * np.finfo(a.dtype).eps)


def get_max_sval_approx(a, m=6, niter=2):
    m = 6 if m is None else m
    n, d = a.shape[-2:]
    if isinstance(a, torch.Tensor):
        if (m > n or m > d) and not isinstance(a, torch.sparse.Tensor):
            sigma = torch.linalg.svdvals(a)
            return sigma[0].item()
        a_t = a.T
        s = torch.randn(n, m, dtype=a.dtype, device=a.device)
        q = torch.linalg.qr(a_t @ s).Q
        for i in range(niter):
            q = torch.linalg.qr(a @ q).Q
            q = torch.linalg.qr(a_t @ q).Q
        b_t = a @ q
        sigma = torch.linalg.svdvals(b_t)
    else:
        if (m > n or m > d) and not issparse(a):
            sigma = np.linalg.svd(a, compute_uv=False)
            return sigma[0].item()
        a_t = a.T
        if issparse(a):
            sa_t = sjlt(a.tocsc(), m, nnz_per_column=1).toarray().T
        else:
            s = np.random.randn(n, m).astype(a.dtype)
            sa_t = a_t @ s
        q = np.linalg.qr(sa_t)[0]
        for i in range(niter):
            q = np.linalg.qr(a @ q)[0]
            q = np.linalg.qr(a_t @ q)[0]
        b_t = a @ q
        sigma = np.linalg.svd(b_t, compute_uv=False)
    return sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma[0]


def cholesky(h, lower=False):
    if isinstance(h, torch.Tensor):
        if lower:
            return torch.linalg.cholesky(h)
        else:
            return torch.linalg.cholesky(h).mH
    else:
        return scipy_cholesky(h, lower=lower)


def solve_triangular(L, y, lower=False):
    if isinstance(L, torch.Tensor):
        return torch.linalg.solve_triangular(L, y, upper=not lower)
    else:
        return scipy_solve_triangular(L, y, lower=lower)

