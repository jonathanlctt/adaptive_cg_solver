import torch
import numpy as np
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc

from ..linear_algebra.linear_algebra import cholesky, solve_triangular, get_max_sval_approx, get_reg_param_threshold

import logging
logging.basicConfig(level=logging.INFO)


def cholesky_factorization_wrapper(upper_mat):
    def factorization(z):
        return solve_triangular(upper_mat, solve_triangular(upper_mat.T, z, lower=True), lower=False)

    return factorization


class QuadraticSolver:
    
    def __init__(self, a, b, reg_param, x_opt=None, rescale_data=True, check_reg_param=True, least_squares=True,
                 enforce_cuda=False, device_index=1):

        if b.ndim == 1:
            b = b.reshape((-1, 1))

        if not isinstance(a, torch.Tensor) and not issparse(a):
            a = torch.from_numpy(a)
            b = torch.from_numpy(b)

        if isinstance(a, torch.Tensor):
            if a.is_cuda:
                self.device = a.device
                b = b.cuda(device=self.device)
            elif enforce_cuda and torch.cuda.is_available():
                self.device = torch.device(type='cuda', index=device_index)
            else:
                self.device = torch.device(type='cpu')
        else:
            self.device = None

        if rescale_data:
            sigma_top = get_max_sval_approx(a=a, niter=1)
            logging.info(f"Rescaling data by max singular value: {sigma_top=}")
            a = a / sigma_top
            b = b / sigma_top if least_squares else b / np.sqrt(sigma_top)
            reg_param = reg_param / sigma_top

        if check_reg_param:
            threshold = get_reg_param_threshold(a=a)
            if threshold > reg_param:
                logging.warning(f"Numerical precision issue for dtype={a.dtype} - regularization parameter is too small: {reg_param=} < {threshold=}")
            self.nu = max(reg_param, threshold)
        else:
            self.nu = reg_param

        self.n_samples_a = a.shape[0]
        self.n_features_a = a.shape[1]

        assert self.n_samples_a >= self.n_features_a, "n_samples must be greater than n_features"

        self.with_torch = isinstance(a, torch.Tensor)
        self.is_sparse = issparse(a)
        self._dtype = a.dtype

        self.X = a
        if self.is_sparse:
            if least_squares:
                y_ = (self.X.T @ b)
            else:
                y_ = b
            if issparse(y_):
                self.y = y_.A
            else:
                self.y = y_
        else:
            if least_squares:
                self.y = self.X.T @ b
            else:
                self.y = b

        self.n, self.d = self.X.shape
        self.c = b.shape[1]
    
        if x_opt is not None and x_opt.ndim == 1:
            x_opt = x_opt.reshape((-1, 1))
        self.x_opt = x_opt

        if self.with_torch and self.x_opt is not None and not isinstance(self.x_opt, torch.Tensor):
            self.x_opt = torch.from_numpy(self.x_opt)
            if a.is_cuda:
                self.x_opt = self.x_opt.cuda(device=self.device, non_blocking=True)
    
        self.norm_sq_b = 0.5 * (b.multiply(b)).sum() if self.is_sparse and not isinstance(b, np.ndarray) else 0.5 * (b ** 2).sum()

    def put_data_on_device(self, non_blocking=False):
        if self.with_torch:
            self.X = self.X.to(device=self.device, non_blocking=non_blocking)
            self.y = self.y.to(device=self.device, non_blocking=non_blocking)
            if self.x_opt is not None:
                self.x_opt = self.x_opt.to(device=self.device, non_blocking=non_blocking)

    def compute_error(self, x):
        if x.ndim == 1:
            x = x.reshape((-1, 1))
    
        if self.x_opt is not None:
            err_ = 0.5 * ((self.X @ (x - self.x_opt)) ** 2).sum()
        else:
            err_ = 0.5 * ((self.X @ x) ** 2).sum() - (self.y * x).sum() + self.norm_sq_b
    
        return err_.item()

    def id_mat(self, d):
        return torch.eye(d, device=self.device, dtype=self._dtype) if self.with_torch else np.eye(d, dtype=self._dtype)

    def _copy(self, x):
        return torch.clone(x) if self.with_torch else np.copy(x)
    
    def _zeros(self, n, d):
        return torch.zeros(n, d, device=self.device, dtype=self._dtype) if self.with_torch else np.zeros((n, d), dtype=self._dtype)

    def hv_product(self, v):
        if v.ndim == 1:
            v = v.reshape((-1, 1))
        hv = self.X.T @ (self.X @ v) + self.nu ** 2 * v
        return hv

    def factor_approx_hessian(self, sa, sasa):
        self.sa = sa
        top_singular_value = get_max_sval_approx(sasa, niter=2)
        threshold = np.sqrt(top_singular_value) * get_reg_param_threshold(sasa)
        if threshold > self.nu:
            logging.warning(f"PCG INNER SOLVER: precision issue for dtype={sasa.dtype} - regularization parameter too small: {self.nu} < {threshold=}")
        self.sasa_nu = max(self.nu, threshold)

        upper_mat = cholesky(sasa + self.sasa_nu**2 * self.id_mat(sasa.shape[0]), lower=False)
        self.factorization = cholesky_factorization_wrapper(upper_mat=upper_mat)

    def solve_approx_newton_system(self, z):
        sketch_size = self.sa.shape[0]

        if z.ndim == 1:
            z = z.reshape((-1, 1))

        if sketch_size > self.d:
            return self.factorization(z)
        else:
            ztmp_ = self.sa.T @ self.factorization(self.sa @ z)
            return 1. / self.sasa_nu ** 2 * (z - ztmp_)
    
    def _init_cg(self):
        x = self._zeros(self.d, self.c)
        r, p = self._copy(self.y), self._copy(self.y)
        return x, r, p
    
    def _cg_iteration(self, x, r, p):
        hp = self.hv_product(p)
        alpha = (r ** 2).sum(axis=0) / (p * hp).sum(axis=0)
        x += alpha * p
        r_ = r - alpha * hp
        beta = (r_ ** 2).sum(axis=0) / (r ** 2).sum(axis=0)
        r = self._copy(r_)
        p = r + beta * p
        return x, r, p

    def _init_pcg(self, x, sa, sasa, factorization=None, sasa_nu=None):
        if factorization is None:
            self.factor_approx_hessian(sa=sa, sasa=sasa)
        else:
            self.sa = sa
            self.sasa_nu = sasa_nu
            self.factorization = factorization
        r = self.y - self.hv_product(x)
        rtilde = self.solve_approx_newton_system(z=r)
        delta = (r * rtilde).sum(axis=0)
        p = self._copy(rtilde)
        return r, p, rtilde, delta
    
    def _pcg_iteration(self, x, r, p, delta):
        hp = self.hv_product(p)
        alpha = delta / (p * hp).sum(axis=0)
        x_new = x + alpha * p
        r_new = r - alpha * hp
        rtilde_new = self.solve_approx_newton_system(z=r_new)
        delta_new = (r_new * rtilde_new).sum(axis=0)
        p_new = rtilde_new + (delta_new / delta) * p
        return x_new, r_new, p_new, rtilde_new, delta_new