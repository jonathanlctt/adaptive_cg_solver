import torch
import numpy as np
from scipy.sparse import issparse

from ..linear_algebra.linear_algebra import cholesky, solve_triangular, get_max_sval_approx, get_reg_param_threshold

import logging
logging.basicConfig(level=logging.INFO)


def cholesky_factorization_wrapper(upper_mat):
    def factorization(z):
        return solve_triangular(upper_mat, solve_triangular(upper_mat.T, z, lower=True), lower=False)

    return factorization


class QuadraticSolver:
    
    def __init__(self, a, b, reg_param, x_opt=None, rescale_data=True, check_reg_param=True, least_squares=True):

        if b.ndim == 1:
            b = b.reshape((-1, 1))

        _dtype = a.dtype

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
            self.Xt = self.X.transpose()
            if least_squares:
                y_ = (self.Xt @ b)
            else:
                y_ = b
            if issparse(y_):
                self.y = y_.A
            else:
                self.y = y_
        else:
            self.Xt = self.X.T
            if least_squares:
                self.y = self.X.T @ b
            else:
                self.y = b

        self.n, self.d = self.X.shape
        self.c = b.shape[1]
    
        if x_opt is not None and x_opt.ndim == 1:
            x_opt = x_opt.reshape((-1, 1))
        self.x_opt = x_opt
    
        self.norm_sq_b = 0.5 * (b.multiply(b)).sum() if self.is_sparse and not isinstance(b, np.ndarray) else 0.5 * (b ** 2).sum()

    def compute_error(self, x):
        if x.ndim == 1:
            x = x.reshape((-1, 1))
    
        if self.x_opt is not None:
            err_ = 0.5 * ((self.X @ (x - self.x_opt)) ** 2).sum()
        else:
            err_ = 0.5 * ((self.X @ x) ** 2).sum() - (self.y * x).sum() + self.norm_sq_b
    
        return err_

    def id_mat(self, d):
        return torch.eye(d, dtype=self._dtype) if self.with_torch else np.eye(d, dtype=self._dtype)
    
    def _sum(self, x, axis=None):
        if self.with_torch:
            return torch.sum(x) if axis is None else torch.sum(x, dim=axis)
        else:
            return np.sum(x) if axis is None else np.sum(x, axis=axis)
    
    def _mean(self, x):
        return torch.mean(x) if self.with_torch else np.mean(x)
    
    def _any(self, x):
        return torch.any(torch.tensor(x)) if self.with_torch else np.any(x)
    
    def _all(self, x):
        return torch.all(torch.tensor(x)) if self.with_torch else np.all(x)
    
    def _copy(self, x):
        return torch.clone(x) if self.with_torch else np.copy(x)
    
    def _zeros(self, n, d):
        return torch.zeros(n, d, dtype=self._dtype) if self.with_torch else np.zeros((n, d), dtype=self._dtype)

    def hv_product(self, v):
        if v.ndim == 1:
            v = v.reshape((-1, 1))
        hv = self.Xt @ (self.X @ v) + self.nu ** 2 * v
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
        alpha = self._sum(r ** 2, axis=0) / self._sum(p * hp, axis=0)
        x += alpha * p
        r_ = r - alpha * hp
        beta = self._sum(r_ ** 2, axis=0) / self._sum(r ** 2, axis=0)
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
        delta = self._sum(r * rtilde, axis=0)
        p = self._copy(rtilde)
        return r, p, rtilde, delta
    
    def _pcg_iteration(self, x, r, p, delta):
        hp = self.hv_product(p)
        alpha = delta / self._sum(p * hp, axis=0)
        x_new = x + alpha * p
        r_new = r - alpha * hp
        rtilde_new = self.solve_approx_newton_system(z=r_new)
        delta_new = self._sum(r_new * rtilde_new, axis=0)
        p_new = rtilde_new + (delta_new / delta) * p
        return x_new, r_new, p_new, rtilde_new, delta_new