from timeit import default_timer as time

from .quadratic import QuadraticSolver

from scipy.sparse import identity as sparse_identity_matrix
from ..linear_algebra.linear_algebra import cholesky, solve_triangular
from scipy.sparse.linalg import factorized as sparse_factorization

import logging
logging.basicConfig(level=logging.INFO)


class DirectMethod(QuadraticSolver):
    def __init__(self, a, b, reg_param, x_opt=None, rescale_data=True, check_reg_param=True, least_squares=True,
                 enforce_cuda=False):
        QuadraticSolver.__init__(self, a, b, reg_param, x_opt=x_opt, rescale_data=rescale_data,
                                 check_reg_param=check_reg_param, least_squares=least_squares, enforce_cuda=enforce_cuda)

    def fit(self):

        start = time()

        self.put_data_on_device(non_blocking=False)

        if not self.is_sparse:
            logging.info(f"direct method using cholesky decomposition")
            h = self.X.T @ self.X + self.nu ** 2 * self.id_mat(self.d)
            upper_ = cholesky(h, lower=False)
            x_opt = solve_triangular(upper_, solve_triangular(upper_.T, self.y, lower=True), lower=False)
        else:
            logging.info(f"direct method using sparse LU decomposition")
            h = self.X.T @ self.X + self.nu ** 2 * sparse_identity_matrix(self.d, dtype=self._dtype)
            factorization = sparse_factorization(h.tocsc())
            x_opt = factorization(self.y)

        t_ = time() - start

        self.x_fit = x_opt
        self.metrics = {'fit_time': t_}



