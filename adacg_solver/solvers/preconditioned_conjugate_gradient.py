import torch
import numpy as np

import logging
from timeit import default_timer as time

from scipy.sparse import issparse

from ..sketching.sketches import hadamard_matrix, sjlt
from ..sketching.srht import srht_sketch
from .quadratic import QuadraticSolver


class PCG(QuadraticSolver):

    def __init__(self, a, b, reg_param, x_opt=None, rescale_data=True, check_reg_param=True, least_squares=True,
                 enforce_cuda=False):

        QuadraticSolver.__init__(self, a=a, b=b, reg_param=reg_param, x_opt=x_opt, rescale_data=rescale_data,
                                 check_reg_param=check_reg_param, least_squares=least_squares,
                                 enforce_cuda=enforce_cuda)

    def get_sketch(self, sketch_fn, sketch_size):

        with_torch = isinstance(self.X, torch.Tensor)

        if sketch_fn == 'srht':
            sa = srht_sketch(self.X, sketch_size, with_stack=True)
        elif sketch_fn == 'naive_srht':
            s = hadamard_matrix(n=self.n, sketch_size=sketch_size, with_torch=with_torch, device=self.device)
            sa = s @ self.X
        elif sketch_fn == 'gaussian':
            if with_torch:
                s = torch.randn(sketch_size, self.n, device=self.device, dtype=self.X.dtype)
            else:
                s = np.random.randn(sketch_size, self.n).astype(self.X.dtype)
            sa = s @ self.X / np.sqrt(sketch_size)
        elif sketch_fn == 'sjlt':
            sa = sjlt(a=self.X, sketch_size=sketch_size, nnz_per_column=1)
        else:
            raise NotImplementedError

        if issparse(sa):
            sa = sa.toarray()

        if sa.shape[0] <= sa.shape[1]:
            sasa = sa @ sa.T
        else:
            sasa = sa.T @ sa

        return sa, sasa

    def fit(self,
            sketch_size=16,
            sketch_fn='gaussian',
            tolerance=1e-10,
            n_iterations=10,
            get_full_metrics=False,
            ):

        if get_full_metrics:
            x, m_err, e_code, errs, res_, ts_, iters_, ssizes_ = self.fit_(sketch_size=sketch_size,
                                                                           sketch_fn=sketch_fn,
                                                                           tolerance=tolerance,
                                                                           n_iterations=n_iterations,
                                                                           get_full_metrics=True)

            self.x_fit = x
            self.metrics = {
                'average_gradient_norms': m_err,
                'exit_code': e_code,
                'errors': errs,
                'gradient_norms': res_,
                'times': np.cumsum(ts_),
                'iterations_array': iters_,
                'sketch_sizes': ssizes_,
            }
        else:
            x, m_err, e_code = self.fit_(sketch_size=sketch_size,
                                         sketch_fn=sketch_fn,
                                         tolerance=tolerance,
                                         n_iterations=n_iterations,
                                         get_full_metrics=False)

            self.x_fit = x
            self.metrics = {
                'average_gradient_norms': m_err,
                'exit_code': e_code,
            }


    def fit_(self,
              sketch_size=16,
              sketch_fn='gaussian',
              tolerance=1e-10,
              n_iterations=10,
              get_full_metrics=False,
              ):

        if get_full_metrics:
            logging.info(f"preconditioned conjugate gradient method: {sketch_fn=}, {sketch_size=}, {tolerance=}, {n_iterations=}")
            start = time()

        self.put_data_on_device(non_blocking=False)

        sketch_size = min(self.n, sketch_size)
        sa, sasa = self.get_sketch(sketch_fn, sketch_size)

        x = self._zeros(self.d, self.c)
        r, p, rtilde, delta = self._init_pcg(x=x, sa=sa, sasa=sasa)
        res_init_ = (r * r).sum(axis=0)

        iteration = 0

        mean_err = 1.
        if get_full_metrics:
            times = [time() - start]
            errors = [self.compute_error(x)]
            sketch_sizes = [0]
            residuals = [mean_err]
            iterations_array = [iteration]

        while iteration < n_iterations:

            if get_full_metrics:
                start = time()

            x, r, p, rtilde, delta = self._pcg_iteration(x=x, r=r, p=p, delta=delta)
            res_ = (r * r).sum(axis=0)
            err_ = res_ / res_init_
            exit_condition = (err_ <= tolerance).all()
            iteration += 1

            mean_err = err_.mean().item()
            if get_full_metrics:
                times.append(time() - start)
                errors.append(self.compute_error(x))
                residuals.append(mean_err)
                sketch_sizes.append(sketch_size)
                iterations_array.append(iteration)

            if exit_condition:
                if get_full_metrics:
                    logging.info(f"success: PCG gradient norms <= tolerance")
                    return x, mean_err, 0, np.array(errors), np.array(residuals), np.array(times), np.array(iterations_array), np.array(sketch_sizes)
                else:
                    return x, mean_err, 0

        if get_full_metrics:
            logging.info(f"termination: maximum number of iterations - {tolerance=}, {mean_err=}")
            return x, mean_err, 1, np.array(errors), np.array(residuals), np.array(times), np.array(iterations_array), np.array(sketch_sizes)
        else:
            return x, mean_err, 1
