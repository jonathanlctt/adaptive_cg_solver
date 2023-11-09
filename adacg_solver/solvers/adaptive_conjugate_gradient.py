from timeit import default_timer as time
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np

from .quadratic import QuadraticSolver
from ..sketching.multiworker_sketcher import SketchLoader


def init_multi_worker(num_workers, with_torch):
    start = time()

    sketch_loader = SketchLoader(num_workers=num_workers, with_torch=with_torch)
    while True:
        if sketch_loader.handshake_done:
            break
        else:
            pass

    return sketch_loader, time() - start


class AdaptiveCG(QuadraticSolver):

    def __init__(self, A, b, reg_param, x_opt=None, rescale_data=True, check_reg_param=True, least_squares=True):

        QuadraticSolver.__init__(self, A, b, reg_param, x_opt=x_opt, rescale_data=rescale_data,
                                 check_reg_param=check_reg_param, least_squares=least_squares)

    def fit(self,
              sketch_loader,
              sketch_size=16,
              max_sketch_size=32,
              sketch_fn='gaussian',
              tolerance=1e-10,
              tolerance_warm_up=1e-5,
              n_iterations_cg=20,
              n_iterations=100,
              get_full_metrics=False
              ):

        if get_full_metrics:
            x, m_err, e_code, errs_, res_, ts_, iters_, ssizes_, t_send_data = self.fit_(
              sketch_loader=sketch_loader,
              sketch_size=sketch_size,
              max_sketch_size=max_sketch_size,
              sketch_fn=sketch_fn,
              tolerance=tolerance,
              tolerance_warm_up=tolerance_warm_up,
              n_iterations_cg=n_iterations_cg,
              n_iterations=n_iterations,
              get_full_metrics=True
              )

            self.x_fit = x
            self.metrics = {
                'average_gradient_norms': m_err,
                'exit_code': e_code,
                'errors': errs_,
                'gradient_norms': res_,
                'times': np.cumsum(ts_),
                'iterations_array': iters_,
                'sketch_sizes': ssizes_,
                'time_to_send_data': t_send_data,
            }

        else:
            x, m_err, e_code = self.fit_(
              sketch_loader=sketch_loader,
              sketch_size=sketch_size,
              max_sketch_size=max_sketch_size,
              sketch_fn=sketch_fn,
              tolerance=tolerance,
              tolerance_warm_up=tolerance_warm_up,
              n_iterations_cg=n_iterations_cg,
              n_iterations=n_iterations,
              get_full_metrics=False
              )

            self.x_fit = x
            self.metrics = {
                'average_gradient_norms': m_err,
                'exit_code': e_code,
            }


    def fit_(self,
              sketch_loader,
              sketch_size=16,
              max_sketch_size=32,
              sketch_fn='gaussian',
              tolerance=1e-10,
              tolerance_warm_up=1e-5,
              n_iterations_cg=20,
              n_iterations=100,
              get_full_metrics=False
              ):

        if get_full_metrics:
            logging.info(
                f"Adaptive CG: {sketch_fn=}, {sketch_size=}, {max_sketch_size=}, {tolerance=}, {tolerance_warm_up=}, {n_iterations_cg=}, {n_iterations=}")
            start = time()

        self.sketch_loader = sketch_loader
        self.sketch_loader.empty_queues()

        if sketch_fn in ['srht', 'naive_srht']:
            assert self.sketch_loader.num_workers == 1, "srht cannot be parallelized across workers"

        self.sketch_loader.send_data_to_workers(a=self.X,
                                                b=self.y,
                                                sketch_fn=sketch_fn,
                                                sketch_size=sketch_size,
                                                max_sketch_size=max_sketch_size,
                                                reg_param=self.nu)

        if get_full_metrics:
            time_to_send_data = time() - start

        self.sketch_loader.prefetch()

        cg_iteration = 0
        x, r, p = self._init_cg()
        res_init_ = self._sum(r * r, axis=0)
        working_on_sketch = True

        mean_err = 1.
        _total_iterations = 0
        if get_full_metrics:
            times = [time() - start]
            errors = [self.compute_error(x)]
            sketch_sizes = [0]
            residuals = [mean_err]
            iterations_array = [_total_iterations]

        while working_on_sketch:
            if get_full_metrics:
                start = time()

            x, r, p = self._cg_iteration(x, r, p)
            res_ = self._sum(r * r, axis=0)
            err_ = res_ / res_init_
            working_on_sketch = not self.sketch_loader.sketch_done()
            exit_condition = self._all([b_ <= tolerance for b_ in err_])
            cg_iteration += 1
            _total_iterations += 1

            mean_err = self._mean(err_)
            if get_full_metrics:
                times.append(time() - start)
                errors.append(self.compute_error(x))
                residuals.append(mean_err)
                sketch_sizes.append(0)
                iterations_array.append(_total_iterations)

            if exit_condition:
                if get_full_metrics:
                    logging.info(f"success: CG warm-up gradient norms <= warm-up tolerance")
                    return x, mean_err, 0, np.array(errors), np.array(residuals), np.array(times), np.array(iterations_array), np.array(sketch_sizes), time_to_send_data
                else:
                    return x, mean_err, 0

            x_opt = self.sketch_loader.get_dm()
            if x_opt is not None:
                if get_full_metrics:
                    logging.info(f"success: DM solver found a solution - v1")
                    errors[-1] = max(self.compute_error(x_opt), tolerance)
                    residuals[-1] = tolerance
                    return x_opt, tolerance, 0, np.array(errors), np.array(residuals), np.array(times), np.array(iterations_array), np.array(sketch_sizes), time_to_send_data
                else:
                    return x_opt, tolerance, 0

        cg_warm_up_valid = self._all([b_ <= tolerance_warm_up for b_ in err_])
        if cg_warm_up_valid:
            while cg_iteration < n_iterations_cg:

                if get_full_metrics:
                    start = time()

                x, r, p = self._cg_iteration(x, r, p)
                res_ = self._sum(r * r, axis=0)
                err_ = res_ / res_init_
                exit_condition = self._all([b_ <= tolerance for b_ in err_])
                cg_iteration += 1
                _total_iterations += 1

                mean_err = self._mean(err_)
                if get_full_metrics:
                    times.append(time() - start)
                    residuals.append(mean_err)
                    errors.append(self.compute_error(x))
                    sketch_sizes.append(0)
                    iterations_array.append(_total_iterations)

                if exit_condition:
                    if get_full_metrics:
                        logging.info(f"STOPPING: CG warm-up gradient norms <= tolerance")
                        return x, mean_err, 0, np.array(errors), np.array(residuals), np.array(times), np.array(iterations_array), np.array(sketch_sizes), time_to_send_data
                    else:
                        return x, mean_err, 0

                x_opt = self.sketch_loader.get_dm()
                if x_opt is not None:
                    if get_full_metrics:
                        logging.info(f"success: DM solver found a solution - v2")
                        errors[-1] = max(self.compute_error(x_opt), tolerance)
                        residuals[-1] = tolerance
                        return x_opt, tolerance, 0, np.array(errors), np.array(residuals), np.array(times), np.array(
                            iterations_array), np.array(sketch_sizes), time_to_send_data
                    else:
                        return x_opt, tolerance, 0

            if get_full_metrics:
                logging.info(f"termination: maximum number of iterations - {tolerance=}, {mean_err=}")
                return x, mean_err, 0, np.array(errors), np.array(residuals), np.array(times), np.array(iterations_array), np.array(sketch_sizes), time_to_send_data
            else:
                return x, mean_err, 0

        logging.info(f"invalid CG warm-up: switching to AdaptiveCG")

        xcg, rcg, pcg = self._copy(x), self._copy(r), self._copy(p)
        iteration = 0
        tmp_errors, tmp_errors_cg, tmp_residuals, tmp_residuals_cg = [], [], [], []
        while iteration < n_iterations:

            if get_full_metrics:
                start = time()

            sa, factorization, sasa_nu = self.sketch_loader.get()
            if sasa_nu > self.nu:
                logging.warning(f"Ada-CG INNER SOLVER: Numerical issue for dtype={sa.dtype} - regularization parameter too small - now {sasa_nu}, before {self.nu}")
            current_sketch_size = sa.shape[0]
            logging.info(f"sketch size increased to {current_sketch_size=}")

            r, p, rtilde, delta = self._init_pcg(x=self._copy(x), sa=sa, sasa=None, factorization=factorization, sasa_nu=sasa_nu)
            res_ = self._sum(r * r, axis=0)
            err_ = res_ / res_init_

            xcg, rcg, pcg = self._cg_iteration(xcg, rcg, pcg)
            res_cg = self._sum(rcg * rcg, axis=0)
            err_cg = res_cg / res_init_

            mean_err = self._mean(err_)
            mean_err_cg = self._mean(err_cg)
            if get_full_metrics:
                times.append(time() - start)
                tmp_residuals.append(mean_err)
                tmp_residuals_cg.append(mean_err_cg)
                tmp_errors.append(self.compute_error(x))
                tmp_errors_cg.append(self.compute_error(xcg))
                sketch_sizes.append(current_sketch_size)
                iterations_array.append(_total_iterations)

            working_on_sketch = True

            while (working_on_sketch or current_sketch_size == max_sketch_size) and iteration < n_iterations:

                if get_full_metrics:
                    start = time()

                x, r, p, rtilde, delta = self._pcg_iteration(x=x, r=r, p=p, delta=delta)
                res_ = self._sum(r * r, axis=0)
                err_ = res_ / res_init_
                exit_condition = self._all([b_ <= tolerance for b_ in err_])

                xcg, rcg, pcg = self._cg_iteration(xcg, rcg, pcg)
                res_cg = self._sum(rcg * rcg, axis=0)
                err_cg = res_cg / res_init_
                exit_condition_cg = self._all([b_ <= tolerance for b_ in err_cg])

                iteration += 1
                _total_iterations += 1

                mean_err = self._mean(err_)
                mean_err_cg = self._mean(err_cg)
                if get_full_metrics:
                    times.append(time() - start)
                    tmp_residuals.append(mean_err)
                    tmp_residuals_cg.append(mean_err_cg)
                    tmp_errors.append(self.compute_error(x))
                    tmp_errors_cg.append(self.compute_error(xcg))
                    sketch_sizes.append(current_sketch_size)
                    iterations_array.append(_total_iterations)

                if exit_condition:
                    if get_full_metrics:
                        logging.info(f"success: AdaptiveCG gradient norms <= tolerance")
                        return x, mean_err, 0, np.array(errors+tmp_errors), np.array(residuals+tmp_residuals), np.array(times), np.array(iterations_array), np.array(sketch_sizes), time_to_send_data
                    else:
                        return x, mean_err, 0

                if exit_condition_cg:
                    if get_full_metrics:
                        logging.info(f"success: AdaptiveCG gradient norms <= tolerance")
                        return xcg, mean_err, 0, np.array(errors+tmp_errors_cg), np.array(residuals+tmp_residuals_cg), np.array(times), np.array(iterations_array), np.array(sketch_sizes), time_to_send_data
                    else:
                        return xcg, mean_err, 0

                if self.sketch_loader is not None:
                    working_on_sketch = not self.sketch_loader.sketch_done()

                x_opt = self.sketch_loader.get_dm()
                if x_opt is not None:
                    if get_full_metrics:
                        logging.info(f"success: DM solver found a solution - v3")
                        if mean_err <= mean_err_cg:
                            tmp_errors[-1] = max(self.compute_error(x_opt), tolerance)
                            tmp_residuals[-1] = tolerance
                            return x_opt, tolerance, 0, np.array(errors + tmp_errors), np.array(residuals + tmp_residuals), np.array(
                                times), np.array(iterations_array), np.array(sketch_sizes), time_to_send_data
                        else:
                            tmp_errors_cg[-1] = max(self.compute_error(x_opt), tolerance)
                            tmp_residuals_cg[-1] = tolerance
                            return x_opt, tolerance, 0, np.array(errors + tmp_errors_cg), np.array(residuals + tmp_residuals_cg), np.array(
                                times), np.array(iterations_array), np.array(sketch_sizes), time_to_send_data
                    else:
                        return x_opt, tolerance, 0

        if get_full_metrics:
            logging.info(f"termination: maximum number of iterations - {tolerance=}, {mean_err=}")
            if mean_err <= mean_err_cg:
                return x, mean_err, 1, np.array(errors+tmp_errors), np.array(residuals+tmp_residuals), np.array(times), np.array(iterations_array), np.array(sketch_sizes), time_to_send_data
            else:
                return xcg, mean_err_cg, 1, np.array(errors+tmp_errors_cg), np.array(residuals+tmp_residuals_cg), np.array(times), np.array(iterations_array), np.array(sketch_sizes), time_to_send_data
        else:
            if mean_err <= mean_err_cg:
                return x, mean_err, 1
            else:
                return xcg, mean_err_cg, 1


