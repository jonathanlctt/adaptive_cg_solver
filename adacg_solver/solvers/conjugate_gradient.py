from .quadratic import QuadraticSolver
from timeit import default_timer as time
import numpy as np
import queue
import logging
logging.basicConfig(level=logging.INFO)


class CG(QuadraticSolver):

    def __init__(self, a, b, reg_param, x_opt=None, rescale_data=True, check_reg_param=True, least_squares=True,
                 enforce_cuda=False):
        QuadraticSolver.__init__(self, a=a, b=b, reg_param=reg_param, x_opt=x_opt,
                                 rescale_data=rescale_data, check_reg_param=check_reg_param, least_squares=least_squares,
                                 enforce_cuda=enforce_cuda)

    def fit(self, n_iterations, tolerance=1e-10, get_full_metrics=False, break_event=None):
        if get_full_metrics:
            x, mean_err, exit_code, errors, residuals, times, iterations_array = self.fit_(n_iterations=n_iterations,
                                                                                           tolerance=tolerance,
                                                                                           get_full_metrics=True,
                                                                                           break_event=break_event)
            self.x_fit = x
            self.metrics = {
                'average_gradient_norms': mean_err,
                'exit_code': exit_code,
                'errors': errors,
                'gradient_norms': residuals,
                'times': np.cumsum(times),
                'iterations_array': iterations_array,
            }
        else:
            x, mean_err, exit_code = self.fit_(n_iterations=n_iterations,
                                               tolerance=tolerance,
                                               get_full_metrics=False,
                                               break_event=break_event)
            self.x_fit = x
            self.metrics = {
                'average_gradient_norms': mean_err,
                'exit_code': exit_code,
            }

    def fit_(self, n_iterations, tolerance=1e-10, get_full_metrics=False, break_event=None):

        if get_full_metrics:
            logging.info(f"conjugate gradient method: {n_iterations=}, {tolerance=}")
            start = time()

        self.put_data_on_device(non_blocking=False)

        x, r, p = self._init_cg()
        res_init_ = (r * r).sum(axis=0)
        iteration = 0

        mean_err = 1.
        if get_full_metrics:
            times = [time() - start]
            errors = [self.compute_error(x)]
            residuals = [mean_err]
            iterations_array = [iteration]

        while iteration < n_iterations:

            if get_full_metrics:
                start = time()
            x, r, p = self._cg_iteration(x, r, p)
            res_ = (r * r).sum(axis=0)
            err_ = res_ / res_init_
            exit_condition = (err_ <= tolerance).all()  #self._all([b_ <= tolerance for b_ in err_])
            iteration += 1

            mean_err = err_.mean().item()
            if get_full_metrics:
                times.append(time() - start)
                errors.append(self.compute_error(x))
                residuals.append(mean_err)
                iterations_array.append(iteration)

            if exit_condition:
                if get_full_metrics:
                    logging.info(f"success: CG gradient norms <= tolerance")
                    return x, mean_err, 0, np.array(errors), np.array(residuals), np.array(times), np.array(iterations_array)
                else:
                    return x, mean_err, 0

            if break_event is not None:
                if break_event.is_set():
                    if get_full_metrics:
                        logging.info(f"termination: break message received - {tolerance=}, {mean_err=}")
                        return x, mean_err, 1, np.array(errors), np.array(residuals), np.array(times), np.array(iterations_array)
                    else:
                        return x, mean_err, 1

        if get_full_metrics:
            logging.info(f"termination: maximum number of CG iterations - {tolerance=}, {mean_err=}")
            return x, mean_err, 1, np.array(errors), np.array(residuals), np.array(times), np.array(iterations_array)
        else:
            return x, mean_err, 1
