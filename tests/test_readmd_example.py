import torch
import numpy as np

from adacg_solver.solvers.direct_method import DirectMethod
from adacg_solver.solvers.conjugate_gradient import CG
from adacg_solver.solvers.preconditioned_conjugate_gradient import PCG

from adacg_solver.solvers.adaptive_conjugate_gradient import AdaptiveCG
from adacg_solver.sketching.multiworker_sketcher import SketchLoader


def readme_fn(a, b, reg_param):

    # solve min_x |Ax - b|_2^2 + reg_param**2 * |x|_2^2
    # a: torch.Tensor or np.ndarray or scipy.sparse array
    # b: a np.ndarray of size (a.shape[0], num_targets)

    n, d = a.shape

    params = {
        'rescale_data': True,
        # rescale data by (quickly approximated) max singular value of a - useful for numerical stability checks
        'check_reg_param': True,
        # check if regularization parameter is above numerically significant threshold - if not, set reg_param to threshold
        'least_squares': True,  # if False, solve problem min_x |Ax|_2^2 - 2 * b' * x + reg_param**2 * |x|^2_2
        'x_opt': None  # if not None, will compute errors relative to x_opt (useful for testing)
    }

    # Sketch S x A is computed in background while iterative solver is running
    sketch_loader = SketchLoader(num_workers=1)

    # WARNING: the above __init__ of MultiWorkerSketcher spawns sub-processes; that introduces an overhead (~2 seconds on a 6 cores / 12 vCPUs MacOS machine).
    # For measuring performance and excluding this overhead, we recommend running the solver once the sub-processes are initialized.
    while not sketch_loader.handshake_done:
        continue

    # initialize adaptive solver
    ada_solver = AdaptiveCG(a, b, reg_param, **params)

    # Adaptive solver starts with conjugate gradient.
    # As soon as sketch is available, the preconditioner is updated and used by the preconditioned CG solver.
    # The sketch_loader computes the next sketch with sketch_size <- min(max_sketch_size, 2 * sketch_size) in the background.

    adacg_fit_params = {
        'sketch_loader': sketch_loader,
        'sketch_size': 32,  # initial sketch size for first sketch
        'max_sketch_size': min(n // 2, 4 * d),  # max sketch size = recommended PCG sketch size
        'sketch_fn': 'sjlt',  # sketch matrix to use; other choices = 'gaussian', 'srht',
        'tolerance': 1e-10,  # exit if squared gradient norms <= tolerance
        'n_iterations': 100,  # max number of iterations of iterative solver
        'n_iterations_cg': 100,  # max number of iterations of conjugate gradient
        'get_full_metrics': False,  # if True, returns additional performance metrics
    }

    ada_solver.fit(**adacg_fit_params)

    x_adacg = ada_solver.x_fit
    metrics_adacg = ada_solver.metrics

    direct_method = DirectMethod(a, b, reg_param, **params)

    # direct method based on Cholesky factorization
    direct_method.fit()

    x_opt = direct_method.x_fit
    metrics_dm = direct_method.metrics

    # conjugate gradient method
    cg = CG(a, b, reg_param, **params)

    cg_fit_params = {
        'n_iterations': 100,  # max number of iterations of iterative solver
        'tolerance': 1e-10,  # exit if squared gradient norms <= tolerance
        'get_full_metrics': False,  # if True, return additional performance metrics
    }

    cg.fit(**cg_fit_params)

    x_cg = cg.x_fit
    metrics_cg = cg.metrics

    # preconditioned conjugate gradient method
    pcg = PCG(a, b, reg_param, **params)

    pcg_fit_params = {
        'sketch_size': min(n // 2, 4 * d),  # sketch size = recommended PCG sketch size
        'sketch_fn': 'sjlt',  # sketch matrix to use; other choices = 'gaussian', 'srht'
        'n_iterations': 100,  # max number of iterations of iterative solver
        'tolerance': 1e-10,  # exit if squared gradient norms <= tolerance
        'get_full_metrics': False  # if True, return additional performance metrics
    }

    pcg.fit(**pcg_fit_params)

    x_pcg = pcg.x_fit
    metrics_pcg = pcg.metrics

    print(f"SUCCESS!")


if __name__ == '__main__':

    n, d = 2048, 128
    reg_param = 1e-1

    a = torch.randn(n, d, dtype=torch.float64) / np.sqrt(n)
    b = torch.randn(n, 1, dtype=torch.float64) / np.sqrt(n)
    readme_fn(a, b, reg_param)

    # device = torch.device(type='cuda', index=1)
    # a = torch.randn(n, d, device=device, dtype=torch.float64) / np.sqrt(n)
    # b = torch.randn(n, 1, device=device, dtype=torch.float64) / np.sqrt(n)
    # readme_fn(a, b, reg_param)

    a = torch.randn(n, d, dtype=torch.float32) / np.sqrt(n)
    b = torch.randn(n, 1, dtype=torch.float32) / np.sqrt(n)
    readme_fn(a, b, reg_param)

    a = np.random.randn(n, d).astype(np.float64) / np.sqrt(n)
    b = np.random.randn(n, 1).astype(np.float64) / np.sqrt(n)
    readme_fn(a, b, reg_param)
