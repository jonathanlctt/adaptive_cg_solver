import os
import itertools
import torch.multiprocessing as torch_mp
import queue

import numpy as np
import scipy
from scipy.sparse import issparse, isspmatrix_csc, identity as sparse_identity_matrix
from scipy.sparse.linalg import factorized as sparse_factorization
import torch

from adacg_solver.linear_algebra.linear_algebra import get_max_sval_approx, get_reg_param_threshold, solve_triangular
from adacg_solver.solvers.conjugate_gradient import CG
from adacg_solver.sketching.srht import srht_sketch
from adacg_solver.sketching.sketches import sjlt
from adacg_solver.sketching.multiworker_sketcher import SketchLoader
from adacg_solver.solvers.quadratic import QuadraticSolver


def base_test():

    n_children = 1
    sketch_loader = SketchLoader(num_workers=n_children)

    n = 4096
    d = 1024

    device = torch.device(type='cuda')

    a = torch.randn(n, d, device=device) / np.sqrt(n)
    b = torch.randn(n, device=device) / np.sqrt(n)
    y = a.T @ b
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    sketch_fn = 'gaussian'
    sketch_size = 128
    max_sketch_size = 1024
    reg_param = 1e-1
    cg_params = {
        'n_iterations': 20,
        'tolerance': 1e-10,
        'get_full_metrics': True,
        'x_opt': None
    }

    while not sketch_loader.handshake_done:
        pass

    sketch_loader.empty_queues()
    sketch_loader.send_data_to_workers(a, y, sketch_fn, sketch_size, max_sketch_size, reg_param, cg_params, device)

    sketch_loader.prefetch()

    for _ in range(5):
        while not sketch_loader.sketch_done():
            pass

        sa, factorization, sasa_nu = sketch_loader.get()

        print(f"{sa.shape=}, {sa.device=}")

    while True:
        x_opt = sketch_loader.get_dm()
        if x_opt is not None:
            break

    print(f"dm opt {x_opt.shape=}, {x_opt.device=}")

    while True:
        x_opt_cg, metrics_cg = sketch_loader.get_cg()
        if x_opt_cg is not None:
            break
    print(f"cg opt {x_opt_cg.shape=}, {x_opt_cg.device=}")

    del sketch_loader


if __name__ == "__main__":
    base_test()












