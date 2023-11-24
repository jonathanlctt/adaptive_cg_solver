import time
from adacg_solver.solvers.direct_method import DirectMethod
from adacg_solver.solvers.conjugate_gradient import CG
from adacg_solver.solvers.preconditioned_conjugate_gradient import PCG
from adacg_solver.solvers.adaptive_conjugate_gradient import AdaptiveCG, init_multi_worker
from adacg_solver.sketching.multiworker_sketcher import SketchLoader
import scipy

import torch
import numpy as np


def run_test(a, b, reg_param, num_workers=1, enforce_cuda=False):

    n, d = a.shape
    pcg_sketch_size = min(n // 2, 4*d)
    ada_sketch_size = 32
    max_sketch_size = min(n // 2, 4*d)
    n_iterations_pcg = 200
    n_iterations_cg = 200
    n_iterations_adacg = 50
    tolerance = 1e-10

    dm = DirectMethod(a=a, b=b, reg_param=reg_param, enforce_cuda=enforce_cuda)
    dm.fit()
    x_opt = dm.x_fit

    cg = CG(a, b, reg_param, x_opt=x_opt, enforce_cuda=enforce_cuda)
    cg.fit(n_iterations=n_iterations_cg,
           tolerance=tolerance,
           get_full_metrics=False)

    if isinstance(x_opt, torch.Tensor):
        err_ = torch.max(torch.abs(cg.x_fit - x_opt))
    else:
        err_ = np.max(np.abs(cg.x_fit - x_opt))
    if err_ < 1e-4:
        print(f"OK solution for CG, {enforce_cuda=}, {num_workers=}, {err_=}")
    else:
        print(f"Wrong solution for CG, {enforce_cuda=}, {num_workers=}, {err_=}")

    for sketch_fn in ['gaussian', 'srht', 'sjlt']:
        pcg = PCG(a, b, reg_param, x_opt=x_opt, enforce_cuda=enforce_cuda)
        pcg.fit(n_iterations=n_iterations_pcg,
                tolerance=tolerance,
                sketch_size=pcg_sketch_size,
                sketch_fn=sketch_fn,
                get_full_metrics=False)
        if isinstance(x_opt, torch.Tensor):
            err_ = torch.max(torch.abs(pcg.x_fit - x_opt))
        else:
            err_ = np.max(np.abs(pcg.x_fit - x_opt))
        if err_ < 1e-4:
            print(f"OK solution for PCG, {enforce_cuda=}, {sketch_fn=}, {num_workers=}, {err_=}")
        else:
            print(f"Wrong solution for PCG, {enforce_cuda=}, {sketch_fn=}, {num_workers=}, {err_=}")

    for sketch_fn in ['gaussian', 'srht', 'sjlt']:
        num_workers_ = 1 if sketch_fn == 'srht' else num_workers
        ada_solver = AdaptiveCG(a, b, reg_param, x_opt=x_opt, enforce_cuda=enforce_cuda)
        sketch_loader = SketchLoader(num_workers=num_workers_)

        while True:
            if sketch_loader.handshake_done:
                break
            else:
                pass

        ada_solver.fit(
            sketch_loader=sketch_loader,
            sketch_size=ada_sketch_size,
            max_sketch_size=max_sketch_size,
            sketch_fn=sketch_fn,
            tolerance=tolerance,
            n_iterations_cg=n_iterations_cg,
            n_iterations=n_iterations_adacg,
            get_full_metrics=True,
        )

        if isinstance(x_opt, torch.Tensor):
            err_ = torch.max(torch.abs(ada_solver.x_fit - x_opt))
        else:
            err_ = np.max(np.abs(ada_solver.x_fit - x_opt))
        if err_ < 1e-4:
            print(f"OK solution for AdaCG, {enforce_cuda=}, {sketch_fn=}, {num_workers=}, {err_=}")
        else:
            print(f"Wrong solution for AdaCG, {enforce_cuda=}, {sketch_fn=}, {num_workers=}, {err_=}")


def test_torch(num_workers=1, dtype=torch.float32, enforce_cuda=False):
    n = 8192
    d = 2048
    a = torch.randn(n, d, dtype=dtype) / np.sqrt(n)
    b = torch.randn(n, 1, dtype=dtype) / np.sqrt(n)
    reg_param = 1e-1

    run_test(a, b, reg_param, num_workers=num_workers, enforce_cuda=enforce_cuda)


def test_scipy_sparse(sptype='csc', num_workers=1):
    n = 1024
    d = 512
    row_ids = np.random.choice(n, d)
    col_ids = np.arange(d, dtype=np.int32)
    data = np.random.randn(d, ) / np.sqrt(d)

    if sptype == 'csc':
        a = scipy.sparse.csc_matrix((data, (row_ids, col_ids)), shape=(n, d))
    elif sptype == 'csr':
        a = scipy.sparse.csr_matrix((data, (row_ids, col_ids)), shape=(n, d))

    b = np.random.randn(n, 1) / np.sqrt(n)
    reg_param = 1e-1

    run_test(a, b, reg_param, num_workers=num_workers, enforce_cuda=False)


def test_numpy(num_workers=1, enforce_cuda=False):
    n = 8000
    d = 5000

    a = np.random.randn(n, d)
    u, _, vh = np.linalg.svd(a, full_matrices=False)
    sigma = np.array([max(0.95**jj, 1e-3) for jj in range(d)]).reshape((-1, 1))
    a = u @ (sigma * vh)
    b = np.random.randn(n, 1) / np.sqrt(n)
    reg_param = 1e-3

    run_test(a, b, reg_param, num_workers=num_workers, enforce_cuda=enforce_cuda)


if __name__ == '__main__':
    #test_numpy(num_workers=1, enforce_cuda=False)
    test_numpy(num_workers=1, enforce_cuda=True)
    test_numpy(num_workers=4, enforce_cuda=True)
    #test_torch(num_workers=1, dtype=torch.float64, enforce_cuda=True)
    #test_torch(num_workers=1, dtype=torch.float32, enforce_cuda=False)
    #test_torch(num_workers=4, dtype=torch.float32, enforce_cuda=True)
    #test_torch(num_workers=4, dtype=torch.float32)
    #test_scipy_sparse(sptype='csc', num_workers=1)
    #test_scipy_sparse(sptype='csc', num_workers=4)
    #test_scipy_sparse(sptype='csr', num_workers=1)
    #test_scipy_sparse(sptype='csr', num_workers=4)

    print(f"SUCCESS!")

    