from adacg_solver.solvers.direct_method import DirectMethod
from adacg_solver.solvers.conjugate_gradient import CG
from adacg_solver.solvers.preconditioned_conjugate_gradient import PCG
from adacg_solver.solvers.adaptive_conjugate_gradient import AdaptiveCG
from adacg_solver.sketching.multiworker_sketcher import SketchLoader
import scipy

import torch
import numpy as np


def run_test(a, b, reg_param, data_type='torch', num_workers=1):

    n, d = a.shape
    pcg_sketch_size = min(n // 2, 4*d)
    ada_sketch_size = 32
    max_sketch_size = min(n // 2, 4*d)
    n_iterations_pcg = 20
    n_iterations_cg = 20
    n_iterations_adacg = 20
    n_iterations_warm_up = 3
    tolerance = 1e-10
    tolerance_warm_up = 1e-4

    dm = DirectMethod(a=a, b=b, reg_param=reg_param)
    dm.fit()
    x_opt = dm.x_fit

    cg = CG(a, b, reg_param, x_opt=x_opt)
    cg.fit(n_iterations=n_iterations_cg,
           tolerance=tolerance,
           get_full_metrics=False)

    assert np.allclose(cg.x_fit, x_opt, atol=1e-4), 'torch: wrong solution for CG'

    for sketch_fn in ['gaussian', 'srht', 'sjlt']:
        pcg = PCG(a, b, reg_param, x_opt=x_opt)
        pcg.fit(n_iterations=n_iterations_pcg,
                tolerance=tolerance,
                sketch_size=pcg_sketch_size,
                sketch_fn=sketch_fn,
                get_full_metrics=False)
        assert np.allclose(pcg.x_fit, x_opt, atol=1e-4), f'torch: wrong solution for PCG and {sketch_fn=}'

    for sketch_fn in ['gaussian', 'srht', 'sjlt']:
        num_workers_ = 1 if sketch_fn == 'srht' else num_workers
        ada_solver = AdaptiveCG(a, b, reg_param, x_opt=x_opt, rescale_data=False)
        sketch_loader = SketchLoader(num_workers=num_workers_, with_torch=True if data_type == 'torch' else False)

        ada_solver.fit(
            sketch_loader=sketch_loader,
            sketch_size=ada_sketch_size,
            max_sketch_size=max_sketch_size,
            sketch_fn=sketch_fn,
            tolerance=tolerance,
            tolerance_warm_up=tolerance_warm_up,
            n_iterations_cg=n_iterations_warm_up,
            n_iterations=n_iterations_adacg,
            get_full_metrics=False,
        )

        assert np.allclose(ada_solver.x_fit, x_opt, atol=1e-4), f'{data_type}: wrong solution for AdaCG, {sketch_fn=}, {num_workers=}'


def test_torch(num_workers=1, dtype=torch.float32):
    n = 1024
    d = 128
    a = torch.randn(n, d, dtype=dtype) / np.sqrt(n)
    b = torch.randn(n, 1, dtype=dtype) / np.sqrt(n)
    reg_param = 1e-1

    run_test(a, b, reg_param, data_type='torch', num_workers=num_workers)


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

    run_test(a, b, reg_param, data_type='scipy_sparse', num_workers=num_workers)


def test_numpy(num_workers=1):
    n = 1024
    d = 128

    a = np.random.randn(n, d)
    b = np.random.randn(n, 1)
    reg_param = 1e-3

    run_test(a, b, reg_param, data_type='scipy_sparse', num_workers=num_workers)


if __name__ == '__main__':
    test_numpy(num_workers=1)
    test_numpy(num_workers=4)
    test_torch(num_workers=1, dtype=torch.float64)
    test_torch(num_workers=1, dtype=torch.float32)
    test_torch(num_workers=4, dtype=torch.float64)
    test_torch(num_workers=4, dtype=torch.float32)
    test_scipy_sparse(sptype='csc', num_workers=1)
    test_scipy_sparse(sptype='csc', num_workers=4)
    test_scipy_sparse(sptype='csr', num_workers=1)
    test_scipy_sparse(sptype='csr', num_workers=4)

    print(f"SUCCESS!")

    