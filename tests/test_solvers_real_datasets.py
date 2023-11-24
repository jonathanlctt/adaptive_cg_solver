import torch
import numpy as np
from scipy.sparse import issparse

from adacg_solver.solvers.direct_method import DirectMethod
from adacg_solver.solvers.conjugate_gradient import CG
from adacg_solver.solvers.preconditioned_conjugate_gradient import PCG
from adacg_solver.solvers.adaptive_conjugate_gradient import AdaptiveCG
from adacg_solver.sketching.multiworker_sketcher import SketchLoader
from experiments.dataloading_utils import load_real_data


def run_test_helper(a, b, reg_param, num_workers=1):

    def dist_fn_w(atol=1e-4):
        def dist_fn(x, y):
            if isinstance(x, torch.Tensor):
                return torch.mean(torch.abs(x-y))
            else:
                return np.mean(np.abs(x-y))
        return dist_fn

    dist_fn = dist_fn_w()

    n, d = a.shape
    pcg_sketch_size = min(n // 2, 4*d)
    ada_sketch_size = 32
    max_sketch_size = min(n // 2, 4*d)
    n_iterations_pcg = 50
    n_iterations_cg = 50
    n_iterations_adacg = 50
    tolerance = 1e-10

    dm = DirectMethod(a=a, b=b, reg_param=reg_param)
    dm.fit()
    x_opt = dm.x_fit

    cg = CG(a, b, reg_param, x_opt=x_opt, rescale_data=True, check_reg_param=True, least_squares=True)
    cg.fit(n_iterations=n_iterations_cg,
           tolerance=tolerance,
           get_full_metrics=False)

    adj_ = 'wrong' if dist_fn(cg.x_fit, x_opt) > 1e-4 else 'OK'
    print(f'============= {adj_} solution for CG - {dist_fn(cg.x_fit, x_opt)=}')

    for sketch_fn in ['gaussian', 'srht', 'sjlt']:
        if issparse(a) and sketch_fn != 'sjlt':
            continue
        pcg = PCG(a, b, reg_param, x_opt=x_opt, rescale_data=True, check_reg_param=True, least_squares=True)
        pcg.fit(n_iterations=n_iterations_pcg,
                tolerance=tolerance,
                sketch_size=pcg_sketch_size,
                sketch_fn=sketch_fn,
                get_full_metrics=False)
        adj_ = 'wrong' if dist_fn(pcg.x_fit, x_opt) > 1e-4 else 'OK'
        print(f'============= {adj_} solution for PCG and {sketch_fn=} - {dist_fn(pcg.x_fit, x_opt)=}')

    for sketch_fn in ['gaussian', 'srht', 'sjlt']:
        if issparse(a) and sketch_fn != 'sjlt':
            continue
        num_workers_ = 1 if sketch_fn == 'srht' else num_workers
        ada_solver = AdaptiveCG(a, b, reg_param, x_opt=x_opt, rescale_data=False)
        sketch_loader = SketchLoader(num_workers=num_workers_)

        ada_solver.fit(
            sketch_loader=sketch_loader,
            sketch_size=ada_sketch_size,
            max_sketch_size=max_sketch_size,
            sketch_fn=sketch_fn,
            tolerance=tolerance,
            n_iterations_cg=n_iterations_cg,
            n_iterations=n_iterations_adacg,
            get_full_metrics=False,
        )
        adj_ = 'wrong' if dist_fn(ada_solver.x_fit, x_opt) > 1e-4 else 'OK'
        print(f'============= {adj_} solution for Ada-CG and {sketch_fn=} - {dist_fn(ada_solver.x_fit, x_opt)=}')


def run_test(dataset_name='california_housing', encode_data=True, with_torch=True, dtype=torch.float64,
             n_samples=-1, n_columns=-1, num_workers=1):

    a, b = load_real_data(dataset_name=dataset_name, encode_data=encode_data, n_samples=n_samples, n_columns=n_columns,
                          with_torch=with_torch, dtype=dtype)
    print(f"{dataset_name=}, {with_torch=}, {dtype=}, {n_samples=}, {num_workers=}")
    print(f"{a.shape=}, {type(a)=}, {a.dtype=}, {b.shape=}, {type(b)=}, {b.dtype=}")
    reg_param = 1e-1
    run_test_helper(a, b, reg_param, num_workers=num_workers)


if __name__ == '__main__':
    for num_workers in [1, 4]:
        for encode_data in [False]:
            run_test(dataset_name='california_housing', encode_data=encode_data, with_torch=False, dtype=np.float64,
                     n_samples=-1, n_columns=-1, num_workers=num_workers)
            run_test(dataset_name='california_housing', encode_data=encode_data, with_torch=True, dtype=torch.float64,
                     n_samples=-1, n_columns=-1, num_workers=num_workers)

    for num_workers in [1, 4]:
        run_test(dataset_name='rcv1', encode_data=False, with_torch=False, dtype=np.float64,
                 n_samples=-1, n_columns=1000, num_workers=num_workers)

    print(f"SUCCESS!")







