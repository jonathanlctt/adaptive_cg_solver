import os
from pathlib import Path

import numpy as np
import torch

from ..solvers.direct_method import DirectMethod
from ..solvers.conjugate_gradient import CG
from ..solvers.preconditioned_conjugate_gradient import PCG
from ..solvers.adaptive_conjugate_gradient import AdaptiveCG, init_multi_worker

from .utils import load_real_data, make_and_write_df, make_synthetic_example

#TODO: experiments with sequence of least-squares problems (Newton's method, trust-region methods)
#TODO: compare to standard libraries implementations (scipy, sklearn, pytorch)
#TODO: optimize implementation with C++ snippets pybind11


def single_run(a, b, reg_param, config, xp_dir, xp_id=0, x_opt=None):

    os.makedirs(xp_dir, exist_ok=True)

    sketch_fn = config['sketch_fn']
    pcg_sketch_size = config['pcg_sketch_size']
    ada_sketch_size = config['ada_sketch_size']
    max_sketch_size = config['max_sketch_size']
    tolerance = config['tolerance']
    tolerance_warmup = config['tolerance_warmup']
    n_iterations = config['n_iterations']
    n_iterations_cg = config['n_iterations_cg']
    num_workers = config['num_workers']
    method = config['method']

    if method == 'dm':
        dm_solver = DirectMethod(a, b, reg_param, rescale_data=False)
        x_opt, t_dm = dm_solver.solve()
        make_and_write_df(config=config, errs=np.array([1., tolerance]), gradient_norms=np.array([1, tolerance]),
                          handshake_time=0.,
                          times=np.array([t_dm, 0]), iters=None, sketch_sizes=None, method='dm', xp_id=xp_id, xp_dir=xp_dir)

    elif method == 'cg':
        cg_solver = CG(a, b, reg_param, x_opt=x_opt, rescale_data=False)
        _, errs, residuals, times, iters = cg_solver.solve(n_iterations=n_iterations_cg, tolerance=tolerance, get_metrics=True)
        make_and_write_df(config=config, errs=errs, gradient_norms=residuals, times=times, iters=iters, sketch_sizes=None,
                          handshake_time=0.,
                          method='cg', xp_id=xp_id, xp_dir=xp_dir)

    elif method == 'pcg':
        pcg_solver = PCG(a, b, reg_param, x_opt=x_opt, rescale_data=False)
        _, errs, residuals, times, iters, ssizes = pcg_solver.solve(n_iterations=n_iterations,
                                                                             sketch_size=pcg_sketch_size,
                                                                             sketch_fn=sketch_fn,
                                                                             tolerance=tolerance,
                                                                             get_metrics=True)
        make_and_write_df(config=config, errs=errs, gradient_norms=residuals, times=times, iters=iters, sketch_sizes=ssizes,
                          handshake_time=0.,
                          method='pcg', xp_id=xp_id, xp_dir=xp_dir)

    elif method == 'adacg':
        ada_solver = AdaptiveCG(a, b, reg_param, x_opt=x_opt, rescale_data=False)

        num_workers_ = 1 if num_workers >= 1 and sketch_fn in ['srht', 'naive_srht'] else num_workers

        sketch_loader, handshake_time = init_multi_worker(num_workers=num_workers_, with_torch=True)

        _, errs, residuals, times, iters, ssizes, time_to_send_data = ada_solver.solve(sketch_loader=sketch_loader,
                                                                                    sketch_size=ada_sketch_size,
                                                                                    max_sketch_size=max_sketch_size,
                                                                                    sketch_fn=sketch_fn,
                                                                                    tolerance=tolerance,
                                                                                    tolerance_warm_up=tolerance_warmup,
                                                                                    n_iterations_cg=n_iterations_cg,
                                                                                    n_iterations=n_iterations,
                                                                                    get_metrics=True)

        make_and_write_df(config=config, errs=errs, gradient_norms=residuals, times=times, iters=iters,
                          sketch_sizes=ssizes,
                          handshake_time=handshake_time, time_to_send_data=time_to_send_data,
                          method='adacg', xp_id=xp_id, xp_dir=xp_dir)

        del sketch_loader
        del ada_solver.sketch_loader
        del ada_solver

    return x_opt


def run_synthetic_experiments(list_n, list_d, list_deff, list_cn, xp_name, with_torch=True, dtype=torch.float32):

    xp_dir = Path(f'/Users/jonathanlacotte/code/numerical_results/effective_dimension_solver/synthetic/single_run') / xp_name

    xp_id = 0
    for n in list_n:
        for d in list_d:
            for deff in list_deff:
                for cn in list_cn:
                    if deff <= d and d <= n:

                        a, b, reg_param, deff_, cn_ = make_synthetic_example(n, d, deff, cn, dtype=dtype)
                        print(f"\n{n=}, {d=}, {deff=}, {deff_=}, {cn=}, {cn_=}, {reg_param=}")

                        config = {'n': n, 'd': d, 'deff': deff_, 'cn': cn,
                                  'pcg_sketch_size': 4 * d if d <= n // 8 else min(2 * d, n),
                                  'ada_sketch_size': min(max(32, d // 16), 8192),
                                  'max_sketch_size': 4 * d if d <= n // 8 else min(2 * d, n),
                                  'tolerance': 1e-10,
                                  'tolerance_warmup': 1e-4,
                                  'n_iterations': 200,
                                  'n_iterations_cg': 1000,
                                  'with_torch': with_torch,
                                  'sketch_fn': 'none',
                                  'num_workers': 1,
                                  'method': 'dm',
                                  'dtype': 'float32' if dtype in [torch.float32, np.float32] else 'float64'}

                        x_opt =  single_run(a, b, reg_param, config, xp_dir=xp_dir, xp_id=xp_id, x_opt=None)
                        xp_id += 1
                        config['method'] = 'cg'
                        single_run(a, b, reg_param, config, xp_dir=xp_dir, xp_id=xp_id, x_opt=x_opt)
                        xp_id += 1

                        for method in ['pcg', 'adacg']:
                            for sketch_fn in ['gaussian', 'sjlt']:
                                config['method'] = method
                                config['sketch_fn'] = sketch_fn
                                single_run(a, b, reg_param, config, xp_dir=xp_dir, xp_id=xp_id, x_opt=x_opt)
                                xp_id += 1


def run_real_experiments(dataset_name, encode_data, xp_name, exponents):
    xp_dir = Path(f'/Users/jonathanlacotte/code/numerical_results/effective_dimension_solver/{dataset_name}/single_run') / xp_name
    a, b = load_real_data(dataset_name, encode_data=encode_data)
    n = a.shape[0]
    d = a.shape[1]

    xp_id = 0
    for exponent in exponents:
        nu2 = float(10**(-exponent))
        reg_param = np.sqrt(nu2)
        print(f"\n{dataset_name=}, {exponent=}, {encode_data=}, {n=}, {d=}, {reg_param=}")

        config = {'n': n, 'd': d, 'reg_param': reg_param, 'dataset_name': dataset_name, 'encode_data': encode_data,
                  'pcg_sketch_size': 4 * d if d <= n // 8 else min(n, 2 * d),
                  'ada_sketch_size': min(max(32, d // 8), 8192),
                  'max_sketch_size': 4 * d if d <= n // 8 else min(n, 2 * d),
                  'n_iterations': 200, 'n_iterations_cg': 1000, 'with_torch': True,
                  'sketch_fn': 'none',
                  'num_workers': 1,
                  'method': 'dm'}

        try:
            x_opt = single_run(a, b, reg_param, config, xp_dir=xp_dir, xp_id=xp_id, x_opt=None)
            xp_id += 1
        except:
            print(f"DM failed for: {dataset_name=}, {exponent=}, {encode_data=}, {n=}, {d=}, {reg_param=}")
            continue

        config['method'] = 'cg'
        single_run(a, b, reg_param, config, xp_dir=xp_dir, xp_id=xp_id, x_opt=x_opt)
        xp_id += 1

        for method in ['pcg', 'adacg']:
            for sketch_fn in ['srht', 'gaussian', 'sjlt']:
                config['method'] = method
                config['sketch_fn'] = sketch_fn
                single_run(a, b, reg_param, config, xp_dir=xp_dir, xp_id=xp_id, x_opt=x_opt)
                xp_id += 1


if __name__ == "__main__":

    list_n = [1024, 2048, 4096, 8192,] #16384, 33768]
    list_d = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    list_deff = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    list_cn = [4, 1024]
    xp_name = '1107_1830'
    run_synthetic_experiments(list_n=list_n, list_d=list_d, list_deff=list_deff, list_cn=list_cn,
                              xp_name=xp_name, with_torch=True, dtype=torch.float64)





