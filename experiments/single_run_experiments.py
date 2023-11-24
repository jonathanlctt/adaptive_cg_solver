import os
from pathlib import Path

import numpy as np
import torch

from adacg_solver.solvers.direct_method import DirectMethod
from adacg_solver.solvers.conjugate_gradient import CG
from adacg_solver.solvers.preconditioned_conjugate_gradient import PCG
from adacg_solver.solvers.adaptive_conjugate_gradient import AdaptiveCG, init_multi_worker

from utils import make_and_write_df, make_synthetic_example
from dataloading_utils import load_real_data

#TODO: experiments with sequence of least-squares problems (Newton's method, trust-region methods)
#TODO: compare to standard libraries implementations (scipy, sklearn, pytorch)
#TODO: optimize implementation with C++ snippets pybind11


def single_run(a, b, reg_param, config, xp_dir, xp_id=0, x_opt=None, rescale_data=False):

    os.makedirs(xp_dir, exist_ok=True)

    sketch_fn = config['sketch_fn']
    pcg_sketch_size = config['pcg_sketch_size']
    ada_sketch_size = config['ada_sketch_size']
    max_sketch_size = config['max_sketch_size']
    tolerance = config['tolerance']
    n_iterations = config['n_iterations']
    n_iterations_cg = config['n_iterations_cg']
    num_workers = config['num_workers']
    method = config['method']

    if method == 'dm':
        dm_solver = DirectMethod(a, b, reg_param, rescale_data=rescale_data)
        dm_solver.fit()
        x_opt = dm_solver.x_fit
        t_dm = dm_solver.metrics['fit_time']
        make_and_write_df(config=config,
                          errs=np.array([1., tolerance]),
                          gradient_norms=np.array([1, tolerance]),
                          handshake_time=0.,
                          times=np.array([t_dm, t_dm]), iters=None, sketch_sizes=None, method='dm', xp_id=xp_id, xp_dir=xp_dir)

    elif method == 'cg':
        cg_solver = CG(a, b, reg_param, x_opt=x_opt, rescale_data=rescale_data)
        cg_solver.fit(n_iterations=n_iterations_cg,
                      tolerance=tolerance,
                      get_full_metrics=True,
                      )
        cg_metrics = cg_solver.metrics
        make_and_write_df(config=config,
                          errs=cg_metrics['errors'],
                          gradient_norms=cg_metrics['gradient_norms'],
                          times=cg_metrics['times'],
                          iters=cg_metrics['iterations_array'],
                          sketch_sizes=None,
                          handshake_time=0.,
                          method='cg',
                          xp_id=xp_id,
                          xp_dir=xp_dir)

    elif method == 'pcg':
        pcg_solver = PCG(a, b, reg_param, x_opt=x_opt, rescale_data=rescale_data)
        pcg_solver.fit(n_iterations=n_iterations,
                       sketch_size=pcg_sketch_size,
                       sketch_fn=sketch_fn,
                       tolerance=tolerance,
                       get_full_metrics=True,
                       )
        pcg_metrics = pcg_solver.metrics
        make_and_write_df(config=config,
                          errs=pcg_metrics['errors'],
                          gradient_norms=pcg_metrics['gradient_norms'],
                          times=pcg_metrics['times'],
                          iters=pcg_metrics['iterations_array'],
                          sketch_sizes=pcg_metrics['sketch_sizes'],
                          handshake_time=0.,
                          method='pcg',
                          xp_id=xp_id,
                          xp_dir=xp_dir)

    elif method == 'adacg':
        ada_solver = AdaptiveCG(a, b, reg_param, x_opt=x_opt, rescale_data=rescale_data)

        num_workers_ = 1 if num_workers >= 1 and sketch_fn in ['srht', 'naive_srht'] else num_workers

        sketch_loader, handshake_time = init_multi_worker(num_workers=num_workers_,
                                                          with_torch=isinstance(a, torch.Tensor))

        ada_solver.fit(sketch_loader=sketch_loader,
                       sketch_size=ada_sketch_size,
                       max_sketch_size=max_sketch_size,
                       sketch_fn=sketch_fn,
                       tolerance=tolerance,
                       n_iterations_cg=n_iterations_cg,
                       n_iterations=n_iterations,
                       get_full_metrics=True)

        adacg_metrics = ada_solver.metrics

        make_and_write_df(config=config,
                          errs=adacg_metrics['errors'],
                          gradient_norms=adacg_metrics['gradient_norms'],
                          times=adacg_metrics['times'],
                          iters=adacg_metrics['iterations_array'],
                          sketch_sizes=adacg_metrics['sketch_sizes'],
                          handshake_time=handshake_time,
                          time_to_send_data=adacg_metrics['time_to_send_data'],
                          method='adacg',
                          xp_id=xp_id,
                          xp_dir=xp_dir,
                          )

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

                        config = {'n': n, 'd': d, 'deff': deff, 'deff_': deff_, 'cn': cn,
                                  'pcg_sketch_size': 4 * d if d <= n // 8 else min(2 * d, n),
                                  'ada_sketch_size': min(max(32, d // 16), 8192),
                                  'max_sketch_size': 4 * d if d <= n // 8 else min(2 * d, n),
                                  'tolerance': 1e-10,
                                  'n_iterations': 500,
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
                            for sketch_fn in ['gaussian', 'sjlt', 'srht']:
                                config['method'] = method
                                config['sketch_fn'] = sketch_fn
                                single_run(a, b, reg_param, config, xp_dir=xp_dir, xp_id=xp_id, x_opt=x_opt)
                                xp_id += 1


def run_real_experiments(dataset_name, encode_data, xp_name, exponents,
                         n_samples=-1, n_columns=-1, encoded_size=3500):
    xp_dir = Path(f'/Users/jonathanlacotte/code/numerical_results/effective_dimension_solver/{dataset_name}/single_run') / xp_name
    a, b = load_real_data(dataset_name,
                          encode_data=encode_data,
                          with_torch=True,
                          dtype=torch.float32,
                          n_samples=n_samples,
                          n_columns=n_columns,
                          encoded_size=encoded_size,
                          )
    n = a.shape[0]
    d = a.shape[1]

    xp_id = 0
    for exponent in exponents:
        nu2 = float(10**(-exponent))
        reg_param = np.sqrt(nu2)
        print(f"\n{dataset_name=}, {a.dtype=}, {exponent=}, {encode_data=}, {n=}, {d=}, {reg_param=}")

        config = {'n': n, 'd': d,
                  'reg_param': reg_param,
                  'dataset_name': dataset_name,
                  'encode_data': encode_data,
                  'exponent': exponent,
                  'pcg_sketch_size': 4 * d if d <= n // 8 else min(n, 2 * d),
                  'ada_sketch_size': min(max(32, d // 8), 8192),
                  'max_sketch_size': 4 * d if d <= n // 8 else min(n, 2 * d),
                  'n_iterations': 500,
                  'n_iterations_cg': 2000,
                  'with_torch': True,
                  'tolerance': 1e-10,
                  'sketch_fn': 'none',
                  'num_workers': 1,
                  'method': 'dm'}

        x_opt = single_run(a, b, reg_param, config, xp_dir=xp_dir, xp_id=xp_id, x_opt=None, rescale_data=True)
        xp_id += 1

        config['method'] = 'cg'
        single_run(a, b, reg_param, config, xp_dir=xp_dir, xp_id=xp_id, x_opt=x_opt, rescale_data=True)
        xp_id += 1

        for method in ['pcg', 'adacg']:
            for sketch_fn in ['gaussian', 'sjlt', 'srht']:
                if dataset_name == 'rcv1' and sketch_fn != 'sjlt':
                    continue
                if sketch_fn == 'gaussian' and method == 'pcg':
                    continue
                config['method'] = method
                config['sketch_fn'] = sketch_fn
                single_run(a, b, reg_param, config, xp_dir=xp_dir, xp_id=xp_id, x_opt=x_opt, rescale_data=True)
                xp_id += 1


if __name__ == "__main__":

    list_n = [16384, 32768]
    list_d = [4096, 8192, 16384]
    list_deff = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    list_cn = [4, 1024]
    xp_name = '1116_0919_2'
    run_synthetic_experiments(list_n=list_n, list_d=list_d, list_deff=list_deff, list_cn=list_cn,
                              xp_name=xp_name, with_torch=True, dtype=torch.float32)

    # xp_name = '1115_1552'
    # xp_name = '1115_1801_other'
    # exponents = [7, 6, 5, 4, 3, 2, 1, 0]
    #
    # for encode_data in [True]:
    #     for dataset_name in ['year_prediction']:
    #
    #         n_samples = -1
    #         n_columns = -1
    #         if dataset_name == 'wesad':
    #             encoded_size = 3000
    #             if encode_data:
    #                 n_samples = 100000
    #         if dataset_name == 'rcv1':
    #             n_columns = 15000
    #             encoded_size = 4000
    #         if dataset_name == 'california_housing':
    #             encoded_size = 4000
    #         if dataset_name == 'year_prediction':
    #             encoded_size = 3000
    #             if encode_data:
    #                 n_samples = 80000
    #
    #         xp_name_ = f"{xp_name}_{encode_data=}"
    #
    #         run_real_experiments(dataset_name=dataset_name,
    #                              encode_data=encode_data,
    #                              xp_name=xp_name_,
    #                              exponents=exponents,
    #                              n_samples=n_samples,
    #                              n_columns=n_columns,
    #                              encoded_size=encoded_size)




