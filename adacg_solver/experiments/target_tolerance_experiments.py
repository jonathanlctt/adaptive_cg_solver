import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timeit import default_timer as time

from ..solvers.direct_method import DirectMethod
from ..solvers.conjugate_gradient import CG
from ..solvers.preconditioned_conjugate_gradient import PCG
from ..solvers.adaptive_conjugate_gradient import AdaptiveCG, init_multi_worker

from .utils import make_synthetic_example
from ..datasets.dataloading_utils import load_real_data


def target_tolerance_real_experiments(list_tolerance, list_reg_param, xp_name, dataset_name, encode_data=True, dtype=torch.float64):

    xp_dir = Path(f'/Users/jonathanlacotte/code/numerical_results/effective_dimension_solver/{dataset_name}/target_tolerance') / xp_name
    os.makedirs(xp_dir, exist_ok=True)

    xp_id = 0
    n_trials = 3

    for tolerance in list_tolerance:
        for reg_param in list_reg_param:

            res = []

            a, b = load_real_data(dataset_name, encode_data=encode_data, dtype=dtype)
            n, d = a.shape

            pcg_sketch_size = 4 * d if d <= n // 8 else min(2 * d, n)
            ada_sketch_size = min(max(32, d // 8), 8192)
            max_sketch_size = 4 * d if d <= n // 8 else min(2 * d, n // 2)
            tolerance_warmup = 1e-4
            n_iterations = 200
            n_iterations_cg = 1000
            num_workers = 1

            cfg = {
                'n': n, 'd': d, 'reg_param': float(reg_param),
                'pcg_sketch_size': pcg_sketch_size, 'ada_sketch_size': ada_sketch_size,
                'max_sketch_size': max_sketch_size, 'tolerance': tolerance, 'tolerance_warmup': tolerance_warmup,
                'n_iterations': n_iterations, 'n_iterations_cg': n_iterations_cg, 'num_workers': num_workers,
                'dtype': 'float32' if dtype in [torch.float32, np.float32] else 'float64',
            }

            start = time()
            dm_solver = DirectMethod(a, b, reg_param, rescale_data=False)
            _ = dm_solver.solve()
            t_dm = time() - start
            df_dm = pd.DataFrame(cfg, index=[0])
            df_dm['time'] = t_dm
            df_dm['method'] = 'dm'
            df_dm['sketch_fn'] = 'none'
            res.append(df_dm)

            t_cg = 0
            for _ in range(n_trials):
                start = time()
                cg_solver = CG(a, b, reg_param, x_opt=None, rescale_data=False)
                _ = cg_solver.solve(n_iterations=n_iterations_cg,
                                    tolerance=tolerance,
                                    get_metrics=False)
                t_cg += 1./n_trials * (time() - start)

            df_cg = pd.DataFrame(cfg, index=[0])
            df_cg['time'] = t_cg
            df_cg['method'] = 'cg'
            df_cg['sketch_fn'] = 'none'
            res.append(df_cg)

            for sketch_fn in ['gaussian', 'sjlt', 'srht']:
                t_pcg = 0
                for _ in range(n_trials):
                    start = time()
                    pcg_solver = PCG(a, b, reg_param, x_opt=None, rescale_data=False)
                    _ = pcg_solver.solve(n_iterations=n_iterations,
                                          sketch_size=pcg_sketch_size,
                                          sketch_fn=sketch_fn,
                                          tolerance=tolerance,
                                          get_metrics=False)
                    t_pcg += 1./n_trials * (time() - start)

                df_pcg = pd.DataFrame(cfg, index=[0])
                df_pcg['time'] = t_pcg
                df_pcg['method'] = 'pcg'
                df_pcg['sketch_fn'] = sketch_fn
                res.append(df_pcg)

            for sketch_fn in ['gaussian', 'sjlt', 'srht']:
                num_workers_ = 1 if num_workers >= 1 and sketch_fn in ['srht', 'naive_srht'] else num_workers
                sketch_loader, handshake_time = init_multi_worker(num_workers=num_workers_)

                t_adacg = 0

                for _ in range(n_trials):

                    start = time()
                    ada_solver = AdaptiveCG(a, b, reg_param, x_opt=None, rescale_data=False)

                    _ = ada_solver.solve(
                        sketch_loader=sketch_loader,
                        sketch_size=ada_sketch_size,
                        max_sketch_size=max_sketch_size,
                        sketch_fn=sketch_fn,
                        tolerance=tolerance,
                        tolerance_warm_up=tolerance_warmup,
                        n_iterations_cg=n_iterations_cg,
                        n_iterations=n_iterations,
                        get_metrics=False)

                    sketch_loader.empty_queues()
                    t_adacg += 1./n_trials * (time() - start)

                del sketch_loader
                del ada_solver.sketch_loader

                df_adacg = pd.DataFrame(cfg, index=[0])
                df_adacg['time'] = t_adacg
                df_adacg['method'] = 'adacg'
                df_adacg['sketch_fn'] = sketch_fn
                res.append(df_adacg)

            df = pd.concat(res, ignore_index=True, axis=0)
            df.to_parquet(xp_dir / f"df_{xp_id}.parquet")
            xp_id += 1


def target_tolerance_synthetic_experiments(tolerance, list_n, list_d, list_deff, list_cn, xp_name, dtype=torch.float32):

    xp_dir = Path(f'/Users/jonathanlacotte/code/numerical_results/effective_dimension_solver/synthetic/target_tolerance') / xp_name
    os.makedirs(xp_dir, exist_ok=True)

    xp_id = 0
    n_trials = 3

    for n in list_n:
        for d in list_d:
            for deff in list_deff:
                for cn in list_cn:
                    if deff <= d and d <= n:

                        res = []

                        a, b, reg_param, deff_, cn_ = make_synthetic_example(n, d, deff, cn, dtype=dtype)
                        print(f"\n{n=}, {d=}, {deff=}, {deff_=}, {cn=}, {cn_=}, {reg_param=}")

                        pcg_sketch_size = 4 * d if d <= n // 8 else min(2 * d, n)
                        ada_sketch_size = min(max(32, d // 8), 8192)
                        max_sketch_size = 4 * d if d <= n // 8 else min(2 * d, n // 2)
                        tolerance_warmup = 1e-4
                        n_iterations = 200
                        n_iterations_cg = 1000
                        num_workers = 1

                        cfg = {
                            'n': n, 'd': d, 'deff': deff, 'cn': cn, 'reg_param': reg_param,
                            'pcg_sketch_size': pcg_sketch_size, 'ada_sketch_size': ada_sketch_size,
                            'max_sketch_size': max_sketch_size, 'tolerance': tolerance, 'tolerance_warmup': tolerance_warmup,
                            'n_iterations': n_iterations, 'n_iterations_cg': n_iterations_cg, 'num_workers': num_workers,
                            'dtype': 'float32' if dtype in [torch.float32, np.float32] else 'float64',
                        }

                        start = time()
                        dm_solver = DirectMethod(a, b, reg_param, rescale_data=False)
                        _ = dm_solver.solve()
                        t_dm = time() - start
                        df_dm = pd.DataFrame(cfg, index=[0])
                        df_dm['time'] = t_dm
                        df_dm['method'] = 'dm'
                        df_dm['sketch_fn'] = 'none'
                        res.append(df_dm)

                        t_cg = 0
                        for _ in range(n_trials):
                            start = time()
                            cg_solver = CG(a, b, reg_param, x_opt=None, rescale_data=False)
                            _ = cg_solver.solve(n_iterations=n_iterations_cg,
                                                tolerance=tolerance,
                                                get_metrics=False)
                            t_cg += 1./n_trials * (time() - start)

                        df_cg = pd.DataFrame(cfg, index=[0])
                        df_cg['time'] = t_cg
                        df_cg['method'] = 'cg'
                        df_cg['sketch_fn'] = 'none'
                        res.append(df_cg)

                        for sketch_fn in ['gaussian', 'sjlt', 'srht']:
                            t_pcg = 0
                            for _ in range(n_trials):
                                start = time()
                                pcg_solver = PCG(a, b, reg_param, x_opt=None, rescale_data=False)
                                _ = pcg_solver.solve(n_iterations=n_iterations,
                                                      sketch_size=pcg_sketch_size,
                                                      sketch_fn=sketch_fn,
                                                      tolerance=tolerance,
                                                      get_metrics=False)
                                t_pcg += 1./n_trials * (time() - start)

                            df_pcg = pd.DataFrame(cfg, index=[0])
                            df_pcg['time'] = t_pcg
                            df_pcg['method'] = 'pcg'
                            df_pcg['sketch_fn'] = sketch_fn
                            res.append(df_pcg)

                        for sketch_fn in ['gaussian', 'sjlt', 'srht']:
                            num_workers_ = 1 if num_workers >= 1 and sketch_fn in ['srht', 'naive_srht'] else num_workers
                            sketch_loader, handshake_time = init_multi_worker(num_workers=num_workers_)

                            t_adacg = 0

                            for _ in range(n_trials):

                                start = time()
                                ada_solver = AdaptiveCG(a, b, reg_param, x_opt=None, rescale_data=False)

                                _ = ada_solver.solve(
                                    sketch_loader=sketch_loader,
                                    sketch_size=ada_sketch_size,
                                    max_sketch_size=max_sketch_size,
                                    sketch_fn=sketch_fn,
                                    tolerance=tolerance,
                                    tolerance_warm_up=tolerance_warmup,
                                    n_iterations_cg=n_iterations_cg,
                                    n_iterations=n_iterations,
                                    get_metrics=False)

                                sketch_loader.empty_queues()
                                t_adacg += 1./n_trials * (time() - start)

                            del sketch_loader
                            del ada_solver.sketch_loader

                            df_adacg = pd.DataFrame(cfg, index=[0])
                            df_adacg['time'] = t_adacg
                            df_adacg['method'] = 'adacg'
                            df_adacg['sketch_fn'] = sketch_fn
                            res.append(df_adacg)

                        df = pd.concat(res, ignore_index=True, axis=0)
                        df.to_parquet(xp_dir / f"df_{xp_id}.parquet")
                        xp_id += 1



if __name__ == "__main__":

    do_synthetic = True

    if do_synthetic:
        list_n = [2048, 4096, 8192, 16384, 32768, 65536]
        list_d = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        list_deff = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        list_cn = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        xp_name = '1107_0624'
        target_tolerance_synthetic_experiments(tolerance=1e-6,
                                               list_n=list_n,
                                               list_d=list_d,
                                               list_deff=list_deff,
                                               list_cn=list_cn,
                                               xp_name=xp_name,
                                               dtype=torch.float64
                                               )
    else:
        list_tolerance = [10 ** (-jj) for jj in [2, 4, 6, 8, 10]]
        list_reg_param = [10 ** (-jj) for jj in [-2, -1, 0, 1, 2, 3, 4]]
        xp_name = '1107_0624'
        for dataset_name in ['california_housing', 'year_prediction']:
            target_tolerance_real_experiments(list_tolerance,
                                              list_reg_param,
                                              xp_name,
                                              dataset_name,
                                              encode_data=False,
                                              dtype=torch.float64)






