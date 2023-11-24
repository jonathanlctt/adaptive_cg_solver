from time import time, sleep

import torch
import numpy as np

from adacg_solver.sketching.multiworker_sketcher import SketchLoader


if __name__ == "__main__":

    start = time()
    n_children = 1
    sketch_loader = SketchLoader(num_workers=n_children)
    while not sketch_loader.handshake_done:
        pass

    device = torch.device('cuda:1')
    #device = torch.device('cpu')
    n = 32000
    d = 2048
    a = torch.randn(n, d, device=device) / np.sqrt(n)
    b = torch.randn(n, device=device) / np.sqrt(n)
    y = a.T @ b
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    sketch_fn = 'gaussian'
    sketch_size = 32
    max_sketch_size = 4096
    reg_param = 1e-1
    cg_params = {
        'n_iterations': 20,
        'tolerance': 1e-10,
        'get_full_metrics': True,
        'x_opt': None
    }

    sketch_loader.send_data_to_workers(a, y, sketch_fn, sketch_size, max_sketch_size, reg_param, cg_params)
    sketch_loader.prefetch()
    # #sketch_loader.kill_and_restart_dm()
    sketch_loader.stop_and_reset_cg_worker()
    sketch_loader.reset_sketch_workers()
    sketch_loader.reset_concat_worker()

    sketch_loader.send_data_to_workers(a, y, sketch_fn, sketch_size, max_sketch_size, reg_param, cg_params)
    sketch_loader.prefetch()
    start = time()
    for _ in range(10):
        while not sketch_loader.sketch_done():
            pass

        sa, factorization, sasa_nu = sketch_loader.get()

        print(f"time={time()-start}, {sa.shape=}, {sa.device=}")

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