from eff_dim_solver.sketching.multiworker_sketcher import SketchLoader
import scipy

import torch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)


def test_numpy(sketch_fn='gaussian', num_workers=1):

    sketch_loader = SketchLoader(num_workers=num_workers, with_torch=False)

    while True:
        if sketch_loader.handshake_done:
            break
        else:
            pass

    n, d = 2048, 1024
    a = np.random.randn(n, d) / np.sqrt(n)
    b = np.random.randn(n, 1) / np.sqrt(n)

    sketch_size = 512
    max_sketch_size = 1024
    reg_param = 1e-1

    sketch_loader.send_data_to_workers(a=a,
                                       b=a.T @ b,
                                       sketch_fn=sketch_fn,
                                       sketch_size=sketch_size,
                                       max_sketch_size=max_sketch_size,
                                       reg_param=reg_param)

    sketch_loader.prefetch()
    for _iter in range(2):
        while not sketch_loader.sketch_done():
            continue
        sa, sasa, sasa_nu = sketch_loader.get()

        if _iter == 0:
            assert sa.shape[0] == sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'
        elif _iter == 1:
            expected_sketch_size = min(2*sketch_size, max_sketch_size)
            assert sa.shape[0] == expected_sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {expected_sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'

    sketch_loader.empty_queues()

    n, d = 4096, 512
    a = np.random.randn(n, d) / np.sqrt(n)
    b = np.random.randn(n, 1) / np.sqrt(n)
    sketch_size = 768
    max_sketch_size = 1024
    reg_param = 1e-1

    sketch_loader.send_data_to_workers(a=a,
                                       b=a.T @ b,
                                       sketch_fn=sketch_fn,
                                       sketch_size=sketch_size,
                                       max_sketch_size=max_sketch_size,
                                       reg_param=reg_param)

    sketch_loader.prefetch()
    for _iter in range(2):
        while not sketch_loader.sketch_done():
            continue
        sa, sasa, sasa_nu = sketch_loader.get()

        if _iter == 0:
            assert sa.shape[0] == sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'
        elif _iter == 1:
            expected_sketch_size = min(2*sketch_size, max_sketch_size)
            assert sa.shape[0] == expected_sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {expected_sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'

    sketch_loader.empty_queues()

    del sketch_loader

    logging.info(f"SUCCESS! numpy - {sketch_fn=}, {num_workers=}")


def test_torch(sketch_fn='gaussian', num_workers=1, dtype=torch.float32):
    sketch_loader = SketchLoader(num_workers=num_workers, with_torch=True)

    while True:
        if sketch_loader.handshake_done:
            break
        else:
            pass

    n, d = 2048, 1024
    a = torch.randn(n, d, dtype=dtype) / np.sqrt(n)
    b = torch.randn(n, 1, dtype=dtype) / np.sqrt(n)

    sketch_size = 512
    max_sketch_size = 1024
    reg_param = 1e-1

    sketch_loader.send_data_to_workers(a=a,
                                       b=a.T @ b,
                                       sketch_fn=sketch_fn,
                                       sketch_size=sketch_size,
                                       max_sketch_size=max_sketch_size,
                                       reg_param=reg_param)

    sketch_loader.prefetch()
    for _iter in range(2):
        while not sketch_loader.sketch_done():
            continue
        sa, sasa, sasa_nu = sketch_loader.get()

        if _iter == 0:
            assert sa.shape[0] == sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'
        elif _iter == 1:
            expected_sketch_size = min(2 * sketch_size, max_sketch_size)
            assert sa.shape[
                       0] == expected_sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {expected_sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'

    sketch_loader.empty_queues()

    n, d = 4096, 512
    a = torch.randn(n, d, dtype=torch.float32) / np.sqrt(n)
    b = torch.randn(n, 1, dtype=torch.float32) / np.sqrt(n)
    sketch_size = 768
    max_sketch_size = 1024
    reg_param = 1e-1

    sketch_loader.send_data_to_workers(a=a,
                                       b=a.T @ b,
                                       sketch_fn=sketch_fn,
                                       sketch_size=sketch_size,
                                       max_sketch_size=max_sketch_size,
                                       reg_param=reg_param)

    sketch_loader.prefetch()
    for _iter in range(2):
        while not sketch_loader.sketch_done():
            continue
        sa, sasa, sasa_nu = sketch_loader.get()

        if _iter == 0:
            assert sa.shape[0] == sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'
        elif _iter == 1:
            expected_sketch_size = min(2 * sketch_size, max_sketch_size)
            assert sa.shape[
                       0] == expected_sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {expected_sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'

    sketch_loader.empty_queues()

    del sketch_loader

    logging.info(f"SUCCESS! torch - {sketch_fn=}, {num_workers=}")


def test_scipy_sparse(sketch_fn='gaussian', sptype='csc', num_workers=1):

    sketch_loader = SketchLoader(num_workers=num_workers, with_torch=False)

    while True:
        if sketch_loader.handshake_done:
            break
        else:
            pass

    n = 1024
    d = 512
    row_ids = np.random.choice(n, d)
    col_ids = np.arange(d, dtype=np.int32)
    data = np.random.randn(d, )
    if sptype == 'csc':
        a = scipy.sparse.csc_array((data, (row_ids, col_ids)), shape=(n, d))
    elif sptype == 'csr':
        a = scipy.sparse.csr_array((data, (row_ids, col_ids)), shape=(n, d))
    b = np.random.randn(n,1) / np.sqrt(n)

    sketch_size = 48
    max_sketch_size = 2048
    reg_param = 1e-1

    sketch_loader.send_data_to_workers(a=a,
                                       b=a.T @ b,
                                       sketch_fn=sketch_fn,
                                       sketch_size=sketch_size,
                                       max_sketch_size=max_sketch_size,
                                       reg_param=reg_param)

    sketch_loader.prefetch()
    for _iter in range(2):
        while not sketch_loader.sketch_done():
            continue
        sa, sasa, sasa_nu = sketch_loader.get()
        if _iter == 0:
            assert sa.shape[0] == sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'
        elif _iter == 1:
            expected_sketch_size = min(2*sketch_size, max_sketch_size)
            assert sa.shape[0] == expected_sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {expected_sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'

    sketch_loader.empty_queues()

    n = 2048
    d = 702
    row_ids = np.random.choice(n, d)
    col_ids = np.arange(d, dtype=np.int32)
    data = np.random.randn(d,)
    if sptype == 'csc':
        a = scipy.sparse.csc_matrix((data, (row_ids, col_ids)), shape=(n, d))
    elif sptype == 'csr':
        a = scipy.sparse.csr_matrix((data, (row_ids, col_ids)), shape=(n, d))
    b = np.random.randn(n,1) / np.sqrt(n)

    sketch_size = 768
    max_sketch_size = 1024
    reg_param = 1e-1

    sketch_loader.send_data_to_workers(a=a,
                                       b=a.T @ b,
                                       sketch_fn=sketch_fn,
                                       sketch_size=sketch_size,
                                       max_sketch_size=max_sketch_size,
                                       reg_param=reg_param)

    sketch_loader.prefetch()
    for _iter in range(2):
        while not sketch_loader.sketch_done():
            continue
        sa, sasa, sasa_nu = sketch_loader.get()
        if _iter == 0:
            assert sa.shape[0] == sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'
        elif _iter == 1:
            expected_sketch_size = min(2*sketch_size, max_sketch_size)
            assert sa.shape[0] == expected_sketch_size, f'unexpected sketch size: {sa.shape[0]=}, {expected_sketch_size=}'
            assert sa.shape[1] == d, f'unexpected shape: {sa.shape[1]=}, {d=}'

    sketch_loader.empty_queues()

    del sketch_loader
    logging.info(f"SUCCESS! sparse - {sketch_fn=}, {sptype=}, {num_workers=}")


if __name__ == '__main__':
    for sketch_fn in ['sjlt', 'gaussian', 'srht']:
        for num_workers in [1, 4]:

            if sketch_fn == 'srht' and num_workers > 1:
                continue

            test_numpy(sketch_fn=sketch_fn, num_workers=num_workers)
            test_torch(sketch_fn=sketch_fn, num_workers=num_workers, dtype=torch.float64)
            test_torch(sketch_fn=sketch_fn, num_workers=num_workers, dtype=torch.float32)
            test_scipy_sparse(sketch_fn=sketch_fn, num_workers=num_workers, sptype='csc')
            test_scipy_sparse(sketch_fn=sketch_fn, num_workers=num_workers, sptype='csr')