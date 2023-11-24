import os
from pathlib import Path
from timeit import default_timer as time

import pandas as pd

import multiprocessing
import itertools
import queue

import torch
from torch import Tensor as torch_tensor
from torch import cat as torch_cat
from torch import eye as torch_eye

import numpy as np
from numpy import eye as numpy_eye
from numpy import vstack as numpy_vstack

from adacg_solver.sketching.srht import srht_sketch
from adacg_solver.sketching.sketches import hadamard_matrix, sjlt


def dummy_worker_fn(index_queue, output_queue):

    handshake_done = False
    while not handshake_done:
        try:
            index = index_queue.get(timeout=0)
            if index == -1:
                output_queue.put(-1)
            handshake_done = True
        except queue.Empty:
            continue

    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break

        output_queue.put((index, 0))


class DummyMultiWorkerSketcher:
    def __init__(self, num_workers=1):

        self.index = 0

        self.num_workers = num_workers
        self.output_queue = multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = itertools.cycle(range(num_workers))
        self.cache = {}
        self.prefetch_index = 0

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            w_ = multiprocessing.Process(target=dummy_worker_fn,
                                         args=(index_queue, self.output_queue)
                                         )
            w_.daemon = True
            w_.start()
            self.workers.append(w_)
            self.index_queues.append(index_queue)

        self.handshake = False
        self.do_handshake()

    def do_handshake(self):
        for index_queue in self.index_queues:
            index_queue.put(-1)

        total_handshakes = 0
        while total_handshakes < self.num_workers:
            try:
                index = self.output_queue.get(timeout=0)
                if index == -1:
                    total_handshakes += 1
            except queue.Empty:
                continue
        self.handshake = True

    def prefetch(self):
        while self.prefetch_index < self.index + 2 * self.num_workers:
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def sketch_done(self):

        idx_ = self.index
        if idx_ not in self.cache:
            try:
                index, item_ = self.output_queue.get(timeout=0)
                self.cache[index] = item_
            except queue.Empty:
                pass

        return idx_ in self.cache

    def get(self):
        assert self.sketch_done(), 'all chunks not yet computed'
        item_ = self.cache[self.index]
        self.index += 1
        self.prefetch()
        return item_

    def __del__(self):
        try:
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:  # close all queues
                q.cancel_join_thread()
                q.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()


class MultiWorkerSketcher:

    def __init__(self, a, reg_param, sketch_size, max_sketch_size, sketch_fn, num_chunks=1, num_workers=1):
        self.a = a
        self.nu = reg_param
        self.with_torch = isinstance(a, torch.Tensor)
        n = self.a.shape[0]
        sketch_size = min(n, sketch_size)
        max_sketch_size = min(n, max_sketch_size)
        if sketch_fn in ['naive_srht', 'srht']:
            num_workers = 1
        num_chunks = min(num_workers, num_chunks)

        self.sketch_size = sketch_size
        self.max_sketch_size = max_sketch_size
        self.num_chunks = num_chunks
        self.sketch_fn = sketch_fn

        self.index = 0

        self.num_workers = num_workers
        self.output_queue = multiprocessing.Queue()
        self.main_queue = multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = itertools.cycle(range(num_workers))
        self.cache = {}
        self.prefetch_index = 0

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            w_ = multiprocessing.Process(target=multi_worker_fn,
                                         args=(self.a, self.sketch_size, self.max_sketch_size, self.num_chunks,
                                               self.sketch_fn, self.with_torch, index_queue, self.output_queue,
                                               )
                                         )
            w_.daemon = True
            w_.start()
            self.workers.append(w_)
            self.index_queues.append(index_queue)

        self.wc_handshake_queue = multiprocessing.Queue()
        self.wc = multiprocessing.Process(target=concat_fn,
                                          args=(self.wc_handshake_queue, self.output_queue, self.main_queue, self.num_chunks, self.nu)
                                          )
        self.wc.daemon = True
        self.wc.start()
        self.handshake = False

        self.do_handshake()

    def do_handshake(self):
        for index_queue in self.index_queues:
            index_queue.put(-1)
        self.wc_handshake_queue.put(-1)

        total_handshakes = 0
        while total_handshakes < self.num_workers + 1:
            try:
                index = self.output_queue.get(timeout=0)
                if index == -1:
                    total_handshakes += 1
            except queue.Empty:
                continue

        self.wc_handshake_queue.put(-1)
        self.handshake = True

    def prefetch(self):
        while self.prefetch_index < self.index + 2 * self.num_workers:
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def sketch_done(self):

        idx_ = self.index // self.num_chunks
        if idx_ not in self.cache:
            try:
                index, sa, u_ = self.main_queue.get(timeout=0)
                self.cache[index] = (sa, u_)
            except queue.Empty:
                pass

        return idx_ in self.cache

    def get(self):

        assert self.sketch_done(), 'all chunks not yet computed'

        sa, u = self.cache[self.index // self.num_chunks]

        if sa.shape[0] < self.max_sketch_size:
            del self.cache[self.index // self.num_chunks]
            self.index += self.num_chunks
            self.prefetch()

        return sa, u

    def __del__(self):
        try:
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:  # close all queues
                q.cancel_join_thread()
                q.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
            self.main_queue.cancel_join_thread()
            self.main_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()
            if self.wc.is_alive():
                self.wc.terminate()


def multi_worker_fn(a, sketch_size, max_sketch_size, num_chunks, sketch_fn, with_torch, index_queue, output_queue):

    handshake_done = False
    while not handshake_done:
        try:
            index = index_queue.get(timeout=0)
            if index == -1:
                output_queue.put(-1)
            handshake_done = True
        except queue.Empty:
            continue

    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break

        _round = index // num_chunks
        _rem = index % num_chunks == num_chunks - 1

        _sketch_size = min((2 ** _round) * sketch_size, max_sketch_size)
        pw_sketch_size = _sketch_size // num_chunks
        if _rem:
            pw_sketch_size += (_sketch_size - pw_sketch_size * num_chunks)

        if sketch_fn == 'gaussian':
            if with_torch:
                s = torch.randn(pw_sketch_size, a.shape[0], dtype=a.dtype)
            else:
                s = np.random.randn(pw_sketch_size, a.shape[0])
                s = np.array(s, dtype=a.dtype)
            sa = s @ a / np.sqrt(_sketch_size)
        elif sketch_fn == 'srht':
            sa = srht_sketch(a, pw_sketch_size, with_stack=True)
        elif sketch_fn == 'naive_srht':
            s = hadamard_matrix(n=a.shape[0], sketch_size=pw_sketch_size, with_torch=with_torch)
            sa = s @ a
        elif sketch_fn == 'sjlt':
            sa = sjlt(a=a, sketch_size=pw_sketch_size, nnz_per_column=num_chunks, with_torch=with_torch)
        else:
            raise NotImplementedError

        output_queue.put((index, sa))


def is_torch_tensor(x):
    return isinstance(x, torch_tensor)


def concat_fn(handshake_queue, output_queue, main_queue, num_chunks, nu):

    handshake_received = False
    while not handshake_received:
        try:
            index = handshake_queue.get(timeout=0)
            if index == -1:
                output_queue.put(-1)
            handshake_received = True
        except queue.Empty:
            continue

    handshake_acknowledged = False
    while not handshake_acknowledged:
        try:
            _ = handshake_queue.get(timeout=0)
            handshake_acknowledged = True
        except queue.Empty:
            continue

    cache = {}
    start_idx = 0

    while True:
        loop_ = True
        while loop_:
            try:
                index, data = output_queue.get(timeout=0)
                cache[index] = data
            except queue.Empty:
                pass

            b_ = True
            for i_ in range(num_chunks):
                if start_idx + i_ not in cache:
                    b_ = False
            if b_:
                loop_ = False

        if is_torch_tensor(cache[start_idx]):
            sa = torch_cat([cache[start_idx + i_] for i_ in range(num_chunks)], dim=0)
        else:
            sa = numpy_vstack([cache[start_idx + i_] for i_ in range(num_chunks)])

        if sa.shape[0] <= sa.shape[1]:
            sasa = sa @ sa.T
        else:
            sasa = sa.T @ sa

        sketch_size = sa.shape[0]
        d = sa.shape[1]

        if sketch_size > d:
            if is_torch_tensor(sasa):
                u_ = cholesky(sasa + nu ** 2 * torch_eye(d, dtype=sasa.dtype), lower=False)
            else:
                u_ = cholesky(sasa + nu ** 2 * numpy_eye(d, dtype=sasa.dtype), lower=False)
        else:
            if is_torch_tensor(sasa):
                u_ = cholesky(sasa + nu ** 2 * torch_eye(sketch_size, dtype=sasa.dtype), lower=False)
            else:
                u_ = cholesky(sasa + nu ** 2 * numpy_eye(sketch_size, dtype=sasa.dtype), lower=False)

        main_queue.put((start_idx // num_chunks, sa, u_))

        for i_ in range(num_chunks):
            del cache[start_idx + i_]

        start_idx += num_chunks


def cholesky(h, lower=False):
    if lower:
        return torch.linalg.cholesky(h)
    else:
        return torch.linalg.cholesky(h).mH


def get_sketch(a, sketch_fn, sketch_size, nu=1e-2):
    n = a.shape[0]
    d = a.shape[1]

    if sketch_fn == 'srht':
        sa = srht_sketch(a, sketch_size, with_stack=True)
    elif sketch_fn == 'naive_srht':
        s = hadamard_matrix(n=n, sketch_size=sketch_size, with_torch=True)
        sa = s @ a
    elif sketch_fn == 'gaussian':
        s = torch.randn(sketch_size, n, dtype=a.dtype)
        sa = s @ a / np.sqrt(sketch_size)
    elif sketch_fn == 'sjlt':
        sa = sjlt(a=a, sketch_size=sketch_size)
    else:
        raise NotImplementedError

    if sa.shape[0] <= sa.shape[1]:
        sasa = sa @ sa.T
    else:
        sasa = sa.T @ sa

    if sketch_size > d:
        u_ = cholesky(sasa + nu ** 2 * torch.eye(d, dtype=sasa.dtype), lower=False)
    else:
        u_ = cholesky(sasa + nu ** 2 * torch.eye(sketch_size, dtype=sasa.dtype), lower=False)

    return sa, sasa


def get_sketch_times(list_n, list_d, list_m, xp_name):

    xp_dir = Path(f'/Users/jonathanlacotte/code/numerical_results/effective_dimension_solver/sketch_times') / xp_name
    os.makedirs(xp_dir, exist_ok=True)

    xp_id = 0
    for n in list_n:
        for d in list_d:
            a = torch.randn(n, d, dtype=torch.float32) / np.sqrt(n)
            for sketch_size in list_m:
                if d <= n // 4 and sketch_size <= min(n // 2, 4 * d):

                    for sketch_fn in ['sjlt', 'srht', 'gaussian']:

                        if n >= 16384:
                            n_trials = 1
                        else:
                            n_trials = 3

                        print(f"\n{n=}, {d=}, {sketch_size=}, {sketch_fn=}")

                        print(f"base")
                        t_base = 0.
                        for _ in range(n_trials):
                            start = time()
                            _ = get_sketch(a, sketch_fn, sketch_size)
                            t_base += 1./n_trials * (time()-start)

                        print(f"dummy single worker")
                        t_dummy_single_worker = 0.
                        for _ in range(n_trials):
                            sketch_loader = DummyMultiWorkerSketcher(num_workers=1)
                            while not sketch_loader.handshake:
                                continue

                            start = time()
                            sketch_loader.prefetch()
                            while True:
                                if sketch_loader.sketch_done():
                                    _ = sketch_loader.get()
                                    break
                            t_dummy_single_worker += 1. / n_trials * (time() - start)

                        print(f"dummy multi worker")
                        t_dummy_multi_worker = 0.
                        for _ in range(n_trials):
                            sketch_loader = DummyMultiWorkerSketcher(num_workers=4)
                            while not sketch_loader.handshake:
                                continue

                            start = time()
                            sketch_loader.prefetch()
                            while True:
                                if sketch_loader.sketch_done():
                                    _ = sketch_loader.get()
                                    break
                            t_dummy_multi_worker += 1. / n_trials * (time() - start)

                        print("single worker")
                        t_single_worker = 0.
                        for _ in range(n_trials):
                            sketch_loader = MultiWorkerSketcher(a=a,
                                                                sketch_size=sketch_size,
                                                                max_sketch_size=4*sketch_size,
                                                                sketch_fn=sketch_fn,
                                                                num_workers=1,
                                                                num_chunks=1,
                                                                reg_param=1e-2,
                                                                )

                            while not sketch_loader.handshake:
                                continue

                            start = time()
                            sketch_loader.prefetch()
                            while True:
                                if sketch_loader.sketch_done():
                                    _ = sketch_loader.get()
                                    break
                            t_single_worker += 1./n_trials * (time()-start)

                        print("multi worker")
                        t_multi_worker = 0.
                        for _ in range(n_trials):
                            sketch_loader = MultiWorkerSketcher(a=a,
                                                                sketch_size=sketch_size,
                                                                max_sketch_size=4*sketch_size,
                                                                sketch_fn=sketch_fn,
                                                                num_workers=4,
                                                                num_chunks=4,
                                                                reg_param=1e-2,
                                                                )
                            while not sketch_loader.handshake:
                                continue
                            start = time()
                            sketch_loader.prefetch()

                            while True:
                                if sketch_loader.sketch_done():
                                    _ = sketch_loader.get()
                                    break

                            t_multi_worker += 1. / n_trials * (time() - start)

                        d_res = {'n': n, 'd': d, 'sketch_size': sketch_size, 'sketch_fn': sketch_fn,
                                 't_base': t_base, 't_dummy_single_worker': t_dummy_single_worker,
                                 't_dummy_multi_worker': t_dummy_multi_worker,
                                 't_single_worker': t_single_worker, 't_multi_worker': t_multi_worker}

                        df = pd.DataFrame(d_res, index=[0])
                        df.to_parquet(xp_dir / f"df_{xp_id}.parquet")

                        xp_id += 1


if __name__ == "__main__":

    xp_name = '0110_1118'
    list_n = [2 ** jj for jj in range(10, 17)]
    list_d = [2 ** jj for jj in range(7, 14)]
    list_m = [2 ** jj for jj in range(7, 14)]

    get_sketch_times(list_n=list_n, list_d=list_d, list_m=list_m, xp_name=xp_name)
