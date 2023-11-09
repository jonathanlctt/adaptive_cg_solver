import os
from timeit import default_timer as time

import multiprocessing
import itertools
import queue

import torch
from torch import Tensor as torch_tensor
from torch import cat as torch_cat

import numpy as np
from numpy import vstack as numpy_vstack

from srht.srht import srht_sketch
from ..sketching.sketches import hadamard_matrix, sjlt


class TorchMultiWorkerSketcher:

    def __init__(self, a, reg_param, sketch_size, max_sketch_size, sketch_fn, num_chunks=1, num_workers=1):
        print(f"MAIN PROCESS: address of a: {hex(a.data.untyped_storage().data_ptr())}")
        print(f"MAIN PROCESS: address of a[0]: {hex(a[0].data.untyped_storage().data_ptr())}")
        self.a = a  #.share_memory_()
        print(f"MAIN PROCESS: address of self.a: {hex(self.a.data.untyped_storage().data_ptr())}")
        print(f"MAIN PROCESS: address of self.a[0]: {hex(self.a[0].data.untyped_storage().data_ptr())}")
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
        self.output_queue = torch.multiprocessing.Queue()
        self.main_queue = torch.multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = itertools.cycle(range(num_workers))
        self.cache = {}
        self.prefetch_index = 0

        worker_id = 0
        for _ in range(num_workers):
            index_queue = torch.multiprocessing.Queue()
            w_ = torch.multiprocessing.Process(target=multi_worker_fn,
                                               args=(self.a, index_queue, self.output_queue, worker_id, num_workers+1)
                                               )
            worker_id += 1

            print(f"===== created child process")
            print(f"MAIN PROCESS: address of self.a: {hex(self.a.data.untyped_storage().data_ptr())}")

            w_.daemon = True
            w_.start()
            self.workers.append(w_)
            self.index_queues.append(index_queue)
            print(f"===== started child process")
            print(f"MAIN PROCESS: address of self.a: {hex(self.a.data.untyped_storage().data_ptr())}")

        self.wc = torch.multiprocessing.Process(target=concat_fn,
                                                args=(self.output_queue, self.main_queue, worker_id, num_workers+1)
                                                )
        worker_id += 1

        self.wc.daemon = True
        self.wc.start()

        self.prefetch()

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
            for q in self.index_queues:
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


class MultiWorkerSketcher:

    def __init__(self, a, reg_param, sketch_size, max_sketch_size, sketch_fn, num_chunks=1, num_workers=1):
        print(f"\nMAIN PROCESS: address of a: {hex(a.untyped_storage().data_ptr())}")
        print(f"MAIN PROCESS: address of a[0]: {hex(a[0].untyped_storage().data_ptr())}")
        self.a = a
        print(f"MAIN PROCESS: address of self.a: {hex(self.a.untyped_storage().data_ptr())}")
        print(f"MAIN PROCESS: address of self.a[0]: {hex(self.a[0].untyped_storage().data_ptr())}")
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

        worker_id = 0
        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            w_ = multiprocessing.Process(target=multi_worker_fn,
                                         args=(self.a, index_queue, self.output_queue, worker_id, num_workers+1)
                                         )
            worker_id += 1

            print(f"===== created child process")
            print(f"MAIN PROCESS: address of self.a: {hex(self.a.untyped_storage().data_ptr())}")

            w_.daemon = True
            w_.start()
            self.workers.append(w_)
            self.index_queues.append(index_queue)
            print(f"===== started child process")
            print(f"MAIN PROCESS: address of self.a: {hex(self.a.untyped_storage().data_ptr())}")

        self.wc = multiprocessing.Process(target=concat_fn,
                                          args=(self.output_queue, self.main_queue, worker_id, num_workers+1)
                                          )
        worker_id += 1

        self.wc.daemon = True
        self.wc.start()

        self.prefetch()

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
            for q in self.index_queues:
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


def multi_worker_fn(a, index_queue, output_queue, worker_id, n_workers):

    torch.manual_seed(42 + worker_id)
    torch.set_num_threads(torch.multiprocessing.cpu_count() // (n_workers+1))

    if worker_id == 0:
        print(f"WRITING on a in {os.getpid()}")
        a[0] = 2.

    print(f"\nCHILD PROCESS {os.getpid()}: n_vcpus={torch.multiprocessing.cpu_count()}, n_threads={torch.get_num_threads()}")

    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break

        print(f"CHILD PROCESS id: {os.getpid()}, address of a: {hex(a.untyped_storage().data_ptr())}, address of a[0]: {hex(a[0].untyped_storage().data_ptr())}")

        if isinstance(a, torch.Tensor):
            s = torch.randn(5, a.shape[0], dtype=torch.float32)
        else:
            s = np.random.randn(5, a.shape[0]).astype(np.float32)
        sa = s @ a

        print(f"CHILD PROCESS id (after writing) {os.getpid()}: {a[0,:3]=}")

        print(f"CHILD PROCESS id (after writing) {os.getpid()}, address of a: {hex(a.untyped_storage().data_ptr())}, address of a[0]: {hex(a[0].untyped_storage().data_ptr())}")

        output_queue.put((index, sa))


def is_torch_tensor(x):
    return isinstance(x, torch_tensor)


def concat_fn(output_queue, main_queue, worker_id, n_workers):

    torch.manual_seed(42 + worker_id)
    torch.set_num_threads(torch.multiprocessing.cpu_count() // (n_workers+1))
    #torch.set_num_threads(6)

    print(f"CHILD PROCESS {os.getpid()}: n_vcpus={torch.multiprocessing.cpu_count()}, n_threads={torch.get_num_threads()}")

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
            for i_ in range(1):
                if start_idx + i_ not in cache:
                    b_ = False
            if b_:
                loop_ = False

        if is_torch_tensor(cache[start_idx]):
            sa = torch_cat([cache[start_idx + i_] for i_ in range(1)], dim=0)
        else:
            sa = numpy_vstack([cache[start_idx + i_] for i_ in range(1)])

        if sa.shape[0] <= sa.shape[1]:
            sasa = sa @ sa.T
        else:
            sasa = sa.T @ sa

        main_queue.put((start_idx // 1, sa, sasa))

        for i_ in range(1):
            del cache[start_idx + i_]

        start_idx += 1


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

    return sa, sasa


def get_sketch_times(n, d, sketch_size, sketch_fn, num_workers, with_torch=True, torch_multiprocessing=True):

    torch.manual_seed(41)
    #torch.set_num_threads(torch.multiprocessing.cpu_count() // (num_workers+2))
    torch.set_num_threads(6)

    print(f"\nMAIN PROCESS: n_vcpus={torch.multiprocessing.cpu_count()}, n_threads={torch.get_num_threads()}")

    if with_torch:
        a = torch.randn(n, d, dtype=torch.float32) / np.sqrt(n)
    else:
        a = np.random.randn(n, d).astype(np.float32) / np.sqrt(n)

    print(f"MAIN PROCESS: a {hex(a.untyped_storage().data_ptr())}, a[0] {hex(a[0].untyped_storage().data_ptr())}")

    if torch_multiprocessing:
        sketcher_class = TorchMultiWorkerSketcher
    else:
        sketcher_class = MultiWorkerSketcher

    sketch_loader = sketcher_class(a=a,
                                    sketch_size=sketch_size,
                                    max_sketch_size=4*sketch_size,
                                    sketch_fn=sketch_fn,
                                    num_workers=num_workers,
                                    num_chunks=num_workers,
                                    reg_param=1e-2,
                                    )

    print(f"MAIN PROCESS (after init MultiWorkerSketcher): a {hex(a.untyped_storage().data_ptr())}, a[0] {hex(sketch_loader.a[0].untyped_storage().data_ptr())}")

    while True:
        if sketch_loader.sketch_done():
            _ = sketch_loader.get()
            print(f"MAIN PROCESS (after .get()): a {hex(a.untyped_storage().data_ptr())}, a[0] {hex(sketch_loader.a[0].untyped_storage().data_ptr())}")
            print(f"MAIN PROCESS (end): {a[0]=}")
            del sketch_loader
            break


if __name__ == "__main__":

    start = time()
    get_sketch_times(n=1, d=1, sketch_size=512, sketch_fn='gaussian', num_workers=2, torch_multiprocessing=True)
    print(f"total time: {time()-start}")

