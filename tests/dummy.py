import torch.multiprocessing as torch_mp
from multiprocessing import shared_memory
import itertools
import queue

import torch
from torch import cat as torch_cat

import numpy as np

from adacg_solver.linear_algebra.linear_algebra import get_max_sval_approx, get_reg_param_threshold


class SketchLoader:

    def __init__(self, num_workers=1):

        self.index = 0
        self.num_workers = max(1, num_workers)
        self.output_queue = torch_mp.Queue()
        self.main_queue = torch_mp.Queue()
        self.index_queues = []
        self.data_to_worker_events = []
        self.workers = []
        self.worker_cycle = itertools.cycle(range(num_workers))
        self.cache = {}
        self.prefetch_index = 0
        self.mega_iteration = -1

        worker_id = 0
        for _ in range(num_workers):
            index_queue = torch_mp.Queue()
            data_to_worker_event = torch_mp.Event()
            w_ = torch_mp.Process(target=multi_worker_fn,
                                  args=(data_to_worker_event,
                                        self.num_workers,
                                        index_queue,
                                        self.output_queue,
                                        worker_id,
                                        )
                                  )
            worker_id += 1
            w_.daemon = True
            w_.start()
            self.workers.append(w_)
            self.index_queues.append(index_queue)
            self.data_to_worker_events.append(data_to_worker_event)

        self.wc_handshake_queue = torch_mp.Queue()
        self.wc = torch_mp.Process(target=concat_fn,
                                   args=(self.wc_handshake_queue,
                                         self.output_queue,
                                         self.main_queue,
                                         self.num_workers,
                                         worker_id,
                                         )
                                   )
        worker_id += 1
        self.wc.daemon = True
        self.wc.start()

        self.handshake_done = False
        self.do_handshake()

    def empty_queues(self):
        for index_queue in self.index_queues:
            is_not_empty = True
            while is_not_empty:
                try:
                    _ = index_queue.get(timeout=0)
                except queue.Empty:
                    is_not_empty = False

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
        self.handshake_done = True

    def send_data_to_workers(self, a, sketch_size, max_sketch_size, reg_param):

        self.mega_iteration += 1
        for k_ in list(self.cache.keys()):
            if k_[0] < self.mega_iteration:
                del self.cache[k_]
        self.index = 0
        self.prefetch_index = 0
        n, d = a.shape
        self.sketch_size = min(n, sketch_size)
        self.max_sketch_size = min(n, max_sketch_size)
        self.nu = reg_param
        self.a = a
        self.a.share_memory_()
        for n_queue, index_queue in enumerate(self.index_queues):
            self.data_to_worker_events[n_queue].clear()
            index_queue.put(('torch_data', self.a, self.nu, self.sketch_size, self.max_sketch_size))
            self.data_to_worker_events[n_queue].wait()

    def prefetch(self):
        while self.prefetch_index < self.index + 2 * self.num_workers:
            self.index_queues[next(self.worker_cycle)].put((('index', self.mega_iteration, self.prefetch_index)))
            self.prefetch_index += 1

    def sketch_done(self):
        k_ = (self.mega_iteration, self.index // self.num_workers)
        if k_ not in self.cache:
            try:
                mega_iteration, index, sa, sasa_nu = self.main_queue.get(timeout=0)
                if mega_iteration == self.mega_iteration:
                    self.cache[(mega_iteration, index)] = (sa, sasa_nu)
            except queue.Empty:
                pass
        return k_ in self.cache

    def get(self):

        assert self.sketch_done(), 'all chunks not yet computed'

        k_ = (self.mega_iteration, self.index // self.num_workers)
        sa, sasa_nu = self.cache[k_]

        if sa.shape[0] < self.max_sketch_size:
            del self.cache[k_]
            self.index += self.num_workers
            self.prefetch()

        return sa, sasa_nu

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
            self.wc_handshake_queue.cancel_join_thread()
            self.wc_handshake_queue.close()
            self.main_queue.cancel_join_thread()
            self.main_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()
            if self.wc.is_alive():
                self.wc.terminate()


def multi_worker_fn(data_to_worker_event, num_workers, index_queue, output_queue, worker_id):

    torch.manual_seed(42 + worker_id)

    while True:
        try:
            _ = index_queue.get(timeout=0)
            output_queue.put(-1)
            break
        except queue.Empty:
            continue

    while True:
        try:
            message = index_queue.get(timeout=0)
            if message is None:
                break
            if message[0] == 'torch_data':
                _, a, nu, sketch_size, max_sketch_size = message
                data_to_worker_event.set()
                has_reached_max_sketch_size = False
                print(f"received data {a.shape=}")
                continue
            elif message[0] == 'index':
                _, mega_iteration, index = message
                if index is None:
                    break
        except queue.Empty:
            continue

        print(f"computing sketch ... ")

        sketch_round = index // num_workers
        current_sketch_size = min((2 ** sketch_round) * sketch_size, max_sketch_size)
        pw_sketch_size = current_sketch_size // num_workers
        remainder = index % num_workers == num_workers - 1
        if remainder:
            pw_sketch_size += (current_sketch_size - pw_sketch_size * num_workers)

        if not has_reached_max_sketch_size:
            has_reached_max_sketch_size = current_sketch_size == max_sketch_size
            s = torch.randn(pw_sketch_size, a.shape[0], dtype=a.dtype)
            print(f"generated s")
            sa = s @ a / np.sqrt(current_sketch_size)
            output_queue.put((mega_iteration, index, sa, nu))
            print(f"computed sketch and put sketch into queue")


def concat_fn(handshake_queue, output_queue, main_queue, num_workers, worker_id):

    torch.manual_seed(42 + worker_id)

    while True:
        try:
            _ = handshake_queue.get(timeout=0)
            output_queue.put(-1)
            break
        except queue.Empty:
            continue

    while True:
        try:
            _ = handshake_queue.get(timeout=0)
            break
        except queue.Empty:
            continue

    cache = {}
    start_idx = 0
    mega_iteration = 0
    while True:
        wait_for_all_chunks = True
        while wait_for_all_chunks:
            try:
                mega_iteration_, index, data, nu = output_queue.get(timeout=0)
                if mega_iteration_ > mega_iteration:
                    for k_ in list(cache.keys()):
                        if k_[0] < mega_iteration_:
                            del cache[k_]
                    mega_iteration = mega_iteration_
                    start_idx = 0
                    cache[(mega_iteration, index)] = (data, nu)
                elif mega_iteration_ == mega_iteration:
                    cache[(mega_iteration, index)] = (data, nu)
                else:
                    continue
            except queue.Empty:
                pass

            all_chunks_in_cache = True
            for i_ in range(num_workers):
                if (mega_iteration, start_idx + i_) not in cache:
                    all_chunks_in_cache = False

            if all_chunks_in_cache:
                wait_for_all_chunks = False

        nu = cache[(mega_iteration, start_idx)][1]

        sa = torch_cat([cache[(mega_iteration, start_idx + i_)][0] for i_ in range(num_workers)], dim=0)

        sasa = sa @ sa.T if sa.shape[0] <= sa.shape[1] else sa.T @ sa

        top_singular_value = get_max_sval_approx(sasa, niter=2)
        threshold = np.sqrt(top_singular_value) * get_reg_param_threshold(sasa)
        sasa_nu = max(nu, threshold)

        main_queue.put((mega_iteration, start_idx // num_workers, sa, sasa_nu))

        for i_ in range(num_workers):
            del cache[(mega_iteration, start_idx + i_)]

        start_idx += num_workers


if __name__ == "__main__":

    n = 2048
    d = 512
    sketch_size = 32
    max_sketch_size = 1024
    reg_param = 1e-1
    a = torch.randn(n, d) / np.sqrt(n)

    num_workers = 1
    sketch_loader = SketchLoader(num_workers=num_workers)
    while True:
        if sketch_loader.handshake_done:
            break

    sketch_loader.send_data_to_workers(a, sketch_size, max_sketch_size, reg_param)

    sketch_loader.prefetch()

    while True:
        if sketch_loader.sketch_done():
            break

    sa, sasa_nu = sketch_loader.get()

    print(f"{sa.shape=}")

    del sketch_loader















