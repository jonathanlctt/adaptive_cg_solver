import torch.multiprocessing as torch_mp
import multiprocessing as mp
from multiprocessing import shared_memory
import itertools
import queue

import torch
from torch import cat as torch_cat
from torch import eye as torch_eye

import numpy as np
from numpy import eye as numpy_eye
from numpy import vstack as numpy_vstack

import scipy
from scipy.sparse import issparse
from scipy.linalg import cholesky as scipy_cholesky
from torch.linalg import cholesky as torch_cholesky
from scipy.sparse import identity as sparse_identity_matrix, isspmatrix_csc
from scipy.sparse.linalg import factorized as sparse_factorization

from .srht import srht_sketch
from .sketches import hadamard_matrix, sjlt
from ..linear_algebra.linear_algebra import get_max_sval_approx, get_reg_param_threshold, solve_triangular


def cholesky_factorization_wrapper(upper_mat):
    def factorization(z):
        return solve_triangular(upper_mat, solve_triangular(upper_mat.T, z, lower=True), lower=False)

    return factorization


class SketchLoader:

    def __init__(self, num_workers=1, with_torch=True):

        self.with_torch = with_torch
        self.a_shm = []
        self.a_data_shm = []
        self.a_indices_shm = []
        self.a_indptr_shm = []
        self.b_shm = []

        self.index = 0
        self.num_workers = max(1, num_workers)
        self.output_queue = torch_mp.Queue() if with_torch else mp.Queue()
        self.main_queue = torch_mp.Queue() if with_torch else mp.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = itertools.cycle(range(num_workers))
        self.cache = {}
        self.prefetch_index = 0
        self.mega_iteration = -1

        n_processes = num_workers + 3

        worker_id = 0
        for _ in range(num_workers):
            index_queue = torch_mp.Queue() if with_torch else mp.Queue()
            w_ = torch_mp.Process(target=multi_worker_fn,
                                  args=(self.num_workers,
                                        index_queue,
                                        self.output_queue,
                                        worker_id,
                                        n_processes)
                                  )
            worker_id += 1
            w_.daemon = True
            w_.start()
            self.workers.append(w_)
            self.index_queues.append(index_queue)

        self.wc_handshake_queue = torch_mp.Queue() if with_torch else mp.Queue()
        self.wc = torch_mp.Process(target=concat_fn,
                                   args=(self.wc_handshake_queue,
                                         self.output_queue,
                                         self.main_queue,
                                         self.num_workers,
                                         worker_id,
                                         n_processes,
                                         )
                                   )
        worker_id += 1
        self.wc.daemon = True
        self.wc.start()

        self.w_dm_in_queue = torch_mp.Queue() if with_torch else mp.Queue()
        self.w_dm_out_queue = torch_mp.Queue() if with_torch else mp.Queue()
        self.w_dm = torch_mp.Process(target=dm_fn,
                                     args=(self.w_dm_in_queue, self.w_dm_out_queue, n_processes)
                                   )
        worker_id += 1
        self.w_dm.daemon = True
        self.w_dm.start()

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
        self.w_dm_in_queue.put(-1)

        total_handshakes = 0
        while total_handshakes < self.num_workers + 1:
            try:
                index = self.output_queue.get(timeout=0)
                if index == -1:
                    total_handshakes += 1
            except queue.Empty:
                continue
        while True:
            try:
                index = self.w_dm_out_queue.get(timeout=0)
                if index == -1:
                    break
            except queue.Empty:
                continue

        self.wc_handshake_queue.put(-1)
        self.w_dm_in_queue.put(-1)
        self.handshake_done = True

    def send_data_to_workers(self, a, b, sketch_fn, sketch_size, max_sketch_size, reg_param):

        self.mega_iteration += 1
        for k_ in list(self.cache.keys()):
            if k_[0] < self.mega_iteration:
                del self.cache[k_]
        self.index = 0
        self.prefetch_index = 0
        n, d = a.shape
        self.sketch_size = min(n, sketch_size)
        self.max_sketch_size = min(n, max_sketch_size)
        self.sketch_fn = sketch_fn
        self.nu = reg_param

        if self.with_torch:
            self.a = a
            self.b = b
            self.a.share_memory_()
            self.b.share_memory_()
            for n_queue, index_queue in enumerate(self.index_queues):
                index_queue.put(('torch_data', self.a, self.nu, self.sketch_fn, self.sketch_size, self.max_sketch_size))
            self.w_dm_in_queue.put(('torch_data', self.a, self.b, self.nu))
        else:
            if not issparse(a):
                idx_shm = len(self.a_shm)
                self.a_shm.append(shared_memory.SharedMemory(create=True, size=a.nbytes, name=f'a_shm_{idx_shm}'))
                self.a = np.ndarray(a.shape, dtype=a.dtype, buffer=self.a_shm[idx_shm].buf)
                self.a[:] = a[:]
                self.b_shm.append(shared_memory.SharedMemory(create=True, size=b.nbytes, name=f'b_shm_{idx_shm}'))
                self.b = np.ndarray(b.shape, dtype=b.dtype, buffer=self.b_shm[idx_shm].buf)
                self.b[:] = b[:]
                for n_queue, index_queue in enumerate(self.index_queues):
                    index_queue.put(('numpy_data', idx_shm, a.shape, a.dtype, self.nu, self.sketch_fn, self.sketch_size, self.max_sketch_size))
                self.w_dm_in_queue.put(('numpy_data', idx_shm, a.shape, a.dtype, b.shape, b.dtype, self.nu))
            else:
                assert scipy.sparse.isspmatrix_csc(a) or scipy.sparse.isspmatrix_csr(a), 'only csc and csr matrices are supported'
                idx_shm = len(self.a_data_shm)
                self.a_data_shm.append(shared_memory.SharedMemory(create=True, size=a.data.nbytes, name=f'a_data_shm_{idx_shm}'))
                self.a_indices_shm.append(shared_memory.SharedMemory(create=True, size=a.indices.nbytes, name=f'a_indices_shm_{idx_shm}'))
                self.a_indptr_shm.append(shared_memory.SharedMemory(create=True, size=a.indptr.nbytes, name=f'a_indptr_shm_{idx_shm}'))
                self.a_data = np.ndarray(a.data.shape, dtype=a.data.dtype, buffer=self.a_data_shm[idx_shm].buf)
                self.a_data[:] = a.data[:]
                self.a_indices = np.ndarray(a.indices.shape, dtype=a.indices.dtype, buffer=self.a_indices_shm[idx_shm].buf)
                self.a_indices[:] = a.indices[:]
                self.a_indptr = np.ndarray(a.indptr.shape, dtype=a.indptr.dtype, buffer=self.a_indptr_shm[idx_shm].buf)
                self.a_indptr[:] = a.indptr[:]
                self.b_shm.append(shared_memory.SharedMemory(create=True, size=b.nbytes, name=f'b_shm_{idx_shm}'))
                self.b = np.ndarray(b.shape, dtype=b.dtype, buffer=self.b_shm[idx_shm].buf)
                self.b[:] = b[:]
                if scipy.sparse.isspmatrix_csc(a):
                    a_type = 'csc'
                elif scipy.sparse.isspmatrix_csr(a):
                    a_type = 'csr'
                for n_queue, index_queue in enumerate(self.index_queues):
                    index_queue.put(('scipy_sparse_data', idx_shm,
                                     a.shape, a_type,
                                     a.data.shape, a.data.dtype,
                                     a.indices.shape, a.indices.dtype,
                                     a.indptr.shape, a.indptr.dtype,
                                     self.nu, self.sketch_fn, self.sketch_size, self.max_sketch_size))
                self.w_dm_in_queue.put(('scipy_sparse_data', idx_shm,
                                        a.shape, a_type,
                                        a.data.shape, a.data.dtype,
                                        a.indices.shape, a.indices.dtype,
                                        a.indptr.shape, a.indptr.dtype,
                                        b.shape, b.dtype,
                                        self.nu))

    def prefetch(self):
        while self.prefetch_index < self.index + 2 * self.num_workers:
            self.index_queues[next(self.worker_cycle)].put((('index', self.mega_iteration, self.prefetch_index)))
            self.prefetch_index += 1

    def sketch_done(self):
        k_ = (self.mega_iteration, self.index // self.num_workers)
        if k_ not in self.cache:
            try:
                mega_iteration, index, sa, prefactorization, sasa_nu = self.main_queue.get(timeout=0)
                if mega_iteration == self.mega_iteration:
                    self.cache[(mega_iteration, index)] = (sa, prefactorization, sasa_nu)
            except queue.Empty:
                pass
        return k_ in self.cache

    def get_dm(self):
        x_opt = None
        try:
            x_opt = self.w_dm_out_queue.get(timeout=0)
        except queue.Empty:
            pass
        return x_opt

    def get(self):

        assert self.sketch_done(), 'all chunks not yet computed'

        k_ = (self.mega_iteration, self.index // self.num_workers)
        sa, prefactorization, sasa_nu = self.cache[k_]

        factorization = cholesky_factorization_wrapper(prefactorization)

        if sa.shape[0] < self.max_sketch_size:
            del self.cache[k_]
            self.index += self.num_workers
            self.prefetch()

        return sa, factorization, sasa_nu

    def __del__(self):
        try:
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:
                q.cancel_join_thread()
                q.close()
            self.w_dm_in_queue.put(None)
            self.w_dm.join(timeout=5.0)

            self.output_queue.cancel_join_thread()
            self.output_queue.close()
            self.wc_handshake_queue.cancel_join_thread()
            self.wc_handshake_queue.close()
            self.main_queue.cancel_join_thread()
            self.main_queue.close()
            self.w_dm_in_queue.cancel_join_thread()
            self.w_dm_in_queue.close()
            self.w_dm_out_queue.cancel_join_thread()
            self.w_dm_out_queue.close()

            for l_ in [self.a_shm, self.a_data_shm, self.a_indices_shm, self.a_indptr_shm, self.b_shm]:
                for i_ in l_:
                    i_.close()
                    i_.unlink()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()
            if self.wc.is_alive():
                self.wc.terminate()
            if self.w_dm.is_alive():
                self.w_dm.terminate()


def multi_worker_fn(num_workers, index_queue, output_queue, worker_id, n_processes):

    torch.manual_seed(42 + worker_id)
    torch.set_num_threads(torch_mp.cpu_count() // n_processes)

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
                _, a, nu, sketch_fn, sketch_size, max_sketch_size = message
                has_reached_max_sketch_size = False
                continue
            elif message[0] == 'numpy_data':
                _, idx_shm, a_shape, a_dtype, nu, sketch_fn, sketch_size, max_sketch_size = message
                a_shm = shared_memory.SharedMemory(name=f'a_shm_{idx_shm}')
                a = np.ndarray(a_shape, dtype=a_dtype, buffer=a_shm.buf)
                has_reached_max_sketch_size = False
                continue
            elif message[0] == 'scipy_sparse_data':
                _, idx_shm, a_shape, a_type, a_data_shape, a_data_type, a_indices_shape, a_indices_type, a_indptr_shape, a_indptr_type, nu, sketch_fn, sketch_size, max_sketch_size = message
                data_shm = shared_memory.SharedMemory(name=f'a_data_shm_{idx_shm}')
                indices_shm = shared_memory.SharedMemory(name=f'a_indices_shm_{idx_shm}')
                indptr_shm = shared_memory.SharedMemory(name=f'a_indptr_shm_{idx_shm}')
                a_data = np.ndarray(a_data_shape, dtype=a_data_type, buffer=data_shm.buf)
                a_indices = np.ndarray(a_indices_shape, dtype=a_indices_type, buffer=indices_shm.buf)
                a_indptr = np.ndarray(a_indptr_shape, dtype=a_indptr_type, buffer=indptr_shm.buf)
                if a_type == 'csc':
                    a = scipy.sparse.csc_matrix((a_data, a_indices, a_indptr), shape=a_shape)
                elif a_type == 'csr':
                    a = scipy.sparse.csr_matrix((a_data, a_indices, a_indptr), shape=a_shape)
                has_reached_max_sketch_size = False
                continue
            elif message[0] == 'index':
                _, mega_iteration, index = message
                if index is None:
                    break
        except queue.Empty:
            continue

        sketch_round = index // num_workers
        current_sketch_size = min((2 ** sketch_round) * sketch_size, max_sketch_size)
        pw_sketch_size = current_sketch_size // num_workers
        remainder = index % num_workers == num_workers - 1
        if remainder:
            pw_sketch_size += (current_sketch_size - pw_sketch_size * num_workers)

        if not has_reached_max_sketch_size:
            with_torch = isinstance(a, torch.Tensor)
            has_reached_max_sketch_size = current_sketch_size == max_sketch_size

            if sketch_fn == 'gaussian':
                s = torch.randn(pw_sketch_size, a.shape[0], dtype=a.dtype) if with_torch else np.random.randn(pw_sketch_size, a.shape[0]).astype(a.dtype)
                sa = s @ a / np.sqrt(current_sketch_size)
            elif sketch_fn == 'srht':
                sa = srht_sketch(a, pw_sketch_size, with_stack=True)
            elif sketch_fn == 'naive_srht':
                s = hadamard_matrix(n=a.shape[0], sketch_size=pw_sketch_size, with_torch=with_torch)
                sa = s @ a
            elif sketch_fn == 'sjlt':
                sa = sjlt(a=a, sketch_size=pw_sketch_size, nnz_per_column=num_workers)
            else:
                raise NotImplementedError

            output_queue.put((mega_iteration, index, sa, nu))


def cholesky(h, lower=False):
    if isinstance(h, torch.Tensor):
        if lower:
            return torch_cholesky(h)
        else:
            return torch_cholesky(h).mH
    else:
        return scipy_cholesky(h, lower=lower)


def dm_fn(w_dm_in_queue, w_dm_out_queue, n_processes):

    torch.set_num_threads(torch_mp.cpu_count() // n_processes)

    while True:
        try:
            _ = w_dm_in_queue.get(timeout=0)
            w_dm_out_queue.put(-1)
            break
        except queue.Empty:
            continue

    while True:
        try:
            _ = w_dm_in_queue.get(timeout=0)
            break
        except queue.Empty:
            continue

    while True:
        try:
            message = w_dm_in_queue.get(timeout=0)
            if message is None:
                break
            elif message[0] == 'torch_data':
                _, a, b, nu = message
            elif message[0] == 'numpy_data':
                _, idx_shm, a_shape, a_dtype, b_shape, b_dtype, nu = message
                a_shm = shared_memory.SharedMemory(name=f'a_shm_{idx_shm}')
                b_shm = shared_memory.SharedMemory(name=f'b_shm_{idx_shm}')
                a = np.ndarray(a_shape, dtype=a_dtype, buffer=a_shm.buf)
                b = np.ndarray(b_shape, dtype=b_dtype, buffer=b_shm.buf)
            elif message[0] == 'scipy_sparse_data':
                _, idx_shm, a_shape, a_type, a_data_shape, a_data_type, a_indices_shape, a_indices_type, a_indptr_shape, a_indptr_type, b_shape, b_dtype, nu = message

                data_shm = shared_memory.SharedMemory(name=f'a_data_shm_{idx_shm}')
                indices_shm = shared_memory.SharedMemory(name=f'a_indices_shm_{idx_shm}')
                indptr_shm = shared_memory.SharedMemory(name=f'a_indptr_shm_{idx_shm}')

                a_data = np.ndarray(a_data_shape, dtype=a_data_type, buffer=data_shm.buf)
                a_indices = np.ndarray(a_indices_shape, dtype=a_indices_type, buffer=indices_shm.buf)
                a_indptr = np.ndarray(a_indptr_shape, dtype=a_indptr_type, buffer=indptr_shm.buf)

                if a_type == 'csc':
                    a = scipy.sparse.csc_matrix((a_data, a_indices, a_indptr), shape=a_shape)
                elif a_type == 'csr':
                    a = scipy.sparse.csr_matrix((a_data, a_indices, a_indptr), shape=a_shape)

                b_shm = shared_memory.SharedMemory(name=f'b_shm_{idx_shm}')
                b = np.ndarray(b_shape, dtype=b_dtype, buffer=b_shm.buf)
        except queue.Empty:
            continue

        with_torch = isinstance(a, torch.Tensor)

        if not issparse(a):
            if with_torch:
                hessian = a.T @ a + nu ** 2 * torch.eye(a.shape[1], dtype=a.dtype)
            else:
                hessian = a.T @ a + nu ** 2 * numpy_eye(a.shape[1], dtype=a.dtype)
            upper_mat = cholesky(hessian, lower=False)
            x_opt = solve_triangular(upper_mat, solve_triangular(upper_mat.T, b, lower=True), lower=False)
        else:
            hessian = a.T @ a + nu ** 2 * sparse_identity_matrix(a.shape[1], dtype=a.dtype)
            if not isspmatrix_csc(hessian):
                hessian = hessian.T.tocsc()
            factorization = sparse_factorization(hessian)
            x_opt = factorization(b)

        w_dm_out_queue.put(x_opt)


def concat_fn(handshake_queue, output_queue, main_queue, num_workers, worker_id, n_processes):

    torch.manual_seed(42 + worker_id)
    torch.set_num_threads(torch_mp.cpu_count() // n_processes)

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

        if isinstance(cache[(mega_iteration, start_idx)][0], torch.Tensor):
            sa = torch_cat([cache[(mega_iteration, start_idx + i_)][0] for i_ in range(num_workers)], dim=0)
        else:
            if issparse(cache[(mega_iteration, start_idx)][0]):
                sa = scipy.sparse.vstack([cache[(mega_iteration, start_idx + i_)][0] for i_ in range(num_workers)])
            else:
                sa = numpy_vstack([cache[(mega_iteration, start_idx + i_)][0] for i_ in range(num_workers)])

        if issparse(sa):
            sa = sa.toarray()

        sasa = sa @ sa.T if sa.shape[0] <= sa.shape[1] else sa.T @ sa

        top_singular_value = get_max_sval_approx(sasa, niter=2)
        threshold = np.sqrt(top_singular_value) * get_reg_param_threshold(sasa)
        sasa_nu = max(nu, threshold)

        if isinstance(sasa, torch.Tensor):
            prefactorization = cholesky(sasa + sasa_nu ** 2 * torch_eye(sasa.shape[0], dtype=sasa.dtype), lower=False)
        else:
            prefactorization = cholesky(sasa + sasa_nu ** 2 * numpy_eye(sasa.shape[0], dtype=sasa.dtype), lower=False)

        main_queue.put((mega_iteration, start_idx // num_workers, sa, prefactorization, sasa_nu))

        for i_ in range(num_workers):
            del cache[(mega_iteration, start_idx + i_)]

        start_idx += num_workers








