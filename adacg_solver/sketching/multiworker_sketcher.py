import itertools
import queue
import time

import torch
import numpy as np

import scipy
from scipy.sparse import issparse
from scipy.linalg import cholesky as scipy_cholesky
from scipy.sparse import identity as sparse_identity_matrix, isspmatrix_csc, isspmatrix_csr
from scipy.sparse.linalg import factorized as sparse_factorization

from .srht import srht_sketch
from .sketches import sjlt
from ..linear_algebra.linear_algebra import get_max_sval_approx, get_reg_param_threshold, solve_triangular
from ..solvers.conjugate_gradient import CG


def cholesky(h, lower=False):
    if isinstance(h, torch.Tensor):
        if lower:
            return torch.linalg.cholesky(h)
        else:
            return torch.linalg.cholesky(h).mH
    else:
        return scipy.linalg.cholesky(h, lower=lower)


def cholesky_factorization_wrapper(upper_mat):
    def factorization(z):
        return solve_triangular(upper_mat, solve_triangular(upper_mat.T, z, lower=True), lower=False)

    return factorization


class SketchLoader:

    def __init__(self, num_workers):

        self.mp_cxt = torch.multiprocessing.get_context('spawn')

        self.n_children = max(1, num_workers)

        self.index = 0
        self.mega_iteration = -1
        self.prefetch_index = 0
        self.cache = {}

        self.signal_are_alive = []
        self.worker_cycle = itertools.cycle(range(self.n_children))

        self.init_sketch_workers()
        self.init_concat_worker()
        self.init_direct_method_worker()
        self.init_cg_worker()

        for e_ in self.signal_are_alive:
            while not e_.is_set():
                pass

        self.handshake_done = True

    def init_sketch_workers(self):
        self.workers = []
        self.q_in = [self.mp_cxt.Queue() for _ in range(self.n_children)]
        self.q_c = self.mp_cxt.Queue()
        for worker_idx in range(self.n_children):
            e_ = self.mp_cxt.Event()
            w_ = self.mp_cxt.Process(target=worker,
                                     args=(self.n_children, e_, worker_idx, self.q_in[worker_idx], self.q_c),
                                     )
            w_.daemon = True
            w_.start()
            self.workers.append(w_)
            self.signal_are_alive.append(e_)

    def reset_sketch_workers(self):
        for index_queue in self.q_in:
            is_not_empty = True
            while is_not_empty:
                try:
                    _ = index_queue.get(timeout=0)
                except queue.Empty:
                    is_not_empty = False

    def init_concat_worker(self):
        self.q_out = self.mp_cxt.Queue()
        self.q_c_break_message = self.mp_cxt.Queue()
        ec = self.mp_cxt.Event()
        self.wc = self.mp_cxt.Process(target=workerc, args=(ec, self.q_c_break_message, self.n_children, self.q_c, self.q_out))
        self.wc.daemon = True
        self.wc.start()
        self.signal_are_alive.append(ec)

    def reset_concat_worker(self):
        next_mega_iteration = self.mega_iteration + 1
        self.q_c_break_message.put(next_mega_iteration)

    def init_direct_method_worker(self):
        e_dm = self.mp_cxt.Event()
        self.q_dm_in = self.mp_cxt.Queue()
        self.q_dm_out = self.mp_cxt.Queue()
        self.w_dm = self.mp_cxt.Process(target=dm_fn, args=(e_dm, self.q_dm_in, self.q_dm_out))
        self.w_dm.daemon = True
        self.w_dm.start()
        self.signal_are_alive.append(e_dm)

    def get_dm(self):
        x_opt = None
        try:
            x_opt = self.q_dm_out.get(timeout=0)
        except queue.Empty:
            pass
        return x_opt

    def kill_and_restart_dm(self):
        try:
            self.q_dm_in.put(None)
            try:
                _ = self.get_dm()
            except:
                pass
            self.q_dm_in.cancel_join_thread()
            self.q_dm_in.close()
            self.q_dm_out.cancel_join_thread()
            self.q_dm_out.close()
        finally:
            if self.w_dm.is_alive():
                self.w_dm.terminate()

        signal_is_alive = self.mp_cxt.Event()
        self.q_dm_in = self.mp_cxt.Queue()
        self.q_dm_out = self.mp_cxt.Queue()
        self.w_dm = self.mp_cxt.Process(target=dm_fn,
                                        args=(signal_is_alive, self.q_dm_in, self.q_dm_out))
        self.w_dm.daemon = True
        self.w_dm.start()  # do not wait for signal_is_alive to be set

    def init_cg_worker(self):

        e_cg = self.mp_cxt.Event()
        self.cg_break_event = self.mp_cxt.Event()
        self.q_cg_in = self.mp_cxt.Queue()
        self.q_cg_out = self.mp_cxt.Queue()
        self.w_cg = self.mp_cxt.Process(target=cg_fn, args=(e_cg, self.cg_break_event, self.q_cg_in, self.q_cg_out))
        self.w_cg.daemon = True
        self.w_cg.start()
        self.signal_are_alive.append(e_cg)

    def stop_and_reset_cg_worker(self):
        self.cg_break_event.set()

    def get_cg(self):
        x_opt = None
        metrics = {}
        try:
            x_opt, metrics = self.q_cg_out.get(timeout=0)
        except queue.Empty:
            pass

        return x_opt, metrics

    def send_data_to_workers(self, a, b, sketch_fn, sketch_size, max_sketch_size, reg_param, cg_params):

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

        if isinstance(a, torch.Tensor):
            a.share_memory_()
            b.share_memory_()
            for idx in range(self.n_children):
                self.q_in[idx].put(('torch_data', a, self.nu, self.sketch_fn, self.sketch_size, self.max_sketch_size))
            self.q_dm_in.put(('torch_data', a, b, self.nu))
            self.q_cg_in.put(('torch_data', a, b, self.nu, cg_params))
        else:
            if isspmatrix_csc(a):
                a_sptype = 'csc'
            else:
                a = a.tocsr()
                a_sptype = 'csr'
            a_data = torch.from_numpy(a.data)
            a_indices = torch.from_numpy(a.indices)
            a_indptr = torch.from_numpy(a.indptr)
            b = torch.from_numpy(b)
            a_data.share_memory_()
            a_indices.share_memory_()
            a_indptr.share_memory_()
            b.share_memory_()

            for n_queue, index_queue in enumerate(self.q_in):
                index_queue.put(('sparse_data', a_sptype, a_data, a_indices, a_indptr, a.shape, self.nu, self.sketch_fn, self.sketch_size, self.max_sketch_size))
            self.q_dm_in.put(('sparse_data', a_sptype, a_data, a_indices, a_indptr, a.shape, b, self.nu))
            self.q_cg_in.put(('sparse_data', a_sptype, a_data, a_indices, a_indptr, a.shape, b, self.nu, cg_params))

    def prefetch(self):
        while self.prefetch_index < self.index + 2 * self.n_children:
            self.q_in[next(self.worker_cycle)].put((('index', self.mega_iteration, self.prefetch_index)))
            self.prefetch_index += 1

    def sketch_done(self):
        k_ = (self.mega_iteration, self.index // self.n_children)
        if k_ not in self.cache:
            try:
                mega_iteration, index, sa, prefactorization, sasa_nu = self.q_out.get(timeout=0)
                if mega_iteration == self.mega_iteration:
                    self.cache[(mega_iteration, index)] = (sa, prefactorization, sasa_nu)
            except queue.Empty:
                pass
        return k_ in self.cache

    def get(self):

        assert self.sketch_done(), 'all chunks not yet computed'

        k_ = (self.mega_iteration, self.index // self.n_children)
        sa, prefactorization, sasa_nu = self.cache[k_]

        factorization = cholesky_factorization_wrapper(prefactorization)

        if sa.shape[0] < self.max_sketch_size:
            del self.cache[k_]
            self.index += self.n_children
            self.prefetch()

        return sa, factorization, sasa_nu

    def __del__(self):
        try:

            self.q_out.cancel_join_thread()
            self.q_out.close()
            self.q_c.put(None)
            self.q_c.cancel_join_thread()
            self.q_c.close()
            self.q_c_break_message.cancel_join_thread()
            self.q_c_break_message.close()
            for i, w in enumerate(self.workers):
                self.q_in[i].put(None)
            for q in self.q_in:
                q.cancel_join_thread()
                q.close()
            self.q_dm_in.put(None)
            try:
                _ = self.get_dm()
            except:
                pass
            self.q_cg_in.put((0,))
            try:
                _ = self.get_cg()
            except:
                pass
            self.q_dm_in.cancel_join_thread()
            self.q_dm_in.close()
            self.q_dm_out.cancel_join_thread()
            self.q_dm_out.close()
            self.q_cg_in.cancel_join_thread()
            self.q_cg_in.close()
            self.q_cg_out.cancel_join_thread()
            self.q_cg_out.close()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()
            if self.wc.is_alive():
                self.wc.terminate()
            if self.w_dm.is_alive():
                self.w_dm.terminate()
            if self.w_cg.is_alive():
                self.w_cg.terminate()


def worker(n_children, signal_is_alive, worker_id, q_in, qc):
    signal_is_alive.set()

    torch.manual_seed(42 + worker_id)

    while True:
        try:
            message = q_in.get(timeout=0)
            if message is None or message[0] not in ['torch_data', 'sparse_data', 'index']:
                del a
                return
            if message[0] == 'torch_data':
                a, nu, sketch_fn, sketch_size, max_sketch_size = message[1:]
                has_reached_max_sketch_size = False
                continue
            elif message[0] == 'sparse_data':
                sp_type, a_data, a_indices, a_indptr, a_shape, nu, sketch_fn, sketch_size, max_sketch_size = message[1:]
                a_data = a_data.numpy()
                a_indices = a_indices.numpy()
                a_indptr = a_indptr.numpy()
                if sp_type == 'csr':
                    a = scipy.sparse.csr_matrix((a_data, a_indices, a_indptr), shape=a_shape)
                else:
                    a = scipy.sparse.csc_matrix((a_data, a_indices, a_indptr), shape=a_shape)
                has_reached_max_sketch_size = False
                continue
            elif message[0] == 'index':
                _, mega_iteration, index = message
                if index is None:
                    break
        except queue.Empty:
            continue

        sketch_round = index // n_children
        current_sketch_size = min((2 ** sketch_round) * sketch_size, max_sketch_size)
        pw_sketch_size = current_sketch_size // n_children
        remainder = index % n_children == n_children - 1

        if remainder:
            pw_sketch_size += (current_sketch_size - pw_sketch_size * n_children)

        if not has_reached_max_sketch_size:
            with_torch = isinstance(a, torch.Tensor)
            has_reached_max_sketch_size = current_sketch_size == max_sketch_size
            if sketch_fn == 'gaussian':
                s = torch.randn(pw_sketch_size, a.shape[0], device=a.device, dtype=a.dtype) if with_torch else np.random.randn(pw_sketch_size, a.shape[0]).astype(a.dtype)
                sa = (s @ a) / np.sqrt(current_sketch_size)
            elif sketch_fn == 'srht':
                sa = srht_sketch(a, pw_sketch_size, with_stack=True)
            elif sketch_fn == 'sjlt':
                sa = sjlt(a=a, sketch_size=pw_sketch_size, nnz_per_column=n_children)

            qc.put((mega_iteration, index, sa, nu))

            del sa


def workerc(signal_is_alive, break_message, n_children, q_c, q_out):

    signal_is_alive.set()

    cache = {}
    start_idx = 0
    mega_iteration = 0

    while True:
        wait_for_all_chunks = True
        while wait_for_all_chunks:
            try:
                message = q_c.get(timeout=0)
                if message is None:
                    del cache
                    return
                else:
                    mega_iteration_, index, data, nu = message
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
                continue

            try:
                next_mega_iteration = break_message.get(timeout=0)
                assert next_mega_iteration > mega_iteration, 'next_mega_iteration <= mega_iteration'
                for k_ in list(cache.keys()):
                    if k_[0] < next_mega_iteration:
                        del cache[k_]
                mega_iteration = next_mega_iteration
                continue
            except queue.Empty:
                pass

            all_chunks_in_cache = True
            for i_ in range(n_children):
                if (mega_iteration_, start_idx+i_) not in cache:
                    all_chunks_in_cache = False

            if all_chunks_in_cache:
                wait_for_all_chunks = False

        nu = cache[(mega_iteration, start_idx)][1]
        if isinstance(cache[(mega_iteration, start_idx)][0], torch.Tensor):
            sa = torch.cat([cache[(mega_iteration, start_idx + i_)][0] for i_ in range(n_children)], dim=0)
        else:
            if issparse(cache[(mega_iteration, start_idx)][0]):
                sa = scipy.sparse.vstack([cache[(mega_iteration, start_idx + i_)][0] for i_ in range(n_children)])
            else:
                sa = np.vstack([cache[(mega_iteration, start_idx + i_)][0] for i_ in range(n_children)])

        if issparse(sa):
            sa = sa.toarray()

        sasa = sa @ sa.T if sa.shape[0] <= sa.shape[1] else sa.T @ sa
        top_singular_value = get_max_sval_approx(sasa, niter=2)
        threshold = np.sqrt(top_singular_value) * get_reg_param_threshold(sasa)
        sasa_nu = max(nu, threshold)
        if isinstance(sasa, torch.Tensor):
            prefactorization = cholesky(sasa + sasa_nu ** 2 * torch.eye(sasa.shape[0], device=sasa.device, dtype=sasa.dtype), lower=False)
        else:
            prefactorization = cholesky(sasa + sasa_nu ** 2 * np.eye(sasa.shape[0], dtype=sasa.dtype), lower=False)

        q_out.put((mega_iteration, start_idx // n_children, sa, prefactorization, sasa_nu))

        del sa, prefactorization

        for i_ in range(n_children):
            del cache[(mega_iteration, start_idx + i_)]

        start_idx += n_children


def dm_fn(signal_is_alive, q_dm_in, q_dm_out):
    signal_is_alive.set()

    while True:
        try:
            message = q_dm_in.get(timeout=0)
            if message is None:
                del a, b, hessian, x_opt, upper_mat
                break
            elif message[0] == 'torch_data':
                a, b, nu = message[1:]
            elif message[0] == 'sparse_data':
                sp_type, a_data, a_indices, a_indptr, a_shape, b, nu = message[1:]
                a_data = a_data.numpy()
                a_indices = a_indices.numpy()
                a_indptr = a_indptr.numpy()
                if sp_type == 'csr':
                    a = scipy.sparse.csr_matrix((a_data, a_indices, a_indptr), shape=a_shape)
                else:
                    a = scipy.sparse.csc_matrix((a_data, a_indices, a_indptr), shape=a_shape)
                b = b.numpy()
        except queue.Empty:
            continue

        with_torch = isinstance(a, torch.Tensor)

        if not issparse(a):
            if with_torch:
                hessian = a.T @ a + nu ** 2 * torch.eye(a.shape[1], device=a.device, dtype=a.dtype)
            else:
                hessian = a.T @ a + nu ** 2 * np.eye(a.shape[1], dtype=a.dtype)
            upper_mat = cholesky(hessian, lower=False)
            x_opt = solve_triangular(upper_mat, solve_triangular(upper_mat.T, b, lower=True), lower=False)
        else:
            hessian = a.T @ a + nu ** 2 * sparse_identity_matrix(a.shape[1], dtype=a.dtype)
            if not isspmatrix_csc(hessian):
                hessian = hessian.T.tocsc()
            factorization = sparse_factorization(hessian)
            x_opt = factorization(b)

        q_dm_out.put(x_opt)


def cg_fn(signal_is_alive, break_event, q_cg_in, q_cg_out):

    signal_is_alive.set()

    while True:
        try:
            message = q_cg_in.get(timeout=0)
            if message is None or message[0] not in ['torch_data', 'sparse_data']:
                del a, b, cg_solver
                return
            elif message[0] == 'torch_data':
                a, b, nu, cg_params = message[1:]
            elif message[0] == 'sparse_data':
                sp_type, a_data, a_indices, a_indptr, a_shape, b, nu, cg_params = message[1:]
                a_data = a_data.numpy()
                a_indices = a_indices.numpy()
                a_indptr = a_indptr.numpy()
                if sp_type == 'csr':
                    a = scipy.sparse.csr_matrix((a_data, a_indices, a_indptr), shape=a_shape)
                else:
                    a = scipy.sparse.csc_matrix((a_data, a_indices, a_indptr), shape=a_shape)
                b = b.numpy()
        except queue.Empty:
            continue

        break_event.clear()

        cg_fit_params = {'n_iterations': cg_params['n_iterations'],
                         'tolerance': cg_params['tolerance'],
                         'get_full_metrics': cg_params['get_full_metrics'],
                         }

        cg_solver = CG(a, b, nu, x_opt=cg_params['x_opt'], rescale_data=False, check_reg_param=False,
                       least_squares=False, enforce_cuda=(a.is_cuda if isinstance(a, torch.Tensor) else False))
        cg_solver.fit(break_event=break_event, **cg_fit_params)

        q_cg_out.put((cg_solver.x_fit, cg_solver.metrics))

        #break_event.clear()










