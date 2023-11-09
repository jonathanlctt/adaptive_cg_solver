import numpy as np
import scipy.sparse
import torch
import warnings

warnings.filterwarnings("ignore")


def hadamard_matrix(n, sketch_size, with_torch=True):
    order = (np.int64(np.ceil(np.log(n) / np.log(2))))
    if with_torch:
        h = torch.tensor([[1]], dtype=torch.int32)
        for _ in range(order):
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        signs = 2 * (torch.rand(1, n) > 0.5) - 1
    else:
        h = np.array([[1]], dtype=np.int32)
        for _ in range(order):
            h = np.vstack([np.hstack([h, h ]), np.hstack([h, -h])])
        signs = np.random.choice([-1, 1], n, replace=True)

    ind = np.random.choice(n, sketch_size, replace=False)
    return h[ind, :n] * signs / np.sqrt(sketch_size)


def sjlt(a, sketch_size, nnz_per_column=1):

    rng = np.random.default_rng()

    n = a.shape[0]
    d = a.shape[1]

    with_torch = isinstance(a, torch.Tensor)

    if sketch_size == 0:
        return torch.zeros((sketch_size, d), dtype=a.dtype) if with_torch else np.zeros((sketch_size, d), dtype=a.dtype)

    if not with_torch:
        data = np.random.randn(n) / np.sqrt(nnz_per_column)
        rows = rng.choice(sketch_size, n, replace=True)
        indptr = np.arange(n + 1, dtype=np.int32)
        s = scipy.sparse.csc_matrix((data, rows, indptr), shape=(sketch_size, n))
        sa = s @ a
        return sa
    else:
        data = torch.randn(n, dtype=a.dtype) / np.sqrt(nnz_per_column)
        rows = rng.choice(sketch_size, n, replace=True)
        ccol_indices = torch.arange(n + 1, dtype=torch.int32)
        s = torch.sparse_csc_tensor(ccol_indices, rows, data, dtype=a.dtype, size=(sketch_size, n))
        sa = s @ a
        return sa