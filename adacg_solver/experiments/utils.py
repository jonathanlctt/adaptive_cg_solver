import numpy as np
import pandas as pd
import torch
import rff

from ..datasets.dataloading_utils import load_year_prediction_data, load_california_housing_data, load_rcv1_data, load_wesad_data


def compute_nu(sigma, de, max_nu=8):
    def compute_de(nu):
        return (sigma ** 2 / (sigma ** 2 + nu ** 2)).sum() / (sigma[0] ** 2 / (sigma[0] ** 2 + nu ** 2))

    nu = 1.
    de_ = compute_de(nu)
    while de_ >= de and nu <= max_nu:
        nu = 2 * nu
        de_ = compute_de(nu)
    # binary search
    nu_max = nu
    nu_min = 0.
    nu = (nu_max + nu_min) / 2
    de_ = compute_de(nu)
    iteration = 0
    while np.abs(de_ - de) > 1 and iteration <= 10000:
        if de_ > de:
            nu_min = nu
            nu = (nu_min + nu_max) / 2
            de_ = compute_de(nu)
        else:
            nu_max = nu
            nu = (nu_min + nu_max) / 2
            de_ = compute_de(nu)
        iteration += 1
    return max(1e-10, nu), de_


def make_synthetic_example(n, d, deff, cn, dtype=torch.float32):
    a = 1. / np.sqrt(n) * torch.randn(d, d, dtype=dtype) / np.sqrt(d)
    b = 1. / np.sqrt(n) * torch.randn(n, 1, dtype=dtype)
    _, _, vh = torch.linalg.svd(a, full_matrices=False)

    exponent, eig_val, deff_, cn_, sigma, reg_param = find_exponent_eig_val(d=d, deff_target=deff, cn_target=cn,
                                                                            dtype=dtype)
    sigma = sigma.reshape((-1, 1))

    u = torch.randn(n, d, dtype=dtype) / np.sqrt(n)
    a = u @ (sigma * vh)

    return a, b, reg_param, deff_, cn_


def compute_eig_val_and_reg_param(exponent, d, de, max_nu=8, eig_val=0.999, dtype=torch.float64):
    def compute_de(sigma, nu):
        return (sigma ** 2 / (sigma ** 2 + nu ** 2)).sum() / (sigma[0] ** 2 / (sigma[0] ** 2 + nu ** 2))

    sigma = torch.tensor([max(eig_val ** (exponent * jj), 1e-4) for jj in range(d)], dtype=dtype).reshape((-1, 1))

    nu_max = max_nu
    nu_min = np.sqrt(torch.finfo(dtype).eps)
    nu = (nu_max + nu_min) / 2
    de_ = compute_de(sigma, nu)
    iteration = 0
    while np.abs(de_ - de) > 1 and iteration <= 100:
        if de_ > de:
            nu_min = nu
            nu = (nu_min + nu_max) / 2
            de_ = compute_de(sigma, nu)
        else:
            nu_max = nu
            nu = (nu_min + nu_max) / 2
            de_ = compute_de(sigma, nu)
        iteration += 1

    cn = np.sqrt((sigma[0] ** 2 + nu ** 2) / (sigma[-1] ** 2 + nu ** 2)[0]).item()

    return sigma, nu, de_.item(), cn


def find_exponent_eig_val(d, deff_target, cn_target, dtype=torch.float32):
    min_dist = np.inf
    best_exponent = 1
    best_eig_val = 0.9
    best_cn = 0
    best_deff = 0
    best_reg_param = 0.
    for exponent in range(1, 20):
        for eig_val in [0.9, 0.92, 0.95, 0.96, 0.97, 0.98, 0.99, 0.992, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9992,
                        0.9995, 0.9996, 0.9997, 0.9998, 0.9999]:
            sigma, reg_param, deff, cn = compute_eig_val_and_reg_param(exponent=exponent, eig_val=eig_val, d=d,
                                                                       de=deff_target, dtype=dtype)

            dist_ = np.abs(cn - cn_target) / min(cn, cn_target)
            if dist_ < min_dist:
                min_dist = dist_
                best_exponent = exponent
                best_eig_val = eig_val
                best_deff = deff
                best_cn = cn
                best_reg_param = reg_param
                best_sigma = sigma
            if np.abs(deff - deff_target) < 2. and dist_ < 0.2:
                return exponent, eig_val, deff, cn, sigma, reg_param

    return best_exponent, best_eig_val, best_deff, best_cn, best_sigma, best_reg_param


def load_real_data(dataset_name, encode_data=True):
    if dataset_name == 'year_prediction':
        a, b = load_year_prediction_data()
        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        a -= torch.mean(a, dim=0)
        a /= torch.std(a, dim=0)
        b -= torch.mean(b, dim=0)
        b /= torch.std(b, dim=0)
        if encode_data:
            encoding = rff.layers.GaussianEncoding(sigma=0.001, input_size=a.shape[1], encoded_size=3500)
            a = encoding(a)
        a /= np.sqrt(a.shape[0])
        b /= np.sqrt(a.shape[0])
    elif dataset_name == 'california_housing':
        a, b = load_california_housing_data()
        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        a -= torch.mean(a, dim=0)
        a /= torch.std(a, dim=0)
        b -= torch.mean(b, dim=0)
        b /= torch.std(b, dim=0)
        if encode_data:
            encoding = rff.layers.GaussianEncoding(sigma=0.1, input_size=a.shape[1], encoded_size=3500)
            a = encoding(a)
        a /= np.sqrt(a.shape[0])
        b /= np.sqrt(a.shape[0])
    elif dataset_name == 'rcv1':
        a, b = load_rcv1_data()
    elif dataset_name == 'wesad':
        if encode_data:
            a, b = load_wesad_data(n_samples=500000)
        else:
            a, b = load_wesad_data()
        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        a -= torch.mean(a, dim=0)
        a /= torch.std(a, dim=0)
        b -= torch.mean(b, dim=0)
        b /= torch.std(b, dim=0)
        if encode_data:
            encoding = rff.layers.GaussianEncoding(sigma=0.001, input_size=a.shape[1], encoded_size=3500)
            a = encoding(a)
        a /= np.sqrt(a.shape[0])
        b /= np.sqrt(a.shape[0])

    return a, b


def make_and_write_df(config, errs, gradient_norms, times, iters, sketch_sizes, method, xp_id, xp_dir,
                      handshake_time=0., time_to_send_data=0.):
    df = pd.DataFrame({'errors': errs / errs[0],
                       'gradient_norms': gradient_norms,
                       'time': np.cumsum(times),
                       })
    df['sketch_size'] = sketch_sizes if sketch_sizes is not None else 0
    df['iters'] = iters if iters is not None else 0
    df['handshake_time'] = handshake_time
    df['time_to_send_data'] = time_to_send_data

    for k_ in config:
        df[k_] = config[k_]

    df['method'] = method
    df['xp_id'] = xp_id

    df.to_parquet(xp_dir / f'df_{xp_id}.parquet')