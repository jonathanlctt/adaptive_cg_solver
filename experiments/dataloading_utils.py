import os
from pathlib import Path
import requests
import zipfile
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_rcv1, fetch_california_housing
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn

import rff


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def download_data_with_url(root_dir=f'/Users/jonathanlacotte/code/datasets',
                           dataset_name="yearpredictionmsd",
                           dataset_base_url="https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"):
    filename_txt = f"{dataset_name}.txt"
    filename_csv = f"{dataset_name}.csv"

    root_dir = Path(root_dir)

    if not os.path.isfile(root_dir / filename_csv):
        os.mkdir(root_dir, exist_ok=True)
        zip_path = root_dir / f"{filename_txt}.zip"

        download_url(dataset_base_url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root_dir)
        os.remove(zip_path)

        file_path = root_dir / filename_txt
        os.rename(file_path, file_path[:-4] + ".csv")


def load_year_prediction_data(path="/Users/jonathanlacotte/code/datasets/yearpredictionmsd.csv", n_samples=515345):
    if n_samples > 0:
        df = pd.read_csv(path, header=None, nrows=n_samples)
    else:
        df = pd.read_csv(path, header=None, nrows=515345)
    y = df.iloc[:, 0].values
    y = y.astype(np.float64)
    X = df.iloc[:, 1:].values

    return X, y


def load_california_housing_data():
    data = fetch_california_housing()
    X, y = data['data'], data['target']
    y = y.reshape((-1, 1))

    return X, y


def _y_to_unique_labels(y, names):
    names = np.array(names)
    y_names = []
    for row in y:
        indices = np.flatnonzero(row)
        y_names.append("".join(names[indices]))
    return y_names


def load_rcv1_data(n_samples=804414):

    data = fetch_rcv1()
    X, y = data['data'], data['target']
    names = np.array(data.target_names.tolist())

    labels = _y_to_unique_labels(y.toarray(), names)

    le = LabelEncoder()
    integers = le.fit_transform(labels)
    y = integers.astype(np.float64)

    y = y.reshape(-1, 1)

    return X[:n_samples], y[:n_samples]


def process_wesad_data(root_dir=f'/Users/jonathanlacotte/code/datasets'):
    import pickle

    root_dir = Path(root_dir)

    for patient in [f"S{jj}" for jj in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]:
        with open(root_dir / f'WESAD/{patient}/{patient}.pkl', 'rb') as f:
            data = pickle.load(f, encoding='latin1')

            X = np.hstack([data['signal']['chest'][k_] for k_ in data['signal']['chest']])
            mask_ = np.any(np.array([data['label'] == v_ for v_ in [1, 2, 3]]).transpose(), axis=1)
            X = X[mask_]
            y = data['label'][mask_]

            np.save(root_dir / f'WESAD/{patient}/X.npy', X)
            np.save(root_dir / f'WESAD/{patient}/y.npy', y)


def load_wesad_data(root_dir=f'/Users/jonathanlacotte/code/datasets',
                    n_samples=-1):

    root_dir = Path(root_dir)

    Xs = []
    ys = []

    for patient in [f"S{jj}" for jj in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]:
        Xs.append(np.load(root_dir / f'WESAD/{patient}/X.npy'))
        ys.append(np.load(root_dir / f'WESAD/{patient}/y.npy').reshape((-1, 1)))

    X = np.vstack(Xs)
    y = np.vstack(ys)
    y = 2 * np.hstack([y == 1, y == 2, y == 3], dtype=np.float32) - 1
    if n_samples > 0:
        rdn_idx = np.random.choice(X.shape[0], n_samples, replace=False)
        return X[rdn_idx], y[rdn_idx]
    else:
        return X, y


def process_image_data(root_dir=f'/Users/jonathanlacotte/code/datasets',
                       dataset_name='cifar10',
                       model_name='mobilenet',
                       train=True,
                       student=True):
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, mobilenet_v3_small, \
        MobileNet_V3_Small_Weights

    root_dir = Path(root_dir)
    t_sfx = 'train' if train else 'test'
    s_sfx = 'student' if student else 'teacher'

    if student:
        output_size = (4, 4)
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=output_size)
        model.classifier = Identity()
    else:
        output_size = 1
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = mobilenet_v3_large(weights=weights)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=output_size)
        model.classifier = Identity()

    model.eval()

    preprocess = weights.transforms()
    transformation = transforms.Compose([transforms.ToTensor(), preprocess])
    if dataset_name == 'tiny_image_net':
        t_sfx_ = 'val' if not train else 'train'
        dataset = torchvision.datasets.ImageFolder(root=root_dir / dataset_name / f'tiny-imagenet-200/{t_sfx_}',
                                                   transform=transformation)
    elif dataset_name == 'cifar10':
        os.makedirs(root_dir / dataset_name, exist_ok=True)
        dataset = torchvision.datasets.CIFAR10(root_dir / dataset_name, train=train, download=True,
                                               transform=preprocess)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=0,
        shuffle=False
    )

    latent_dir = root_dir / dataset_name / f'latent_{dataset_name}_{model_name}' / t_sfx / s_sfx
    os.makedirs(latent_dir, exist_ok=True)

    for n_, (x, y) in enumerate(data_loader):
        if n_ % 10 == 0:
            print(f"{n_=}")
        x_latent = model(x).detach()
        torch.save(x_latent, latent_dir / f'xb{n_}.pt')
        torch.save(y, latent_dir / f'yb{n_}.pt')


def tiny_image_net_util_fn():
    import shutil
    data_dir = Path('/Users/jonathanlacotte/code/datasets/tiny_image_net/tiny-imagenet-200/tiny-imagenet-200/test')
    data = pd.read_csv(data_dir / f'val_annotations.txt', sep="\t", header=None)
    data = data.rename(columns={0: 'file_name', 1: 'class'})
    data = data[['file_name', 'class']]
    for class_ in np.unique(data['class']):
        os.makedirs(data_dir / class_, exist_ok=True)
        data_c = data[data['class'] == class_]
        for file_n in data_c['file_name']:
            shutil.copy(data_dir / "images" / file_n, data_dir / class_ / file_n)


def load_real_data(dataset_name, encode_data=True, with_torch=True, dtype=torch.float64,
                   encoded_size=3500,
                   n_samples=-1, n_columns=-1):

    if dataset_name == 'year_prediction':
        a, b = load_year_prediction_data(n_samples=n_samples)
        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        a -= torch.mean(a, dim=0)
        a /= torch.std(a, dim=0)
        b -= torch.mean(b, dim=0)
        b /= torch.std(b, dim=0)
        if encode_data:
            encoding = rff.layers.GaussianEncoding(sigma=0.001,
                                                   input_size=a.shape[1],
                                                   encoded_size=encoded_size)
            a = encoding(a)
        if with_torch:
            a = torch.tensor(a, dtype=dtype)
            b = torch.tensor(b, dtype=dtype)
        else:
            a = a.numpy().astype(dtype)
            b = b.numpy().astype(dtype)
        a /= np.sqrt(a.shape[0])
        b /= np.sqrt(a.shape[0])
        a = a[:n_samples, :n_columns]
        b = b[:n_samples]
    elif dataset_name == 'california_housing':
        a, b = load_california_housing_data()
        a = a[:n_samples]
        b = b[:n_samples]
        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        a -= torch.mean(a, dim=0)
        a /= torch.std(a, dim=0)
        b -= torch.mean(b, dim=0)
        b /= torch.std(b, dim=0)
        if encode_data:
            encoding = rff.layers.GaussianEncoding(sigma=0.1,
                                                   input_size=a.shape[1],
                                                   encoded_size=encoded_size)
            a = encoding(a)
        if with_torch:
            a = torch.tensor(a, dtype=dtype)
            b = torch.tensor(b, dtype=dtype)
        else:
            a = a.numpy().astype(dtype)
            b = b.numpy().astype(dtype)
        a /= np.sqrt(a.shape[0])
        b /= np.sqrt(a.shape[0])
        a = a[:, :n_columns]
    elif dataset_name == 'rcv1':
        a, b = load_rcv1_data()
        a = a[:n_samples]
        b = b[:n_samples]
        b -= np.mean(b, axis=0)
        b /= np.std(b, axis=0)
        a /= np.sqrt(a.shape[0])
        b /= np.sqrt(a.shape[0])
        a = a[:, :n_columns]
    elif dataset_name == 'wesad':
        a, b = load_wesad_data(n_samples=n_samples)
        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        a -= torch.mean(a, dim=0)
        a /= torch.std(a, dim=0)
        b -= torch.mean(b, dim=0)
        b /= torch.std(b, dim=0)
        if encode_data:
            encoding = rff.layers.GaussianEncoding(sigma=0.001,
                                                   input_size=a.shape[1],
                                                   encoded_size=encoded_size,
                                                   )
            a = encoding(a)
        if with_torch:
            a = torch.tensor(a, dtype=dtype)
            b = torch.tensor(b, dtype=dtype)
        else:
            a = a.numpy().astype(dtype)
            b = b.numpy().astype(dtype)
        a /= np.sqrt(a.shape[0])
        b /= np.sqrt(a.shape[0])
        a = a[:, :n_columns]

    return a, b


if __name__ == "__main__":

    process_image_data(model_name='mobilenet', dataset_name='cifar10', train=True, student=False)
    process_image_data(model_name='mobilenet', dataset_name='tiny_image_net', train=True, student=False)
    process_image_data(model_name='mobilenet', dataset_name='tiny_image_net', train=False, student=False)
