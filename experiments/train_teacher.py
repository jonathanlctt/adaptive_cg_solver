import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


class small_mlp(nn.Module):

    def __init__(self, d_h=128, d_in=960, d_out=10):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_h = 128

        self.fc = nn.Sequential(
            nn.Linear(self.d_in, self.d_h),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(self.d_h, self.d_out),
        )

    def forward(self, x):
        return self.fc(x)


def load_data(dataset_name, model_name, root_dir):
    root_dir = Path(root_dir)
    data_dir = root_dir / dataset_name / f"latent_{dataset_name}_{model_name}"

    xb_fnames_train = [f_ for f_ in os.listdir(data_dir / 'train' / 'teacher') if 'xb' in f_]
    yb_fnames_train = [f_.replace('xb', 'yb') for f_ in xb_fnames_train]
    Xtrain = torch.cat([torch.load(data_dir / 'train' / 'teacher' / xf_) for xf_ in xb_fnames_train])
    ytrain = torch.cat([torch.load(data_dir / 'train' / 'teacher' / yf_) for yf_ in yb_fnames_train])

    xb_fnames_test = [f_ for f_ in os.listdir(data_dir / 'test' / 'teacher') if 'xb' in f_]
    yb_fnames_test = [f_.replace('xb', 'yb') for f_ in xb_fnames_test]
    Xtest = torch.cat([torch.load(data_dir / 'test' / 'teacher' / xf_) for xf_ in xb_fnames_test])
    ytest = torch.cat([torch.load(data_dir / 'test' / 'teacher' / yf_) for yf_ in yb_fnames_test])

    print(f"{Xtrain.shape=}, {Xtest.shape=}")

    return Xtrain, ytrain, Xtest, ytest, xb_fnames_train, yb_fnames_train, xb_fnames_test, yb_fnames_test


def train_model(dataset_name, model_name, root_dir):
    d_h = 512
    mini_batch_size = 128
    n_epochs = 200

    if dataset_name == 'cifar10':
        d_out = 10
    elif dataset_name == 'tiny_image_net':
        d_out = 200

    net = small_mlp(d_h=d_h, d_out=d_out)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    Xtrain, ytrain, Xtest, ytest, xb_fnames_train, yb_fnames_train, xb_fnames_test, yb_fnames_test = load_data(
        dataset_name, model_name, root_dir)

    n_samples_train = Xtrain.shape[0]
    n_samples_test = Xtest.shape[0]
    batch_indices_test = [min(jj * mini_batch_size, n_samples_test) for jj in
                          range(n_samples_test // mini_batch_size + 2) if
                          jj * mini_batch_size < n_samples_test + mini_batch_size]
    batch_indices_test = np.array(
        [(batch_indices_test[jj], batch_indices_test[jj + 1]) for jj in range(len(batch_indices_test[:-1]))])

    for epoch in range(n_epochs):
        print(f"{epoch=}")

        batch_indices_train = [min(jj * mini_batch_size, n_samples_train) for jj in
                               range(n_samples_train // mini_batch_size + 2) if
                               jj * mini_batch_size < n_samples_train + mini_batch_size]
        batch_indices_train = np.random.permutation(
            [(batch_indices_train[jj], batch_indices_train[jj + 1]) for jj in range(len(batch_indices_train[:-1]))])

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for ii, batch_index in enumerate(batch_indices_test):
                xb = Xtest[batch_index[0]:batch_index[1]]
                yb = ytest[batch_index[0]:batch_index[1]]
                ypredb = net(xb)
                _, predicted = torch.max(ypredb.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        print(f'Accuracy of the network on the test images: {100 * correct / total} %')

        correct = 0
        total = 0
        with torch.no_grad():
            for ii, batch_index in enumerate(batch_indices_train):
                xb = Xtrain[batch_index[0]:batch_index[1]]
                yb = ytrain[batch_index[0]:batch_index[1]]
                ypredb = net(xb)
                _, predicted = torch.max(ypredb.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        print(f'Accuracy of the network on the train images: {100 * correct / total} %')

        net.train()
        for ii, batch_index in enumerate(batch_indices_train):
            xb = Xtrain[batch_index[0]:batch_index[1]]
            yb = ytrain[batch_index[0]:batch_index[1]]

            optimizer.zero_grad()

            ypred = net(xb)
            loss = criterion(ypred, yb)
            loss.backward()
            optimizer.step()

    print('Finished Training')

    # inference
    logits_test = []
    y_test_ = []
    with torch.no_grad():
        for ii, batch_index in enumerate(batch_indices_test):
            xb = Xtest[batch_index[0]:batch_index[1]]
            yb = ytest[batch_index[0]:batch_index[1]]
            ypredb = net(xb)

            logits_test.append(ypredb)
            y_test_.append(yb)

    batch_indices_train = [min(jj * mini_batch_size, n_samples_train) for jj in
                           range(n_samples_train // mini_batch_size + 2) if
                           jj * mini_batch_size < n_samples_train + mini_batch_size]
    batch_indices_train = [(batch_indices_train[jj], batch_indices_train[jj + 1]) for jj in
                           range(len(batch_indices_train[:-1]))]

    logits_train = []
    y_train_ = []
    with torch.no_grad():
        for ii, batch_index in enumerate(batch_indices_train):
            xb = Xtrain[batch_index[0]:batch_index[1]]
            yb = ytrain[batch_index[0]:batch_index[1]]
            ypredb = net(xb)

            logits_train.append(ypredb)
            y_train_.append(yb)

    logits_train = torch.cat(logits_train, dim=0)
    y_train_ = torch.cat(y_train_, dim=0)
    logits_test = torch.cat(logits_test, dim=0)
    y_test_ = torch.cat(y_test_, dim=0)

    root_dir = Path(root_dir)
    torch.save(logits_train,
               root_dir / dataset_name / f"latent_{dataset_name}_{model_name}" / 'train' / 'teacher' / 'logits_train.pt')
    torch.save(y_train_,
               root_dir / dataset_name / f"latent_{dataset_name}_{model_name}" / 'train' / 'teacher' / 'y_train.pt')
    torch.save(logits_test,
               root_dir / dataset_name / f"latent_{dataset_name}_{model_name}" / 'test' / 'teacher' / 'logits_test.pt')
    torch.save(y_test_,
               root_dir / dataset_name / f"latent_{dataset_name}_{model_name}" / 'test' / 'teacher' / 'y_test.pt')


if __name__ == "__main__":
    dataset_name = 'tiny_image_net'
    model_name = 'mobilenet'
    root_dir = f'/Users/jonathanlacotte/code/datasets'

    train_model(dataset_name, model_name, root_dir)