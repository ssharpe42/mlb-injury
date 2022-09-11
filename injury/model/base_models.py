import os
import pickle

import pytorch_lightning as pl
import torch
from IPython import embed
from torch import nn, optim

from .dataloader import PointProcessDataset


class PointProcessModel(pl.LightningModule):
    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 0.001,
        event_dim: int = None,
        bos: bool = True,
        has_eos: bool = False,
        tmax: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.event_dim = event_dim
        self.bos = bos
        self.has_eos = has_eos
        self.tmax = tmax
        self.lr = lr

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def create_dataset(self, dataset):
        return PointProcessDataset(
            **dataset,
            event_dim=self.event_dim,
            bos=self.bos,
            has_eos=self.has_eos,
            tmax=self.tmax,
        )

    def prepare_data(self, train=None, dev=None, test=None):
        if train is not None:
            self.train_data, self.val_data, self.test_data = train, dev, test

    def setup(self, stage):
        self.train_dataset = self.create_dataset(self.train_data)
        self.val_dataset = self.create_dataset(self.val_data)
        self.test_dataset = self.create_dataset(self.test_data)

    def create_dataloader(self, dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )

    def train_dataloader(self):
        return self.create_dataloader(
            self.train_dataset,
            shuffle=True,
        )

    def val_dataloader(self):
        return self.create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.create_dataloader(self.test_dataset)
