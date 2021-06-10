#!/usr/bin/env python3

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from my_project.model import MyModel

train_loader = DataLoader(
    MNIST("~/data", download=True, transform=transforms.ToTensor())
)
logger = TensorBoardLogger("results", "my-model")
trainer = pl.Trainer(logger=logger)
model = MyModel()
trainer.fit(model, train_loader)
