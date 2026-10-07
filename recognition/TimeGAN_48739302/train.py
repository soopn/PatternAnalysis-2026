import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import LOBSTERDataset
from timegan import Discriminator, Embedder, Generator, Recoverer, Supervisor

# level 10 dataset
train_dataset = LOBSTERDataset(
    "./LOBSTER_DATA/LOBSTER_SampleFile_AMZN_2012-06-21_10"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
)
