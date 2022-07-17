import os.path as osp
import pickle

from pytorch_lightning import LightningDataModule
from torch_geometric.data import InMemoryDataset

from src.datamodules.components.processing_utils import *


class TrajectoryDataset(InMemoryDataset):
    def __init__(self, root='data', name='ocean', split='train'):
        self.root = root
        self.name = name
        self.split = split
        super().__init__(self.root)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def raw_file_names(self):
        names = ['coords.pkl', 'data.pkl', 'incidence.pkl']
        return [osp.join(self.raw_dir, name) for name in names]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        data = pickle.load(open(self.raw_file_names[1], "rb"))[self.split]
        incidences = pickle.load(open(self.raw_file_names[2], "rb"))
        B1 = torch.tensor(incidences["B1"], dtype=torch.long)
        B2 = torch.tensor(incidences["B2"], dtype=torch.long)
        lower_index, lower_values = dense_to_sparse(B1.T @ B1)
        upper_index, upper_values = dense_to_sparse(B2 @ B2.T)
        data_list = []
        for X, y in data:
            if self.name == 'ocean':
                X = torch.tensor(X, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.long)
            cochain = CochainData(lower_index, upper_index, lower_values, upper_values, X, y)
            data_list.append(cochain)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class TrajectoryDataModule(LightningDataModule):
    def __init__(self, root, name, batch_size, num_workers=None):
        super().__init__()
        self.root = root
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.ocean_train = TrajectoryDataset(root=self.root, name=self.name, split='train')
        self.ocean_val = TrajectoryDataset(root=self.root, name=self.name, split='val')
        self.ocean_test = TrajectoryDataset(root=self.root, name=self.name, split='test')

    @property
    def in_channels(self):
        return 1

    @property
    def num_classes(self):
        return 2

    def train_dataloader(self):
        return DataLoader(self.ocean_train, batch_size=self.batch_size, shuffle=True, follow_batch=['x_edge'],
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ocean_val, batch_size=self.batch_size, follow_batch=['x_edge'],
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ocean_test, batch_size=self.batch_size, follow_batch=['x_edge'],
                          num_workers=self.num_workers)
