import os.path as osp

from pytorch_lightning import LightningDataModule
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

from src.datamodules.components.processing_utils import *


class SimplicialTUDataset(InMemoryDataset):
    def __init__(self, root, name, normalize=True):
        self.root = root
        self.name = name
        self.normalize = normalize

        super().__init__(root)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'simplicial')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        self.data_original = TUDataset(root=self.root, name=self.name).shuffle()

    def process(self):
        data_list = []
        for data in tqdm(self.data_original):
            cochain = convert_to_cochain(data)
            if self.normalize:
                cochain = normalize(cochain)
            data_list.append(cochain)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_triangles(self) -> int:
        return self.data.num_triangles.sum().item()

    @property
    def num_edge_features(self) -> int:
        return self.data.x_edge.shape[1]


class TUDataModule(LightningDataModule):
    def __init__(self, root: str, name: str, normalize: bool = True, batch_size: int = 32,
                 num_workers=None, split_sizes: str = '0.8,0.1,0.1'):
        super().__init__()
        self.root = root
        self.name = name
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        # TODO: deal better with this
        self.split_sizes = tuple(map(float, split_sizes.split(',')))


    def setup(self, stage=None):
        self.data = SimplicialTUDataset(root=self.root, name=self.name, normalize=self.normalize)

        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            self.data, self.batch_size, splits=self.split_sizes, num_workers=self.num_workers)

    @property
    def num_classes(self) -> int:
        return self.data.num_classes

    @property
    def in_channels_edges(self) -> int:
        return self.data.num_edge_features

    @property
    def in_channels_nodes(self) -> int:
        return self.data.data_original.num_features

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
