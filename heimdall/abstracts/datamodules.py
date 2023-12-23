from typing import Dict, Any
from abc import abstractmethod, ABC

from torch.utils.data import Dataset
import pytorch_lightning as pl




class Dataset(Dataset, ABC):

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError


class DataModule(pl.LightningDataModule, ABC):

    @abstractmethod
    def prepare_data(self) -> None:
        pass
    
    @abstractmethod
    def setup(self, stage: str = None):
        pass

    @abstractmethod
    def train_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self, batch_size: int = None):
        raise NotImplementedError