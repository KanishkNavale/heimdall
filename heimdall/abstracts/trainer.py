from abc import abstractmethod, ABC, abstractproperty

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from heimdall.abstracts.mlp import MLP
from heimdall.abstracts.datamodules import DataModule

class Trainer(ABC):

    @abstractproperty
    def precision(self, precision:str) -> None:
        torch.set_float32_matmul_precision(precision)

    @abstractproperty
    def lr_monitor(self) -> LearningRateMonitor:
        raise NotImplementedError

    @abstractproperty
    def progress_bar(self) -> RichProgressBar:
        raise NotImplementedError

    @abstractproperty
    def logger(self) -> TensorBoardLogger:
        pass
    
    @abstractproperty
    def checkpoints(self) -> ModelCheckpoint:
        raise NotImplementedError

    @abstractproperty
    def model(self) -> pl.LightningModule | torch.nn.Module | MLP:
        raise NotImplementedError
    
    @abstractproperty
    def datamodule(self) -> DataModule:
        raise NotImplementedError
    
    @abstractproperty
    def trainer(self) -> pl.Trainer:
        raise NotImplementedError
    
    @abstractmethod
    def fit(self) -> None:
        self.trainer.fit(self.model, datamodule=self.datamodule)