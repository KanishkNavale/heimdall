from abc import abstractmethod, ABC

import torch
import pytorch_lightning as pl


class MLP(pl.LightningModule, ABC):

    @abstractmethod
    def configure_optimizers(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    @abstractmethod
    def lr_scheduler_step(self, scheduler, optimizer_idx: int, metric: None) -> None:
        pass
    
    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError