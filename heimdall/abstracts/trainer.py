from abc import ABC, abstractmethod, abstractproperty
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from heimdall.abstracts.datamodules import DataModule
from heimdall.abstracts.mlp import MLP
from heimdall.configurations.trainer import TrainerConfig


class Trainer(ABC):
    @abstractproperty
    def config(self) -> TrainerConfig:
        raise NotImplementedError

    @abstractproperty
    def model(self) -> pl.LightningModule | torch.nn.Module | MLP:
        raise NotImplementedError

    @abstractproperty
    def datamodule(self) -> DataModule:
        raise NotImplementedError

    def precision(self) -> None:
        torch.set_float32_matmul_precision(self.config.precision)

    def lr_monitor(self) -> LearningRateMonitor:
        return LearningRateMonitor(logging_interval="step")

    def progress_bar(self) -> RichProgressBar:
        return RichProgressBar()

    def logger(self) -> Optional[TensorBoardLogger]:
        if self.config.enable_logging:
            return TensorBoardLogger(
                save_dir=self.config.tensorboard_path,
                name=self.config.models_checkpoint_name,
            )
        else:
            return None

    def model_checkpoints(self) -> ModelCheckpoint:
        return ModelCheckpoint(
            monitor="val_loss",
            filename=self.config.models_checkpoint_name,
            dirpath=self.config.models_path,
            mode="min",
            save_top_k=1,
            verbose=True,
        )

    def trainer(self) -> pl.Trainer:
        return pl.Trainer(
            logger=self.logger,
            enable_checkpointing=self.config.enable_checkpointing,
            check_val_every_n_epoch=self.config.validation_frequency,
            max_epochs=self.config.epochs,
            log_every_n_steps=self.config.logging_frequency,
            default_root_dir=self.config.training_directory,
            callbacks=[self.model_checkpoints, self.progress_bar, self.lr_monitor],
            enable_model_summary=True,
            detect_anomaly=True,
            accelerator="auto",
            devices="auto",
            inference_mode=False,
        )

    @abstractmethod
    def fit(self) -> None:
        self.trainer.fit(self.model, datamodule=self.datamodule)
