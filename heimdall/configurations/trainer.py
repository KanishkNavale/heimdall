import os

from heimdall.abstracts.dataschemes import BaseDataClass


class TrainerConfig(BaseDataClass):
    precision: str
    model_checkpoint_name: str
    epochs: int
    enable_logging: bool
    tensorboard_path: bool
    training_directory: str
    enable_checkpointing: bool
    model_path: str
    logging_frequency: bool
    validation_frequency: int

    @staticmethod
    def _create_dir(path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

    def __post_init__(self) -> None:
        self.training_directory = os.path.abspath(self.training_directory)
        self.model_path = os.path.abspath(self.model_path)
        self.tensorboard_path = os.path.abspath(self.tensorboard_path)

        if self.precision not in ["medium", "high"]:
            raise ValueError("Precision must be either {medium, high}")

        self._create_dir(self.training_directory)
        self._create_dir(self.tensorboard_path)
