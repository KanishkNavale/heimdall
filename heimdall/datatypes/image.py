import numpy as np
import torch

from heimdall.abstracts.dataschemes import BaseDataClass
from heimdall.datatypes.camera import CameraInformation


class Image(BaseDataClass):
    rgb: np.ndarray | torch.Tensor
    depth: np.ndarray | torch.Tensor
    camera_information: CameraInformation
