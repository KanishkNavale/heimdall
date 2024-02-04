import msgpack
import numpy as np
import torch

from heimdall.abstracts.dataschemes import BaseDataClass
from heimdall.datatypes.camera import CameraInformation


class Image(BaseDataClass):
    rgb: np.ndarray | torch.Tensor
    depth: np.ndarray | torch.Tensor
    camera_information: CameraInformation

    def __init__(
        self,
        rgb: np.ndarray | torch.Tensor,
        depth: np.ndarray | torch.Tensor,
        camera_information: CameraInformation,
    ) -> None:
        super().__init__(rgb=rgb, depth=depth, camera_information=camera_information)

    @property
    def compressed_rgb(self) -> np.ndarray | torch.Tensor:
        return self.rgb

    @property
    def compressed_depth(self) -> np.ndarray | torch.Tensor:
        return self.depth

    @property
    def serialized(self) -> bytes:
        info = {
            "rgb": self.compressed_rgb.tolist(),
            "depth": self.compressed_depth.tolist(),
            "camera_information": self.camera_information.as_dictionary,
        }

        return msgpack.packb(info)

    @classmethod
    def deserialize(cls, data: bytes) -> "Image":
        info = msgpack.load(data)
        return cls(
            rgb=np.array(info["rgb"]),
            depth=np.array(info["depth"]),
            camera_information=CameraInformation.from_dictionary(
                info["camera_information"]
            ),
        )
