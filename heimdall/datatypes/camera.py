import numpy as np

from heimdall.abstracts.dataschemes import BaseDataClass
from heimdall.datatypes.pose import Pose


class CameraIntrinsics(BaseDataClass):
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy])


class CameraInformation(BaseDataClass):
    intrinsics: CameraIntrinsics
    pose: Pose
