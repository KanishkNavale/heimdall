from typing import Optional

import numpy as np

from heimdall.abstracts.dataschemes import BaseDataClass
from heimdall.datatypes.pose import Pose


class CameraIntrinsics(BaseDataClass):
    fx: float
    fy: float
    cx: float
    cy: float

    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        super().__init__(fx=fx, fy=fy, cx=cx, cy=cy)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy])


class CameraInformation(BaseDataClass):
    intrinsics: CameraIntrinsics
    pose: Optional[Pose] = None

    def __init__(
        self, intrinsics: CameraIntrinsics, pose: Optional[Pose] = None
    ) -> None:
        super().__init__(intrinsics=intrinsics, pose=pose)
