from typing import Callable

import kornia
import torch

from heimdall.abstracts.dataschemes import BaseDataClass


class SO3(BaseDataClass):
    matrix: torch.Tensor

    def __post_init__(self, *kwargs) -> None:
        if self.matrix.ndim > 3 or self.matrix.ndim < 2:
            raise ValueError("SO3 matrix must be of shape (3, 3) or (N, 3, 3))")

    @staticmethod
    def _convert(x: torch.Tensor, function: Callable) -> torch.Tensor:
        if x.ndim == 2:
            return function(x[None, ...].contiguous()).squeeze(0)
        return function(x)

    @property
    def quaternion(self) -> torch.Tensor:
        return self._convert(self.matrix, kornia.geometry.rotation_matrix_to_quaternion)

    @property
    def angle_axis(self) -> torch.Tensor:
        return self._convert(self.matrix, kornia.geometry.rotation_matrix_to_angle_axis)

    @property
    def rotation_matrix(self) -> torch.Tensor:
        return self.matrix

    @classmethod
    def from_quaternion(cls: "SO3", quaternion: torch.Tensor) -> "SO3":
        return cls(
            matrix=cls._convert(
                quaternion, kornia.geometry.quaternion_to_rotation_matrix
            )
        )

    @classmethod
    def from_angle_axis(cls: "SO3", angle_axis: torch.Tensor) -> "SO3":
        return cls(
            matrix=cls._convert(
                angle_axis, kornia.geometry.angle_axis_to_rotation_matrix
            )
        )


class Pose(BaseDataClass):
    translation: torch.Tensor
    SO3: SO3

    def __post_init__(self, *kwargs) -> None:
        if self.translation.ndim > 2 or self.translation.ndim < 1:
            raise ValueError("Translation vector must be of shape (3,) or (N, 3)")

        if self.SO3.matrix.ndim > 3 or self.SO3.matrix.ndim < 2:
            raise ValueError("SO3 matrix must be of shape (3, 3) or (N, 3, 3))")

        if self.translation.ndim == 2 and self.SO3.matrix.ndim == 3:
            if self.translation.shape[0] != self.SO3.matrix.shape[0]:
                raise ValueError("Translation and SO3 must have same batch dimension")

        self.translation = self.translation[..., None]

    @property
    def inverted(self) -> "Pose":
        return Pose(
            translation=-torch.matmul(self.SO3.matrix, self.translation).squeeze(-1),
            SO3=SO3(matrix=self.SO3.matrix.transpose(-1, -2)),
        )

    @property
    def SE3(self) -> torch.Tensor:
        if self.translation.ndim == 3 and self.SO3.matrix.ndim == 3:
            return kornia.geometry.Rt_to_matrix4x4(self.SO3.matrix, self.translation)
        else:
            return kornia.geometry.Rt_to_matrix4x4(
                self.SO3.matrix[None, ...], self.translation[None, ...]
            ).squeeze(0)

    @classmethod
    def from_translation_and_rotation_matrix(
        cls: "Pose", translation: torch.Tensor, rotation_matrix: torch.Tensor
    ) -> "Pose":
        return cls(translation=translation, SO3=SO3(matrix=rotation_matrix))

    @classmethod
    def from_translation_and_quaternion(
        cls: "Pose", translation: torch.Tensor, quaternion: torch.Tensor
    ) -> "Pose":
        return cls(
            translation=translation, SO3=SO3.from_quaternion(quaternion=quaternion)
        )

    @classmethod
    def from_translation_and_angle_axis(
        cls: "Pose", translation: torch.Tensor, angle_axis: torch.Tensor
    ) -> "Pose":
        return cls(
            translation=translation, SO3=SO3.from_angle_axis(angle_axis=angle_axis)
        )

    @classmethod
    def from_SE3(cls: "Pose", SE3: torch.Tensor) -> "Pose":
        if SE3.ndim == 3:
            return cls.from_translation_and_rotation_matrix(
                SE3[:, :3, 3], SE3[:, :3, :3]
            )
        else:
            return cls.from_translation_and_rotation_matrix(SE3[:3, 3], SE3[:3, :3])
