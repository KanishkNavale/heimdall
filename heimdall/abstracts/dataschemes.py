from pydantic import BaseModel
from typing import TypeVar

T = TypeVar("T", bound="BaseDataClass")


class BaseDataClass(BaseModel):
    class Config:
        extra = "forbid"
        validate_assignment = True
        arbitrary_types_allowed = True

    @property
    def as_dictionary(self) -> dict:
        return self.model_dump()

    @property
    def as_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_dictionary(cls: T, dictionary: dict) -> T:
        return cls(**dictionary)

    def __post_init__(self, *kwargs) -> None:
        pass

    def model_post_init(self, *kwargs) -> None:
        return self.__post_init__(*kwargs)
