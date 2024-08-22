from pydantic import ConfigDict, BaseModel
from abc import ABC


class AsimovBase(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pass
