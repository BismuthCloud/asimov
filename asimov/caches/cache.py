from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Dict, Any, Set, Optional
from pydantic import ConfigDict, Field
from asimov.asimov_base import AsimovBase
from contextlib import asynccontextmanager


class Cache(AsimovBase, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # TODO: this is treated more like a "namespace" - perhaps rename?
    default_prefix: str = Field(default="")
    default_suffix: str = Field(default="")
    affix_sep: str = ":"
    _prefix: ContextVar[str]
    _suffix: ContextVar[str]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prefix = ContextVar("prefix", default=self.default_prefix)
        self._suffix = ContextVar("suffix", default=self.default_suffix)

    async def get_prefix(self) -> str:
        return self._prefix.get()

    async def get_suffix(self) -> str:
        return self._suffix.get()

    async def apply_key_modifications(self, key: str) -> str:
        prefix = await self.get_prefix()
        suffix = await self.get_suffix()

        if prefix:
            key = f"{prefix}{self.affix_sep}{key}"
        if suffix:
            key = f"{key}{self.affix_sep}{suffix}"
        return key

    @asynccontextmanager
    async def with_prefix(self, prefix: str):
        old_prefix = await self.get_prefix()
        self._prefix.set(prefix)
        try:
            yield self
        finally:
            self._prefix.set(old_prefix)

    @asynccontextmanager
    async def with_suffix(self, suffix: str):
        old_suffix = await self.get_suffix()
        self._suffix.set(suffix)
        try:
            yield self
        finally:
            self._suffix.set(old_suffix)

    def __getitem__(self, key: str):
        return self.get(key)

    @abstractmethod
    async def get(self, key: str, default: Optional[Any] = None, raw: bool = False):
        pass

    @abstractmethod
    async def set(self, key: str, value, raw: bool = False):
        pass

    @abstractmethod
    async def delete(self, key: str):
        pass

    @abstractmethod
    async def clear(self):
        pass

    @abstractmethod
    async def get_all(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def publish_to_mailbox(self, mailbox_id: str, value):
        pass

    @abstractmethod
    async def get_message(self, mailbox: str, timeout: Optional[float] = None):
        pass

    @abstractmethod
    async def close(self):
        pass

    @abstractmethod
    async def keys(self) -> Set[str]:
        pass
