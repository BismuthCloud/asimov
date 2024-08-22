from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Set, Optional, List
from pydantic import ConfigDict, Field, PrivateAttr
from asimov.asimov_base import AsimovBase
from contextlib import asynccontextmanager
import jsonpickle
import asyncio


class Cache(AsimovBase, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    default_prefix: str = Field(default="")
    default_suffix: str = Field(default="")
    affix_sep: str = ":"
    _task_context_stacks: Dict[asyncio.Task, List[str]] = PrivateAttr(
        default_factory=dict
    )
    DEFAULT_CONTEXT: str = "__default__"
    _context_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def _get_current_context(self) -> str:
        async with self._context_lock:
            task = asyncio.current_task()
            stack = self._task_context_stacks.get(task, [])
            return stack[-1] if stack else self.DEFAULT_CONTEXT

    async def get_prefix(self, context: Optional[str] = None) -> str:
        if context is None:
            context = await self._get_current_context()
        return await self.get(
            f"{self.default_prefix}{self.affix_sep}{context}{self.affix_sep}prefix",
            self.default_prefix,
            raw=True,
        )

    async def get_suffix(self, context: Optional[str] = None) -> str:
        if context is None:
            context = await self._get_current_context()
        return await self.get(
            f"{self.default_prefix}{self.affix_sep}{context}{self.affix_sep}suffix",
            self.default_suffix,
            raw=True,
        )

    async def apply_key_modifications(self, key: str) -> str:
        context = await self._get_current_context()
        prefix = await self.get_prefix(context)
        suffix = await self.get_suffix(context)

        if prefix:
            key = f"{prefix}{self.affix_sep}{key}"
        if suffix:
            key = f"{key}{self.affix_sep}{suffix}"
        return key

    @asynccontextmanager
    async def with_context(self, context: str):
        async with self._context_lock:
            task = asyncio.current_task()
            if task not in self._task_context_stacks:
                self._task_context_stacks[task] = []
            self._task_context_stacks[task].append(context)
        try:
            yield self
        finally:
            async with self._context_lock:
                self._task_context_stacks[task].pop()
                if not self._task_context_stacks[task]:
                    del self._task_context_stacks[task]

    @asynccontextmanager
    async def with_prefix(self, prefix: str, context: Optional[str] = None):
        if context is not None:
            async with self.with_context(context):
                async with self._with_prefix_internal(prefix):
                    yield self
        else:
            async with self._with_prefix_internal(prefix):
                yield self

    @asynccontextmanager
    async def _with_prefix_internal(self, prefix: str):
        context = await self._get_current_context()
        old_prefix = await self.get_prefix(context)
        await self.set(
            f"{self.default_prefix}{self.affix_sep}{context}{self.affix_sep}prefix",
            prefix,
            raw=True,
        )
        try:
            yield self
        finally:
            await self.set(
                f"{self.default_prefix}{self.affix_sep}{context}{self.affix_sep}prefix",
                old_prefix,
                raw=True,
            )

    @asynccontextmanager
    async def with_suffix(self, suffix: str, context: Optional[str] = None):
        if context is not None:
            async with self.with_context(context):
                async with self._with_suffix_internal(suffix):
                    yield self
        else:
            async with self._with_suffix_internal(suffix):
                yield self

    @asynccontextmanager
    async def _with_suffix_internal(self, suffix: str):
        context = await self._get_current_context()
        old_suffix = await self.get_suffix(context)
        await self.set(
            f"{self.default_prefix}{self.affix_sep}{context}{self.affix_sep}suffix",
            suffix,
            raw=True,
        )
        try:
            yield self
        finally:
            await self.set(
                f"{self.default_prefix}{self.affix_sep}{context}{self.affix_sep}suffix",
                old_suffix,
                raw=True,
            )

    def __getitem__(self, key: str):
        return self.get(key)

    @abstractmethod
    async def get(self, key: str, default: Optional[any] = None, raw: bool = False):
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
    async def create_mailbox(self, mailbox_id: str):
        pass

    @abstractmethod
    async def publish_to_mailbox(self, mailbox_id: str, value):
        pass

    @abstractmethod
    async def subscribe_to_mailbox(self, mailbox_id: str):
        pass

    @abstractmethod
    async def unsubscribe_from_mailbox(self, mailbox_id: str):
        pass

    @abstractmethod
    async def close(self):
        pass

    @abstractmethod
    async def keys(self) -> Set[str]:
        pass
