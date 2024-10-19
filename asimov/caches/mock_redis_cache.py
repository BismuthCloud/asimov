import asyncio
from collections import defaultdict
from asimov.caches.cache import Cache
from pydantic import Field
from typing import Any, Optional, Dict, Set
from asimov.caches.cache import Cache
from pydantic import Field, field_validator
import copy


RAISE_ON_NONE = object()


class MockRedisCache(Cache):
    data: dict = Field(default_factory=dict)
    mailboxes: Dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("mailboxes", mode="before")
    def set_mailboxes(cls, v):
        return defaultdict(list, v)

    def __init__(self):
        super().__init__()
        self.mailboxes = defaultdict(list)

    async def clear(self):
        self.data = {}
        self.mailboxes = defaultdict(list)

    async def get(
        self, key: str, default: Any = RAISE_ON_NONE, raw: bool = False
    ) -> Optional[Any]:
        modified_key = key

        if not raw:
            modified_key = await self.apply_key_modifications(key)

        out = self.data.get(modified_key, default)
        if out is RAISE_ON_NONE:
            raise KeyError(key)
        return copy.deepcopy(out)

    async def get_all(self) -> Dict[str, Any]:
        prefix = await self.get_prefix()
        return {k: v for k, v in self.data.items() if k.startswith(prefix)}

    async def set(self, key: str, value: Any, raw: bool = False) -> None:
        modified_key = key

        if not raw:
            modified_key = await self.apply_key_modifications(key)

        self.data[modified_key] = value

    async def delete(self, key: str) -> None:
        modified_key = key
        self.data.pop(modified_key, None)

    async def peek_mailbox(self, mailbox_id: str) -> list:
        return self.mailboxes[mailbox_id][:]

    async def peek_message(self, mailbox_id: str) -> str:
        return self.mailboxes[mailbox_id][0]

    async def get_message(self, mailbox: str, timeout: Optional[float] = None):
        async def _get():
            while True:
                if len(self.mailboxes[mailbox]) > 0:
                    return self.mailboxes[mailbox].pop(0)
                await asyncio.sleep(0.1)

        try:
            return await asyncio.wait_for(_get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def publish_to_mailbox(self, mailbox: str, message: Any):
        self.mailboxes[mailbox].append(message)

    async def keys(self) -> Set[str]:
        prefix = await self.get_prefix()
        suffix = await self.get_suffix()
        return set(
            k for k in self.data.keys() if k.startswith(prefix) and k.endswith(suffix)
        )

    async def close(self) -> None:
        # No-op for this mock implementation
        pass
