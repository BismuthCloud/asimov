from collections import defaultdict
from asimov.caches.cache import Cache
from pydantic import Field
from typing import Any, Optional, Dict, Set
from queue import Queue
from asimov.caches.cache import Cache
from pydantic import Field, field_validator
import copy


def create_queue() -> Queue:
    return Queue()


RAISE_ON_NONE = object()


class MockRedisCache(Cache):
    data: dict = Field(default_factory=dict)
    mailboxes: Dict[str, Queue] = Field(default_factory=dict)

    @field_validator("mailboxes", mode="before")
    def set_mailboxes(cls, v):
        return defaultdict(create_queue, v)

    def __init__(self):
        super().__init__()
        self.mailboxes = defaultdict(Queue)

    async def clear(self):
        self.data = {}

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

    async def clear(self) -> None:
        prefix = await self.get_prefix()
        keys_to_delete = [key for key in self.data if key.startswith(prefix)]
        for key in keys_to_delete:
            await self.delete(key)

    async def peek_mailbox(self, mailbox_id: str) -> list:
        return list(self.mailboxes[mailbox_id].queue)

    async def peek_message(self, mailbox_id: str) -> str:
        return list(self.mailboxes[mailbox_id].queue)[-1]

    async def get_message(self, mailbox: str, timeout=None):
        try:
            return self.mailboxes[mailbox].get(block=True, timeout=timeout)
        except:
            return None

    async def publish_to_mailbox(self, mailbox: str, message: Any):
        self.mailboxes[mailbox].put(message)

    async def get_all_messages(self, mailbox: str):
        return list(self.mailboxes[mailbox].queue)

    async def keys(self) -> Set[str]:
        prefix = await self.get_prefix()
        suffix = await self.get_suffix()
        return set(
            k for k in self.data.keys() if k.startswith(prefix) and k.endswith(suffix)
        )

    async def close(self) -> None:
        # No-op for this mock implementation
        pass
