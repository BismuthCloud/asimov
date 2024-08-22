import asyncio
from collections import defaultdict
from lib.caches.cache import Cache
from pydantic import Field
from typing import Any, DefaultDict, Optional, Dict
from queue import Queue, Empty
from lib.caches.cache import Cache  # Assuming Cache is defined in your project
from pydantic import Field, field_validator
from typing import Any, DefaultDict, Optional, Dict


def create_queue() -> Queue:
    return Queue()


class MockRedisCache(Cache):
    data: dict = Field(default_factory=dict)
    mailboxes: Dict[str, Queue] = Field(default_factory=dict)

    @field_validator("mailboxes", mode="before")
    def set_mailboxes(cls, v):
        return defaultdict(create_queue, v)

    def __init__(self):
        super().__init__()
        self.mailboxes = defaultdict(Queue)

    def clear(self):
        self.data = {}

    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def get_all(self) -> Dict[str, Any]:
        return self.data

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def delete(self, key: str) -> None:
        self.data.pop(key, None)

    def clear(self, prefix: str) -> None:
        keys_to_delete = [key for key in self.data if key.startswith(prefix)]
        for key in keys_to_delete:
            self.delete(key)

    def create_mailbox(self, mailbox_id: str) -> None:
        # No-op for this mock implementation
        pass

    def peek_mailbox(self, mailbox_id: str) -> list:
        return list(self.mailboxes[mailbox_id].queue)

    def peek_message(self, mailbox_id: str) -> str:
        return list(self.mailboxes[mailbox_id].queue)[-1]

    def get_message(self, mailbox: str):
        if not self.mailboxes[mailbox].empty():
            return self.mailboxes[mailbox].get_nowait()
        return None

    def publish_to_mailbox(self, mailbox: str, message: Any):
        self.mailboxes[mailbox].put(message)

    def get_all_messages(self, mailbox: str):
        return list(self.mailboxes[mailbox].queue)

    def subscribe_to_mailbox(self, mailbox_id: str) -> None:
        # In a real Redis instance, this would subscribe to a channel.
        # Here, we simply ensure the mailbox exists in the mock channels.
        self.mailboxes[mailbox_id]

    def unsubscribe_from_mailbox(self, mailbox_id: str) -> None:
        # In a real Redis instance, this would unsubscribe from a channel.
        # Here, we simulate that by removing the channel from our mock.
        if mailbox_id in self.mailboxes:
            del self.mailboxes[mailbox_id]

    def close(self) -> None:
        # No-op for this mock implementation
        pass
