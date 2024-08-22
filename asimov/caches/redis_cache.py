import redis
import json
import jsonpickle
from typing import Dict, Any, Set
from pydantic import model_validator, PrivateAttr
import threading

from asimov.caches.cache import Cache


def lock_factory():
    return threading.Lock()


class RedisCache(Cache):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    client: redis.Redis | None = None
    pubsub: redis.client.PubSub | None = None
    _lock: threading.Lock = PrivateAttr(default_factory=lock_factory)

    @model_validator(mode="after")
    def set_client(self):
        client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
        )

        self.client = client
        self.pubsub = self.client.pubsub()
        return self

    async def get_message(self, timeout=None):
        with self._lock:
            message = self.pubsub.get_message(timeout=timeout)
            if message and message["type"] == "message":
                return json.loads(message["data"])
        return None

    async def get(self, key: str, default=None, raw=False):

        modified_key = key

        if not raw:
            modified_key = await self.apply_key_modifications(key)

        value = self.client.get(modified_key)
        return jsonpickle.decode(value) if value else default

    async def set(self, key: str, value, raw: bool = False):
        modified_key = key

        if not raw:
            modified_key = await self.apply_key_modifications(key)

        self.client.set(modified_key, jsonpickle.encode(value))

    async def delete(self, key: str):
        modified_key = await self.apply_key_modifications(key)
        self.client.delete(modified_key)

    async def clear(self):
        with self._lock:
            all_keys = self.client.keys("*")
            if all_keys:
                self.client.delete(*all_keys)

    async def get_all(self) -> Dict[str, Any]:
        all_keys = self.client.keys("*")
        result = {}
        for key in all_keys:
            value = await self.get(key.decode("utf-8"), raw=True)
            result[key.decode("utf-8")] = value
        return result

    async def create_mailbox(self, mailbox_id: str):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        self.client.publish(f"mailbox:{modified_mailbox_id}", "")

    async def publish_to_mailbox(self, mailbox_id: str, value):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        self.client.publish(f"mailbox:{modified_mailbox_id}", jsonpickle.encode(value))

    async def subscribe_to_mailbox(self, mailbox_id: str):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        with self._lock:
            self.pubsub.subscribe(f"mailbox:{modified_mailbox_id}")

    async def unsubscribe_from_mailbox(self, mailbox_id: str):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        with self._lock:
            self.pubsub.unsubscribe(f"mailbox:{modified_mailbox_id}")

    async def keys(self) -> Set[str]:
        keys: Set[str] = set()

        context = await self._get_current_context()

        cursor = 0
        prefix = await self.get_prefix(context)
        suffix = await self.get_suffix(context)
        key_string = f"*"

        if prefix:
            key_string = f"{prefix}{self.affix_sep}{key_string}"
        if suffix:
            key_string = f"{key_string}{self.affix_sep}{suffix}"

        while True:
            cursor, partial_keys = self.client.scan(
                cursor=cursor, match=key_string, count=1000
            )

            keys.update([k.decode("utf-8") for k in partial_keys])
            if cursor == 0:
                break

        return keys

    async def close(self):
        with self._lock:
            if self.pubsub:
                self.pubsub.close()
            if self.client:
                self.client.close()
