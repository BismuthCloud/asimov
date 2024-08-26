import redis.asyncio as redis
import json
import jsonpickle
from typing import Dict, Any, Set

from asimov.caches.cache import Cache


class RedisCache(Cache):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    _client: redis.Redis
    _pubsub: redis.client.PubSub
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
        )
        self._pubsub = self._client.pubsub()
    
    async def get_message(self, timeout=None):
        message = await self._pubsub.get_message(timeout=timeout)
        if message and message["type"] == "message":
            return json.loads(message["data"])
        return None

    async def get(self, key: str, default=None, raw=False):
        modified_key = key

        if not raw:
            modified_key = await self.apply_key_modifications(key)

        value = await self._client.get(modified_key)
        return jsonpickle.decode(value) if value else default

    async def set(self, key: str, value, raw: bool = False):
        modified_key = key

        if not raw:
            modified_key = await self.apply_key_modifications(key)

        await self._client.set(modified_key, jsonpickle.encode(value))

    async def delete(self, key: str):
        modified_key = await self.apply_key_modifications(key)
        await self._client.delete(modified_key)

    async def clear(self):
        prefix = await self.get_prefix()
        all_keys = await self._client.keys(f"{prefix}{self.affix_sep}*")
        if all_keys:
            await self._client.delete(*all_keys)

    async def get_all(self) -> Dict[str, Any]:
        prefix = await self.get_prefix()
        all_keys = await self._client.keys(f"{prefix}{self.affix_sep}*")
        result = {}
        for key in all_keys:
            value = await self.get(key.decode("utf-8"), raw=True)
            result[key.decode("utf-8")] = value
        return result

    async def create_mailbox(self, mailbox_id: str):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        await self._client.publish(f"mailbox:{modified_mailbox_id}", "")

    async def publish_to_mailbox(self, mailbox_id: str, value):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        await self._client.publish(f"mailbox:{modified_mailbox_id}", jsonpickle.encode(value))

    async def subscribe_to_mailbox(self, mailbox_id: str):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        await self._pubsub.subscribe(f"mailbox:{modified_mailbox_id}")

    async def unsubscribe_from_mailbox(self, mailbox_id: str):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        await self._pubsub.unsubscribe(f"mailbox:{modified_mailbox_id}")

    async def keys(self) -> Set[str]:
        keys: Set[str] = set()

        cursor = 0
        prefix = await self.get_prefix()
        suffix = await self.get_suffix()
        key_string = f"*"

        if prefix:
            key_string = f"{prefix}{self.affix_sep}{key_string}"
        if suffix:
            key_string = f"{key_string}{self.affix_sep}{suffix}"

        while True:
            cursor, partial_keys = await self._client.scan(
                cursor=cursor, match=key_string, count=1000
            )

            keys.update([k.decode("utf-8") for k in partial_keys])
            if cursor == 0:
                break

        return keys

    async def close(self):
        if self._pubsub:
            await self._pubsub.close()
        if self._client:
            await self._client.aclose()
