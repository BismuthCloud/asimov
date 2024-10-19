import redis.asyncio
import redis.exceptions
import jsonpickle
from typing import Dict, Any, Optional, Set

from asimov.caches.cache import Cache


RAISE_ON_NONE = object()


class RedisCache(Cache):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    _client: redis.asyncio.Redis

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = redis.asyncio.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
        )

    async def get(self, key: str, default=RAISE_ON_NONE, raw=False):
        modified_key = key

        if not raw:
            modified_key = await self.apply_key_modifications(key)

        value = await self._client.get(modified_key)
        if value is None and default is RAISE_ON_NONE:
            raise KeyError(key)
        return jsonpickle.decode(value) if value is not None else default

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
            try:
                value = await self.get(key.decode("utf-8"), raw=True)
            except redis.exceptions.ResponseError:
                # Attempt to GET a non-normal key, e.g. a mailbox list
                continue
            result[key.decode("utf-8")] = value
        return result

    async def publish_to_mailbox(self, mailbox_id: str, value):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        await self._client.rpush(modified_mailbox_id, jsonpickle.encode(value))  # type: ignore

    async def get_message(self, mailbox_id: str, timeout: Optional[float] = None):
        modified_mailbox_id = await self.apply_key_modifications(mailbox_id)
        res = await self._client.blpop([modified_mailbox_id], timeout=timeout)  # type: ignore
        if res is None:
            return None
        _key, message = res
        return jsonpickle.decode(message)

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
        if self._client:
            await self._client.aclose()
