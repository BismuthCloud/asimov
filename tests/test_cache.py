import pytest
import pytest_asyncio
from asimov.caches.redis_cache import RedisCache


@pytest_asyncio.fixture
async def redis_cache():
    cache = RedisCache(default_prefix="test")
    await cache.clear()
    try:
        yield cache
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_mailbox(redis_cache):
    MBOX = "test_mailbox"

    assert (await redis_cache.get_message(MBOX, timeout=0.5)) is None
    await redis_cache.publish_to_mailbox(MBOX, {"k": "v"})
    assert (await redis_cache.get_message(MBOX, timeout=0.5)) == {"k": "v"}


@pytest.mark.asyncio
async def test_get_all(redis_cache):
    """
    Mainly just regression test that get_all works with (i.e. excludes) list types used for mailboxes
    """
    prefix = redis_cache.default_prefix

    await redis_cache.publish_to_mailbox("mailbox", {"k": "v"})
    await redis_cache.set("key1", "value1")
    await redis_cache.set("key2", 2)
    assert (await redis_cache.get_all()) == {
        f"{prefix}:key1": "value1",
        f"{prefix}:key2": 2,
    }
