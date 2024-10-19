from abc import ABC, abstractmethod
from typing import Any
from pydantic import Field
import asyncio

from asimov.caches.cache import Cache
from asimov.graph import AgentModule, ModuleType


class Executor(AgentModule, ABC):
    type: ModuleType = Field(default=ModuleType.EXECUTOR)

    @abstractmethod
    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> dict[str, Any]:
        raise NotImplementedError
