from abc import ABC, abstractmethod
from lib.agents.graph_agent import AgentModule, ModuleConfig, ModuleType
from typing import Any, Dict
from lib.caches.cache import Cache
from pydantic import Field
import asyncio


class Executor(AgentModule, ABC):
    type: ModuleType = Field(default=ModuleType.EXECUTOR)

    @abstractmethod
    async def process(self, cache: Cache, semaphore: asyncio.Semaphore) -> Any:
        raise NotImplementedError
