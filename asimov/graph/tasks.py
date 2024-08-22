# BISMUTH FILE: tasks.py

from enum import Enum
from uuid import uuid4, UUID
from typing import Dict, Any, Optional
from asimov.asimov_base import AsimovBase
from pydantic import ConfigDict, Field


class TaskStatus(Enum):
    WAITING = "waiting"
    EXECUTING = "executing"
    COMPLETE = "complete"
    FAILED = "failed"
    PARTIAL = "partial_failure"


class Task(AsimovBase):
    id: UUID = Field(default_factory=uuid4)
    type: str
    objective: str
    params: Dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.WAITING
    result: Optional[Any] = None
    error: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_status(self, status: TaskStatus) -> None:
        self.status = status

    def set_result(self, result: Any) -> None:
        self.result = result
        self.status = TaskStatus.COMPLETE

    def set_error(self, error: str) -> None:
        self.error = error
        self.status = TaskStatus.FAILED
