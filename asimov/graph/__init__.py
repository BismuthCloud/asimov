import asyncio
from enum import Enum
import jsonpickle
import os
import pathlib
import pickle
import re
import logging
import opentelemetry.trace
from typing import List, Dict, Any, AsyncGenerator, Optional
from typing_extensions import TypedDict
from pydantic import (
    Field,
    PrivateAttr,
    model_validator,
)
import traceback

from lupa import LuaRuntime
from collections import defaultdict

from asimov.graph.tasks import Task, TaskStatus
from asimov.caches.cache import Cache
from asimov.asimov_base import AsimovBase
from asimov.caches.redis_cache import RedisCache

lua = LuaRuntime(unpack_returned_tuples=True)  # type: ignore[call-arg]

tracer = opentelemetry.trace.get_tracer("asimov")


class TasksFinished(Exception):
    """Exception raised when we want to stop after one success."""

    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)


class ModuleType(Enum):
    EXECUTOR = "executor"
    SUBGRAPH = "subgraph"
    FLOW_CONTROL = "flow_control"


class Middleware(AsimovBase):
    async def process(self, data: Dict[str, Any], cache: Cache) -> Dict[str, Any]:
        return data


class ModuleConfig(AsimovBase):
    stop_on_success: bool = False
    middlewares: List[Middleware] = Field(default_factory=list)
    timeout: float = 60.0
    context: Dict[str, Any] = Field(default_factory=dict)


class NodeConfig(AsimovBase):
    parallel: bool = False
    condition: Optional[str] = None
    retry_on_failure: bool = True
    retry_from: Optional[str] = None
    max_retries: int = 3
    max_visits: int = Field(default=5)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)


class AgentModule(AsimovBase):
    name: str
    type: ModuleType
    config: ModuleConfig = Field(default_factory=ModuleConfig)
    dependencies: List[str] = Field(default_factory=list)
    input_mailboxes: List[str] = Field(default_factory=list)
    output_mailbox: str = Field(default="")
    container: Optional["CompositeModule"] = None
    executions: int = Field(default=0)
    trace: bool = Field(default=False)
    _generator: Optional[AsyncGenerator] = None

    async def run(self, cache: Cache, semaphore: asyncio.Semaphore) -> dict[str, Any]:
        try:
            if self._generator:
                output = await self._generator.__anext__()
            else:
                task = self.process(cache, semaphore)

                if isinstance(task, AsyncGenerator):
                    self._generator = task

                    output = await task.__anext__()
                else:
                    output = await task
        except StopAsyncIteration as e:
            output = {"status": "success", "result": "Generator Exhausted"}
        except StopIteration as e:
            output = {"status": "success", "result": "Generator Exhausted"}

        if self.output_mailbox:
            await cache.publish_to_mailbox(self.output_mailbox, output)

        return output

    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> dict[str, Any]:
        raise NotImplementedError

    def is_success(self, result: Dict[str, Any]) -> bool:
        return result.get("status") == "success"


class FlowDecision(AsimovBase):
    next_node: str
    condition: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    condition_variables: List[str] = Field(default_factory=list)
    cleanup_on_jump: bool = Field(default=False)


class FlowControlConfig(AsimovBase):
    decisions: List[FlowDecision]
    default: Optional[str] = None
    cleanup_on_default: bool = True


class FlowControlModule(AgentModule):
    flow_config: FlowControlConfig

    async def run(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        # Get keys once
        cache_keys = await cache.keys()

        for decision in self.flow_config.decisions:
            if decision.condition:
                if await self.evaluate_condition(decision.condition, cache, cache_keys):
                    # Unset variables used in the condition
                    for var in decision.condition_variables:
                        await cache.delete(var)
                        lua_globals = lua.globals()
                        lua_globals[var] = None

                    return {
                        "status": "success",
                        "cleanup": decision.cleanup_on_jump,
                        "decision": decision.next_node,
                        "metadata": decision.metadata,
                    }
            else:
                # Unconditional jump
                return {
                    "status": "success",
                    "cleanup": decision.cleanup_on_jump,
                    "decision": decision.next_node,
                    "metadata": decision.metadata,
                }

        if self.flow_config.default:
            return {
                "status": "success",
                "decision": self.flow_config.default,
                "cleanup": self.flow_config.cleanup_on_default,
                "metadata": {},
            }
        else:
            # If no decisions were met, fall through
            return {
                "status": "success",
                "decision": None,  # Indicates fall-through
                "cleanup": True,
                "metadata": {},
            }

    async def _apply_cache_affixes_condition(
        self, condition: str, cache: Cache, cache_keys: set[str]
    ) -> str:
        parts = condition.split(" ")

        try:
            keys = {k.split(cache.affix_sep)[1] for k in cache_keys}
        except IndexError:
            print("No affixes, returning raw keys.")
            keys = cache_keys

        new_parts = []

        for part in parts:
            new_part = part

            if new_part in keys:
                new_part = await cache.apply_key_modifications(part)
                new_part = "v_" + new_part.replace(cache.affix_sep, "_")

            new_parts.append(new_part)

        return " ".join(new_parts)

    async def evaluate_condition(
        self, condition: str, cache: Cache, cache_keys: set[str]
    ) -> bool:
        # TODO: these are never cleaned up
        lua_globals = lua.globals()

        lua_vars = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", condition)
        for orig_var in lua_vars:
            renamed_var = await cache.apply_key_modifications(orig_var)
            if renamed_var in cache_keys:
                lua_safe_key = "v_" + renamed_var.replace(cache.affix_sep, "_")
                lua_globals[lua_safe_key] = await cache.get(orig_var)

        modified_condition = await self._apply_cache_affixes_condition(
            condition, cache, cache_keys
        )
        return lua.eval(modified_condition)


class CompositeModule(AgentModule):
    modules: List[AgentModule]
    node_config: NodeConfig

    async def run(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        if self.node_config.parallel:
            result = await self.run_parallel_modules(cache, semaphore)
        else:
            result = await self.run_sequential_modules(cache, semaphore)

        if self.output_mailbox:
            await cache.publish_to_mailbox(self.output_mailbox, result)

        return result

    async def apply_middlewares(
        self, middlewares: List[Middleware], data: Dict[str, Any], cache: Cache
    ) -> Dict[str, Any]:
        for middleware in middlewares:
            data = await middleware.process(data, cache)
        return data

    async def run_sequential_modules(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        results = []
        for module in self.modules:
            module.container = self
            result = await self.run_module_with_concurrency_control(
                module, cache, semaphore
            )
            if not self.is_success(result):
                return result

            results.append(result)

            if self.config.stop_on_success and self.is_success(result):
                break
        return {"status": "success", "results": results}

    async def run_parallel_modules(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        tasks = []
        for module in self.modules:
            module.container = self
            task = asyncio.create_task(
                self.run_module_with_concurrency_control(module, cache, semaphore)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        for result in results:
            if not self.is_success(result):
                result["all_results"] = results
                return result

        return {"status": "success", "results": results}

    async def run_module_with_concurrency_control(
        self,
        module: AgentModule,
        cache: Cache,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        async with semaphore:
            try:
                async with asyncio.timeout(module.config.timeout):
                    if module.trace:
                        with tracer.start_as_current_span("run_module") as span:
                            span.set_attribute("module_name", module.name)
                            result = await module.run(cache, semaphore)
                    else:
                        result = await module.run(cache, semaphore)
                    return await self.apply_middlewares(
                        module.config.middlewares, result, cache
                    )
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "error": f"Module {module.name} execution timed out after {module.config.timeout} seconds",
                    "module": module.name,
                }


class SnapshotControl(Enum):
    NEVER = "never"
    ONCE = "once"
    ALWAYS = "always"


class Node(CompositeModule):
    _nodes: Dict[str, "Node"] = PrivateAttr()
    modules: List[AgentModule] = Field(default_factory=list)
    node_config: NodeConfig = Field(default_factory=NodeConfig)
    type: ModuleType = Field(default=ModuleType.SUBGRAPH)
    dependencies: List[str] = Field(default_factory=list)
    snapshot: SnapshotControl = Field(default=SnapshotControl.NEVER)

    def set_nodes(self, nodes: Dict[str, "Node"]):
        self._nodes = nodes

    def get_nodes(self):
        return self._nodes

    async def cancel_gen(self, agen):
        await agen.aclose()

    async def cleanup(self):
        for module in self.modules:
            if module._generator:
                await self.cancel_gen(module._generator)

                module._generator = None


class ExecutionStep(TypedDict):
    executed: List[bool]
    nodes: List[str]
    skipped: List[bool]


ExecutionPlan = List[ExecutionStep]


class ExecutionState(AsimovBase):
    execution_index: int = Field(default=0)
    current_plan: ExecutionPlan = Field(default_factory=list)
    execution_history: List[ExecutionPlan] = Field(default_factory=list)
    total_iterations: int = Field(default=0)

    def mark_executed(self, step_index: int, node_index: int):
        if 0 <= step_index < len(self.current_plan):
            step = self.current_plan[step_index]
            if 0 <= node_index < len(step["nodes"]):
                step["executed"][node_index] = True

    def was_executed(self, node_name: str) -> bool:
        for step in self.current_plan:
            if node_name in step["nodes"]:
                index = step["nodes"].index(node_name)
                return step["executed"][index]
        return False

    def was_skipped(self, node_name: str) -> bool:
        for step in self.current_plan:
            if node_name in step["nodes"]:
                index = step["nodes"].index(node_name)
                return step["skipped"][index]
        return False


def create_redis_cache():
    return RedisCache()


class Agent(AsimovBase):
    cache: Cache = Field(default_factory=create_redis_cache)
    nodes: Dict[str, Node] = Field(default_factory=dict)
    graph: Dict[str, List[str]] = Field(default_factory=dict)
    max_total_iterations: int = Field(default=100)
    max_concurrent_tasks: int = Field(default=5)
    _semaphore: asyncio.Semaphore = PrivateAttr()
    node_results: Dict[str, Any] = Field(default_factory=dict)
    output_mailbox: str = "agent_output"
    error_mailbox: str = "agent_error"
    execution_state: ExecutionState = Field(default_factory=ExecutionState)
    auto_snapshot: bool = False
    _logger: logging.Logger
    _snapshot_generation: int = 0

    @model_validator(mode="after")
    def set_semaphore(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self._logger = logging.getLogger(__name__)

        return self

    def __getstate__(self) -> Dict[Any, Any]:
        # N.B. This does _not_ use pydantic's __getstate__ so unpickling results in a half-initialized object
        # This is fine for snapshot purposes though since we only need to copy some state out.
        state = self.__dict__.copy()
        del state["cache"]
        # Don't serialize nodes since those can contain unserializable things like callbacks
        del state["nodes"]
        return state

    def __setstate__(self, state: Dict[Any, Any]) -> None:
        self.__dict__.update(state)

    def is_success(self, result):
        return result["status"] == "success"

    def add_node(self, node: Node):
        if node.name in self.nodes:
            raise ValueError(f"Node with name {node.name} already exists")
        self.nodes[node.name] = node
        self.graph[node.name] = list(node.dependencies)

        node.set_nodes(self.nodes)

    def add_multiple_nodes(self, nodes: List[Node]):
        for node in nodes:
            self.add_node(node)

    async def run_task(self, task: Task) -> None:
        task.status = TaskStatus.EXECUTING

        self.execution_state.current_plan = self.compile_execution_plan()
        self.execution_state.execution_history.append(self.execution_state.current_plan)

        with tracer.start_as_current_span("run_task") as span:
            span.set_attribute("task_id", str(task.id))
            await self._run_task(task)

    async def _run_task(self, task: Task) -> None:
        await self.cache.set("task", task)

        failed_chains: set[tuple[str, ...]] = set()
        node_visit_count: defaultdict[str, int] = defaultdict(int)

        while self.execution_state.current_plan:
            while self.execution_state.execution_index < len(
                self.execution_state.current_plan
            ):
                self.execution_state.total_iterations += 1
                if self.execution_state.total_iterations > self.max_total_iterations:
                    self._logger.warning(
                        f"Graph execution exceeded maximum total iterations ({self.max_total_iterations})"
                    )
                    await self.cache.publish_to_mailbox(
                        self.error_mailbox,
                        {
                            "status": "error",
                            "error": f"Graph execution exceeded maximum total iterations ({self.max_total_iterations})",
                        },
                    )

                    task.status = TaskStatus.FAILED
                    await self.cache.publish_to_mailbox(
                        self.output_mailbox,
                        {
                            "status": TaskStatus.FAILED,
                            "failed_chains": list(failed_chains),
                            "result": self.node_results,
                        },
                    )

                    return

                parallel_group = self.execution_state.current_plan[
                    self.execution_state.execution_index
                ]

                if self.auto_snapshot and any(
                    self.nodes[n].snapshot == SnapshotControl.ALWAYS
                    or (
                        self.nodes[n].snapshot == SnapshotControl.ONCE
                        and node_visit_count[n] == 0
                    )
                    for n in parallel_group["nodes"]
                ):
                    dir = await self.snapshot(
                        task,
                        "-".join(
                            sorted(
                                n
                                for n in parallel_group["nodes"]
                                if self.nodes[n].snapshot != SnapshotControl.NEVER
                            )
                        ),
                    )
                    self._logger.info(f"Snapshot: {dir}")

                tasks = []

                # print(parallel_group["nodes"])
                from pprint import pprint

                # pprint(self.execution_state.current_plan)

                for i, node_name in enumerate(parallel_group["nodes"]):
                    if (
                        not self.execution_state.was_executed(node_name)
                        and not self.execution_state.was_skipped(node_name)
                        and not any(
                            dep in chain
                            for chain in failed_chains
                            for dep in self.nodes[node_name].dependencies
                        )
                    ):
                        # print(node_name)
                        node_visit_count[node_name] += 1
                        if (
                            self.nodes[node_name].node_config.max_visits > 0
                            and node_visit_count[node_name]
                            > self.nodes[node_name].node_config.max_visits
                        ):
                            self._logger.warning(
                                f"Node {node_name} exceeded maximum visits ({self.nodes[node_name].node_config.max_visits})"
                            )
                            await self.cache.publish_to_mailbox(
                                self.error_mailbox,
                                {
                                    "status": "error",
                                    "error": f"Node {node_name} exceeded maximum visits ({self.nodes[node_name].node_config.max_visits})",
                                },
                            )
                            failed_chains.add(self.get_dependent_chains(node_name))
                        else:
                            tasks.append((i, self.run_node(node_name)))

                results = await asyncio.gather(
                    *[task[1] for task in tasks], return_exceptions=True
                )

                flow_control_executed = False
                new_nodes_added = False
                for (i, _), result in zip(tasks, results):
                    node_name = parallel_group["nodes"][i]

                    if not self.is_success(result):
                        dependent_chain = self.get_dependent_chains(node_name)
                        failed_chains.add(dependent_chain)
                        self._logger.warning(f"Node '{node_name}' error '{result}'")
                        await self.cache.publish_to_mailbox(
                            self.error_mailbox,
                            {
                                "status": "error",
                                "node": node_name,
                                "error": str(result),
                            },
                        )
                        continue

                    self.node_results[node_name] = result

                    self.execution_state.mark_executed(
                        self.execution_state.execution_index, i
                    )

                    if isinstance(result, dict) and "results" in result:
                        for module_result in result["results"]:
                            if isinstance(module_result, dict):
                                if "new_nodes" in module_result:
                                    for new_node in module_result["new_nodes"]:
                                        self.add_node(new_node)
                                        self.update_dependencies(
                                            node_name, new_node.name
                                        )
                                        new_nodes_added = True
                                elif "decision" in module_result:
                                    next_node = module_result["decision"]
                                    if module_result["cleanup"]:
                                        # TODO This probably isn't entirely correct so look into it.
                                        for step in self.execution_state.current_plan:
                                            if next_node in step["nodes"]:
                                                await self.nodes[next_node].cleanup()
                                                break

                                            for node in step["nodes"]:
                                                await self.nodes[node].cleanup()

                                    if (
                                        next_node is not None
                                    ):  # A specific next node was chosen
                                        if next_node in self.nodes:
                                            if next_node not in failed_chains:
                                                new_plan = (
                                                    self.compile_execution_plan_from(
                                                        next_node
                                                    )
                                                )

                                                self.execution_state.current_plan = (
                                                    new_plan
                                                )
                                                self.execution_state.execution_index = 0
                                                self.execution_state.execution_history.append(
                                                    new_plan
                                                )
                                                flow_control_executed = True
                                            else:
                                                self._logger.warning(
                                                    f"Flow control attempted to jump to failed node: {next_node}"
                                                )
                                                await self.cache.publish_to_mailbox(
                                                    self.error_mailbox,
                                                    {
                                                        "status": "error",
                                                        "error": f"Flow control attempted to jump to failed node: {next_node}",
                                                    },
                                                )
                                                # Mark the chain containing this flow control node as failed
                                                failed_chains.add(
                                                    self.get_dependent_chains(node_name)
                                                )
                                                break
                                        else:
                                            self._logger.warning(
                                                f"Invalid next node: {next_node}"
                                            )
                                            await self.cache.publish_to_mailbox(
                                                self.error_mailbox,
                                                {
                                                    "status": "error",
                                                    "error": f"Invalid next node: {next_node}",
                                                },
                                            )
                                    else:  # Fall-through case
                                        flow_control_executed = False  # Continue with the next node in the current plan

                if new_nodes_added:
                    new_plan = self.compile_execution_plan_from(
                        module_result["new_nodes"][0].name
                    )
                    self.execution_state.current_plan = new_plan
                    self.execution_state.execution_index = 0
                    self.execution_state.execution_history.append(new_plan)
                elif not flow_control_executed:
                    self.execution_state.execution_index += 1

            if self.execution_state.execution_index >= len(
                self.execution_state.current_plan
            ):
                # Current plan completed, prepare for the next plan if any
                self.execution_state.current_plan = []
                self.execution_state.execution_index = 0

        if failed_chains:
            failed_count = 0
            for chain in list(failed_chains):
                failed_count += len(chain)

            if failed_count >= len(self.nodes):
                task.status = TaskStatus.FAILED
                await self.cache.publish_to_mailbox(
                    self.output_mailbox,
                    {
                        "status": TaskStatus.FAILED,
                        "failed_chains": list(failed_chains),
                        "result": self.node_results,
                    },
                )
            else:
                task.status = TaskStatus.PARTIAL
                await self.cache.publish_to_mailbox(
                    self.output_mailbox,
                    {
                        "status": TaskStatus.PARTIAL,
                        "failed_chains": list(failed_chains),
                        "result": self.node_results,
                    },
                )
        else:
            task.status = TaskStatus.COMPLETE
            await self.cache.publish_to_mailbox(
                self.output_mailbox,
                {"status": TaskStatus.COMPLETE, "result": self.node_results},
            )

        await self.cache.set("task", task)

    async def run_node(self, node_name: str) -> Dict[str, Any]:
        node = self.nodes[node_name]
        node.executions += 1
        retries = 0

        result = None
        last_exception = None

        while (
            retries < node.node_config.max_retries and node.node_config.retry_on_failure
        ):
            try:
                if node.trace:
                    with tracer.start_as_current_span("run_node") as span:
                        span.set_attribute("node_name", node_name)
                        span.set_attribute("retry", retries)
                        result = await node.run(
                            cache=self.cache, semaphore=self._semaphore
                        )
                        span.set_attribute("success", self.is_success(result))
                else:
                    result = await node.run(cache=self.cache, semaphore=self._semaphore)

                if not self.is_success(result):
                    retries += 1
                    continue

                return result
            except Exception as e:
                last_exception = e

                print(
                    f"Error in node {node_name} (attempt {retries + 1}/{node.node_config.max_retries}): {str(e)}"
                )
                print(traceback.format_exc())
                retries += 1
        print(f"Node {node_name} failed after {node.node_config.max_retries} attempts")

        if not result:
            result = {"status": "error", "result": str(last_exception)}

        return result

    def get_dependent_chains(self, node_name: str) -> tuple[str, ...]:
        dependent_chains: set[str] = set()
        dependent_chains.add(node_name)
        queue = [node_name]
        while queue:
            current = queue.pop(0)
            for name, node in self.nodes.items():
                if current in node.dependencies:
                    dependent_chains.add(name)
                    queue.append(name)
        return tuple(list(dependent_chains))

    def update_dependencies(self, parent_node: str, new_node: str):
        # Find all nodes that depend on the parent node
        dependent_nodes = [
            node
            for node, deps in self.graph.items()
            if parent_node in deps and node != new_node
        ]

        # Update their dependencies to include the new node instead of the parent
        for node in dependent_nodes:
            # Update graph
            self.graph[node] = [
                new_node if dep == parent_node else dep for dep in self.graph[node]
            ]

            # Update node dependencies
            self.nodes[node].dependencies = [
                new_node if dep == parent_node else dep
                for dep in self.nodes[node].dependencies
            ]

        # Set the new node's dependency to the parent node
        self.graph[new_node] = [parent_node]
        self.nodes[new_node].dependencies = [parent_node]

        # Ensure the parent node is not dependent on the new node
        if new_node in self.graph[parent_node]:
            self.graph[parent_node].remove(new_node)
        if new_node in self.nodes[parent_node].dependencies:
            self.nodes[parent_node].dependencies.remove(new_node)

    def slice_before_node(self, execution_plan: List[dict], node: str) -> List[dict]:
        for index, step in enumerate(execution_plan):
            if node in step["nodes"]:
                return execution_plan[index:]
        return execution_plan  # or return an empty list if the node isn't found

    def mark_node_and_deps_as_skipped(self, node_name: str):
        for step in self.execution_state.current_plan:
            if node_name in step["nodes"]:
                index = step["nodes"].index(node_name)
                step["skipped"][index] = True

                # Mark all dependencies as skipped
                deps_to_skip = set(self.get_dependent_chains(node_name))
                for dep_step in self.execution_state.current_plan:
                    for i, dep_node in enumerate(dep_step["nodes"]):
                        if dep_node in deps_to_skip:
                            dep_step["skipped"][i] = True

                break

    def _topological_sort(self, graph):
        visited = set()
        stack = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph.get(node, []):
                dfs(neighbor)
            stack.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        # Reverse the stack to get the correct topological order
        stack.reverse()

        # Group nodes that can be executed in parallel
        result = []
        while stack:
            parallel_group = []
            next_stack = []
            for node in stack:
                if all(dep not in stack for dep in graph.get(node, [])):
                    parallel_group.append(node)
                else:
                    next_stack.append(node)
            result.append(parallel_group)
            stack = next_stack

        return result

    def compile_execution_plan(self) -> ExecutionPlan:
        plan = self._topological_sort(self.graph)
        return [
            ExecutionStep(
                executed=[False] * len(group), nodes=group, skipped=[False] * len(group)
            )
            for group in plan
        ]

    def compile_execution_plan_from(self, start_node: str) -> ExecutionPlan:
        full_plan = self.compile_execution_plan()
        start_index = next(
            (i for i, step in enumerate(full_plan) if start_node in step["nodes"]), None
        )

        if start_index is None:
            return full_plan  # Return the full plan if start_node is not found

        new_plan = full_plan[start_index:]

        # Mark nodes as skipped in the new plan
        for step_index, step in enumerate(new_plan):
            if start_node in step["nodes"]:
                # We've found the start node
                # Mark all previous nodes as skipped
                for prev_step in new_plan[:step_index]:
                    prev_step["skipped"] = [True] * len(prev_step["nodes"])

                # Mark siblings in this parallel group as skipped
                for i, node in enumerate(step["nodes"]):
                    if node != start_node:
                        step["skipped"][i] = True
                        # Mark all dependents of this sibling as skipped
                        dependents = self.get_dependent_chains(node)
                        for future_step in new_plan[step_index + 1 :]:
                            for j, future_node in enumerate(future_step["nodes"]):
                                if future_node in dependents:
                                    future_step["skipped"][j] = True

                # We unmark as skipped any shared dependents of the node we're jumping to.
                # Users have to be aware of this fact in docs as it can footgun them.
                dependents = self.get_dependent_chains(start_node)
                for future_step in new_plan[step_index + 1 :]:
                    for j, future_node in enumerate(future_step["nodes"]):
                        if future_node in dependents:
                            future_step["skipped"][j] = False
                # No need to process further steps
                return new_plan

        return new_plan

    def _is_reachable(self, start: str, end: str) -> bool:
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node not in visited:
                if node == end:
                    return True
                visited.add(node)
                stack.extend(self.graph[node])
        return False

    async def snapshot(self, task: Task, name_hint: str) -> pathlib.Path:
        out_dir = pathlib.Path(
            os.environ.get("ASIMOV_SNAPSHOT", "/tmp/asimov_snapshot")
        )
        out_dir = out_dir / str(task.id)
        out_dir = out_dir / f"{self._snapshot_generation}_{name_hint}"
        self._snapshot_generation += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "agent.pkl", "wb") as f:
            pickle.dump(self, f)
        with open(out_dir / "cache.json", "w") as f:
            f.write(jsonpickle.encode(await self.cache.get_all()))
        with open(out_dir / "task.pkl", "wb") as f:
            pickle.dump(task, f)
        return out_dir

    async def run_from_snapshot(self, snapshot_dir: pathlib.Path):
        with open(snapshot_dir / "agent.pkl", "rb") as f:
            agent = pickle.load(f)
            self.graph = agent.graph
            self.max_total_iterations = agent.max_total_iterations
            self.max_concurrent_tasks = agent.max_concurrent_tasks
            self.node_results = agent.node_results
            self.output_mailbox = agent.output_mailbox
            self.error_mailbox = agent.error_mailbox
            self.execution_state = agent.execution_state
        with open(snapshot_dir / "cache.json", "r") as f:
            await self.cache.clear()
            cache = jsonpickle.decode(f.read())
            for k, v in cache.items():
                # get_all returns keys with prefixes, so set with raw=True
                await self.cache.set(k, v, raw=True)
        with open(snapshot_dir / "task.pkl", "rb") as f:
            task = pickle.load(f)
        await self._run_task(task)
