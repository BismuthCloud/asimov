import pytest
import asyncio
from typing import Dict, Any, AsyncGenerator
from pydantic import Field
from asimov.graph.tasks import Task, TaskStatus
from pprint import pprint

from asimov.caches.mock_redis_cache import MockRedisCache
from asimov.graph.graph import (
    Agent,
    Node,
    AgentModule,
    ModuleConfig,
    NodeConfig,
    Middleware,
    FlowControlModule,
    FlowControlConfig,
    FlowDecision,
    ModuleType,
    Cache,
)


@pytest.fixture
def mock_cache():
    return MockRedisCache()


@pytest.fixture
def simple_agent(mock_cache):
    agent = Agent(cache=mock_cache)
    return agent


class MockModule(AgentModule):
    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        return {"status": "success", "result": "test"}


class SlowMockModule(AgentModule):
    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"status": "success", "result": f"test_{self.name}"}


class TimeoutModule(AgentModule):
    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        await asyncio.sleep(5)
        return {"status": "success", "result": "test"}


class DependentModule(AgentModule):
    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"status": "success", "result": f"Executed {self.name}"}


class FailingModule(AgentModule):
    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        raise Exception(f"Failure in {self.name}")


@pytest.mark.asyncio
async def test_dependency_resolution(simple_agent, mock_cache):
    simple_agent.add_node(
        Node(
            name="NodeA",
            modules=[
                DependentModule(
                    name="A",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeB",
            modules=[
                DependentModule(
                    name="B",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=["NodeA"],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeC",
            modules=[
                DependentModule(
                    name="C",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=["NodeA"],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeD",
            modules=[
                DependentModule(
                    name="D",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=["NodeB", "NodeC"],
        )
    )

    task = Task(type="test", objective="Test dependency resolution", params={})
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    assert result["status"] == TaskStatus.COMPLETE
    assert len(result["result"]) == 4  # 4 node results
    assert "Executed A" in str(result["result"]["NodeA"])
    assert "Executed B" in str(result["result"]["NodeB"])
    assert "Executed C" in str(result["result"]["NodeC"])
    assert "Executed D" in str(result["result"]["NodeD"])


@pytest.mark.asyncio
async def test_parallel_execution(simple_agent, mock_cache):
    simple_agent.add_node(
        Node(
            name="NodeA",
            modules=[
                SlowMockModule(
                    name="A",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeB",
            modules=[
                SlowMockModule(
                    name="B",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeC",
            modules=[
                SlowMockModule(
                    name="C",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=[],
        )
    )

    start_time = asyncio.get_event_loop().time()
    task = Task(type="test", objective="Test parallel execution", params={})
    await simple_agent.run_task(task)
    end_time = asyncio.get_event_loop().time()

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    assert result["status"] == TaskStatus.COMPLETE
    assert len(result["result"]) == 3  # 3 node results
    assert (
        end_time - start_time < 0.3
    )  # Should take less than 0.3 seconds if executed in parallel


@pytest.mark.asyncio
async def test_error_handling(simple_agent, mock_cache):
    simple_agent.add_node(
        Node(
            name="NodeA",
            modules=[
                DependentModule(
                    name="A",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeB",
            modules=[
                FailingModule(
                    name="B",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeC",
            modules=[
                DependentModule(
                    name="C",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=["NodeA"],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeD",
            modules=[
                DependentModule(
                    name="D",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=["NodeB"],
        )
    )

    task = Task(type="test", objective="Test error handling", params={})
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    error = await mock_cache.get_message(simple_agent.error_mailbox)

    assert result["status"] == TaskStatus.PARTIAL
    assert "NodeB" in result["failed_chains"][0]
    assert "Executed A" in str(result["result"]["NodeA"])
    assert "Executed C" in str(result["result"]["NodeC"])
    assert "NodeB" not in result["result"]
    assert "NodeD" not in result["result"]
    assert error["status"] == "error" and "NodeB" in error["node"]


@pytest.mark.asyncio
async def test_parallel_mailbox_communication(simple_agent, mock_cache):
    class ParallelMailboxModule(AgentModule):
        async def process(
            self, cache: Cache, semaphore: asyncio.Semaphore
        ) -> Dict[str, Any]:
            if not self.input_mailboxes:
                if self.output_mailbox:
                    await cache.publish_to_mailbox(
                        self.output_mailbox, f"Message from {self.name}"
                    )
                return {"status": "success", "result": f"Sent message from {self.name}"}

            messages = []
            for mailbox in self.input_mailboxes:
                message = (await cache.peek_mailbox(mailbox))[0]
                if message:
                    messages.append(message)

            if self.output_mailbox:
                await cache.publish_to_mailbox(
                    self.output_mailbox,
                    f"Processed by {self.name}: {', '.join(messages)}",
                )

            return {
                "status": "success",
                "result": f"Processed in {self.name}: {', '.join(messages)}",
            }

    simple_agent.add_node(
        Node(
            name="NodeA",
            modules=[
                ParallelMailboxModule(
                    name="A",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=[],
                    output_mailbox="mailbox_A",
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeB1",
            modules=[
                ParallelMailboxModule(
                    name="B1",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["mailbox_A"],
                    output_mailbox="mailbox_B1",
                )
            ],
            dependencies=["NodeA"],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeB2",
            modules=[
                ParallelMailboxModule(
                    name="B2",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["mailbox_A"],
                    output_mailbox="mailbox_B2",
                )
            ],
            dependencies=["NodeA"],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeC",
            modules=[
                ParallelMailboxModule(
                    name="C",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["mailbox_B1", "mailbox_B2"],
                    output_mailbox="mailbox_C",
                )
            ],
            dependencies=["NodeB1", "NodeB2"],
        )
    )

    task = Task(type="test", objective="Test parallel mailbox communication", params={})
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    assert result["status"] == TaskStatus.COMPLETE
    assert len(result["result"]) == 4  # A, B1, B2, C results
    assert "Sent message from A" in str(result["result"]["NodeA"])
    assert "Processed in B1: Message from A" in str(result["result"]["NodeB1"])
    assert "Processed in B2: Message from A" in str(result["result"]["NodeB2"])
    assert (
        "Processed in C: Processed by B1: Message from A, Processed by B2: Message from A"
        in str(result["result"]["NodeC"])
    )


@pytest.mark.asyncio
async def test_cross_graph_communication(simple_agent, mock_cache):
    class CrossGraphModule(AgentModule):
        async def process(
            self, cache: Cache, semaphore: asyncio.Semaphore
        ) -> Dict[str, Any]:
            received_messages = []
            for mailbox in self.input_mailboxes:
                message = (await cache.peek_mailbox(mailbox))[0]
                if message:
                    received_messages.append(message)

            if self.output_mailbox:
                output_message = f"Message from {self.name}"
                if received_messages:
                    output_message += f" (Received: {', '.join(received_messages)})"
                await cache.publish_to_mailbox(self.output_mailbox, output_message)

            return {
                "status": "success",
                "result": f"Processed in {self.name}: {', '.join(received_messages)}",
            }

    simple_agent.add_node(
        Node(
            name="NodeA1",
            modules=[
                CrossGraphModule(
                    name="A1",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=[],
                    output_mailbox="mailbox_A1",
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeB1",
            modules=[
                CrossGraphModule(
                    name="B1",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["mailbox_A1", "mailbox_A2"],
                    output_mailbox="mailbox_B1",
                )
            ],
            dependencies=["NodeA1"],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeA2",
            modules=[
                CrossGraphModule(
                    name="A2",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=[],
                    output_mailbox="mailbox_A2",
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeB2",
            modules=[
                CrossGraphModule(
                    name="B2",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["mailbox_A2"],
                    output_mailbox="mailbox_B2",
                )
            ],
            dependencies=["NodeA2"],
        )
    )

    task = Task(type="test", objective="Test cross-graph communication", params={})
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    assert result["status"] == TaskStatus.COMPLETE
    assert len(result["result"]) == 4  # A1, A2, B1, B2 results
    assert "Processed in A1: " in str(result["result"]["NodeA1"])
    assert "Processed in A2: " in str(result["result"]["NodeA2"])
    assert "Processed in B1: Message from A1, Message from A2" in str(
        result["result"]["NodeB1"]
    )
    assert "Processed in B2: Message from A2" in str(result["result"]["NodeB2"])


@pytest.mark.asyncio
async def test_flow_control(simple_agent, mock_cache):
    await mock_cache.set("condition_var", True)

    flow_config = FlowControlConfig(
        decisions=[
            FlowDecision(condition="condition_var == true", next_node="Node_1"),
            FlowDecision(next_node="Node_2"),
        ]
    )

    flow_module = FlowControlModule(
        name="flow",
        type=ModuleType.FLOW_CONTROL,
        config=ModuleConfig(),
        flow_config=flow_config,
    )

    simple_agent.add_node(
        Node(
            name="Node_0",
            modules=[flow_module],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="Node_1",
            modules=[
                MockModule(name="m1", type=ModuleType.EXECUTOR, config=ModuleConfig())
            ],
            dependencies=["Node_0"],
        )
    )
    simple_agent.add_node(
        Node(
            name="Node_2",
            modules=[
                MockModule(name="m2", type=ModuleType.EXECUTOR, config=ModuleConfig())
            ],
            dependencies=["Node_0"],
        )
    )

    task = Task(type="test", objective="Test objective", params={})
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    assert result["status"] == TaskStatus.COMPLETE
    assert "Node_1" in result["result"]
    print(result["result"])
    assert "Node_2" not in result["result"]


@pytest.mark.asyncio
async def test_module_timeout(simple_agent, mock_cache):
    timeout_module = TimeoutModule(
        name="timeout",
        type=ModuleType.EXECUTOR,
        config=ModuleConfig(timeout=1),
    )
    simple_agent.add_node(
        Node(
            name="TimeoutNode",
            modules=[timeout_module],
            dependencies=[],
        )
    )

    task = Task(type="test", objective="Test objective", params={})
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    error = await mock_cache.get_message(simple_agent.error_mailbox)

    print(result)

    assert result["status"] == TaskStatus.FAILED
    assert "execution timed out after 1" in error["error"]


@pytest.mark.asyncio
async def test_complex_graph_construction_and_execution(simple_agent, mock_cache):
    class TestModule(AgentModule):
        async def process(
            self, cache: Cache, semaphore: asyncio.Semaphore
        ) -> Dict[str, Any]:
            await asyncio.sleep(0.1)  # Simulate some work
            if self.input_mailboxes:
                messages = [
                    (await cache.peek_mailbox(mb))[0] for mb in self.input_mailboxes
                ]
                result = (
                    f"Processed in {self.name}: {', '.join(filter(None, messages))}"
                )

            else:
                result = f"Initial processing in {self.name}"

            if self.output_mailbox:
                await cache.publish_to_mailbox(
                    self.output_mailbox, f"Output from {self.name}"
                )

            return {"status": "success", "result": result}

    # Construct a complex graph
    simple_agent.add_node(
        Node(
            name="Start",
            modules=[
                TestModule(
                    name="StartModule",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    output_mailbox="start_output",
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="ProcessA",
            modules=[
                TestModule(
                    name="ProcessAModule",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["start_output"],
                    output_mailbox="process_a_output",
                )
            ],
            dependencies=["Start"],
        )
    )
    simple_agent.add_node(
        Node(
            name="ProcessB",
            modules=[
                TestModule(
                    name="ProcessBModule",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["start_output"],
                    output_mailbox="process_b_output",
                )
            ],
            dependencies=["Start"],
        )
    )
    simple_agent.add_node(
        Node(
            name="Join",
            modules=[
                TestModule(
                    name="JoinModule",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["process_a_output", "process_b_output"],
                    output_mailbox="join_output",
                )
            ],
            dependencies=["ProcessA", "ProcessB"],
        )
    )
    simple_agent.add_node(
        Node(
            name="End",
            modules=[
                TestModule(
                    name="EndModule",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["join_output"],
                )
            ],
            dependencies=["Join"],
        )
    )

    task = Task(
        type="test",
        objective="Test complex graph construction and execution",
        params={},
    )
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)

    assert result["status"] == TaskStatus.COMPLETE
    assert len(result["result"]) == 5  # 5 node results

    assert "Initial processing in StartModule" in str(result["result"]["Start"])
    assert "Processed in ProcessAModule: Output from StartModule" in str(
        result["result"]["ProcessA"]
    )
    assert "Processed in ProcessBModule: Output from StartModule" in str(
        result["result"]["ProcessB"]["results"][0]["result"]
    )
    assert (
        "Processed in JoinModule: Output from ProcessAModule, Output from ProcessBModule"
        in str(result["result"]["Join"])
    )
    assert "Processed in EndModule: Output from JoinModule" in str(
        result["result"]["End"]
    )

    # Verify the graph structure
    assert set(simple_agent.graph.keys()) == {
        "Start",
        "ProcessA",
        "ProcessB",
        "Join",
        "End",
    }

    assert simple_agent.graph["Start"] == []
    assert simple_agent.graph["ProcessA"] == ["Start"]
    assert simple_agent.graph["ProcessB"] == ["Start"]
    assert simple_agent.graph["Join"] == ["ProcessA", "ProcessB"]
    assert simple_agent.graph["End"] == ["Join"]


@pytest.mark.asyncio
async def test_caching_results(simple_agent, mock_cache):
    simple_agent.add_node(
        Node(
            name="NodeA",
            modules=[
                DependentModule(
                    name="A",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeB",
            modules=[
                DependentModule(
                    name="B",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=["NodeA"],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeC",
            modules=[
                FailingModule(
                    name="C",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=[],
        )
    )

    task = Task(type="test", objective="Test caching results", params={})
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    assert "NodeA" in result["result"]
    assert "NodeB" in result["result"]
    assert "NodeC" not in result["result"]


@pytest.mark.asyncio
async def test_agent_initialization(simple_agent):
    assert isinstance(simple_agent._semaphore, asyncio.Semaphore)
    assert simple_agent._semaphore._value == 5  # default max_concurrent_tasks


def test_add_node(simple_agent):
    module = AgentModule(name="module1", type=ModuleType.EXECUTOR)
    simple_agent.add_node(
        Node(
            name="NodeA",
            modules=[module],
            dependencies=[],
        )
    )
    assert len(simple_agent.nodes) == 1
    assert simple_agent.nodes["NodeA"].modules == [module]
    assert simple_agent.nodes["NodeA"].dependencies == []


@pytest.mark.asyncio
async def test_run_task_simple(simple_agent, mock_cache):
    mock_module = MockModule(
        name="test",
        type=ModuleType.EXECUTOR,
        config=ModuleConfig(),
    )
    simple_agent.add_node(
        Node(
            name="NodeA",
            modules=[mock_module],
            dependencies=[],
        )
    )

    task = Task(type="test", objective="Test objective", params={})
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    assert result["status"] == TaskStatus.COMPLETE
    assert result["result"]["NodeA"]["results"][0]["result"] == "test"


@pytest.mark.asyncio
async def test_run_task_max_visits(simple_agent, mock_cache):
    mock_module = MockModule(
        name="test",
        type=ModuleType.EXECUTOR,
        config=ModuleConfig(max_visits=2),
    )
    simple_agent.add_node(
        Node(
            name="NodeA",
            modules=[mock_module],
            dependencies=[],  # Self-dependency to create a loop
        )
    )

    flow_config = FlowControlConfig(
        decisions=[
            FlowDecision(
                condition="true",
                next_node="NodeA",
                condition_variables=[],
            ),
        ]
    )

    simple_agent.add_node(
        Node(
            name="FlowControl",
            modules=[
                FlowControlModule(
                    name="flow",
                    type=ModuleType.FLOW_CONTROL,
                    config=ModuleConfig(),
                    flow_config=flow_config,
                )
            ],
            dependencies=["NodeA"],
        )
    )

    task = Task(type="test", objective="Test objective", params={})
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    error = await mock_cache.get_message(simple_agent.error_mailbox)

    assert result["status"] == TaskStatus.FAILED

    assert "NodeA" in result["failed_chains"][0]
    assert "exceeded maximum visits" in error["error"]


@pytest.mark.asyncio
async def test_concurrency_control(simple_agent, mock_cache):
    simple_agent.max_concurrent_tasks = 2

    for i in range(5):
        simple_agent.add_node(
            Node(
                name=f"Node_{i}",
                modules=[
                    SlowMockModule(
                        name=f"slow_{i}",
                        type=ModuleType.EXECUTOR,
                        config=ModuleConfig(),
                    ),
                ],
                dependencies=[],
            )
        )

    task = Task(type="test", objective="Test objective", params={})
    start_time = asyncio.get_event_loop().time()
    await simple_agent.run_task(task)
    end_time = asyncio.get_event_loop().time()

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    assert result["status"] == TaskStatus.COMPLETE
    assert len(result["result"]) == 5  # 5 module results
    print(end_time - start_time)
    assert (
        0.0 <= (end_time - start_time) < 0.2
    )  # These do nothing so they're bounded by the sleep call in the SlowMockModule


@pytest.mark.asyncio
async def test_mailbox_communication(simple_agent, mock_cache):
    class MailboxModule(AgentModule):
        async def process(
            self, cache: Cache, semaphore: asyncio.Semaphore
        ) -> Dict[str, Any]:
            for mailbox in self.input_mailboxes:
                message = await cache.get_message(mailbox)
                if message:
                    result = f"Processed {message} in {self.name}"
                    if self.output_mailbox:
                        await cache.publish_to_mailbox(
                            self.output_mailbox, f"Message from {self.name}"
                        )
                    return {"status": "success", "result": result}

            if self.output_mailbox:
                await cache.publish_to_mailbox(
                    self.output_mailbox, f"Message from {self.name}"
                )
            return {"status": "success", "result": f"No input for {self.name}"}

    simple_agent.add_node(
        Node(
            name="NodeA",
            modules=[
                MailboxModule(
                    name="A",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=[],
                    output_mailbox="mailbox_A",
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeB",
            modules=[
                MailboxModule(
                    name="B",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["mailbox_A"],
                    output_mailbox="mailbox_B",
                )
            ],
            dependencies=["NodeA"],
        )
    )
    simple_agent.add_node(
        Node(
            name="NodeC",
            modules=[
                MailboxModule(
                    name="C",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                    input_mailboxes=["mailbox_B"],
                    output_mailbox="mailbox_C",
                )
            ],
            dependencies=["NodeB"],
        )
    )

    task = Task(type="test", objective="Test mailbox communication", params={})
    await simple_agent.run_task(task)

    print("Output mailbox:", simple_agent.output_mailbox)
    print("Error mailbox:", simple_agent.error_mailbox)
    print("All data:", await mock_cache.get_all())
    print(
        "Mailbox content:", await mock_cache.peek_mailbox(simple_agent.output_mailbox)
    )

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    print("Result:", result)

    assert result is not None, "No message in output mailbox"
    assert result["status"] == TaskStatus.COMPLETE
    assert "NodeA" in result["result"]
    assert "NodeB" in result["result"]
    assert "NodeC" in result["result"]
    pprint(result)
    assert "No input for A" in result["result"]["NodeA"]["results"][0]["result"]
    assert (
        "Processed Message from A in B"
        in result["result"]["NodeB"]["results"][0]["result"]
    )
    assert (
        "Processed Message from B in C"
        in result["result"]["NodeC"]["results"][0]["result"]
    )


@pytest.mark.asyncio
async def test_complex_graph_execution(simple_agent, mock_cache):
    class FailingModule(AgentModule):
        attempt_count: int = Field(default=0)

        async def process(
            self, cache: Cache, semaphore: asyncio.Semaphore
        ) -> Dict[str, Any]:
            self.attempt_count += 1
            if self.attempt_count == 1:
                raise Exception("Simulated failure")
            return {
                "status": "success",
                "result": f"Succeeded on attempt {self.attempt_count}",
            }

    dynamic_node = Node(
        name="DynamicallyAddedNode",
        modules=[
            MockModule(name="dynamic", type=ModuleType.EXECUTOR, config=ModuleConfig())
        ],
    )

    class DynamicModule(AgentModule):
        async def process(
            self, cache: Cache, semaphore: asyncio.Semaphore
        ) -> Dict[str, Any]:
            new_node = dynamic_node
            new_node.dependencies = ([self.container.name],)

            await cache.set("dynamic_node_added", "true")  # Set as string "true"

            return {
                "status": "success",
                "result": f"Added new node: {new_node.name}",
                "new_nodes": [new_node],
            }

    class MiddlewareTestModule(AgentModule):
        async def process(
            self, cache: Cache, semaphore: asyncio.Semaphore
        ) -> Dict[str, Any]:
            return {"status": "success", "result": "Original result"}

    class TestMiddleware(Middleware):
        async def process(
            self, data: Dict[str, Any], cache: Cache
        ) -> AsyncGenerator[Dict[str, Any], None]:
            data["result"] = f"Modified: {data['result']}"
            yield data

    flow_config = FlowControlConfig(
        decisions=[
            FlowDecision(
                condition="dynamic_node_added == 'true'",
                next_node="DynamicallyAddedNode",
                condition_variables=["dynamic_node_added"],
            ),
            # No need for an explicit fall-through decision
        ]
    )
    simple_agent.add_node(
        Node(
            name="Start",
            modules=[
                MockModule(
                    name="start", type=ModuleType.EXECUTOR, config=ModuleConfig()
                )
            ],
            dependencies=[],
        )
    )
    simple_agent.add_node(
        Node(
            name="ParallelA",
            modules=[
                MockModule(
                    name="parallelA", type=ModuleType.EXECUTOR, config=ModuleConfig()
                )
            ],
            dependencies=["Start"],
        )
    )
    simple_agent.add_node(
        Node(
            name="ParallelB",
            modules=[
                MockModule(
                    name="parallelB", type=ModuleType.EXECUTOR, config=ModuleConfig()
                )
            ],
            dependencies=["Start"],
        )
    )
    simple_agent.add_node(
        Node(
            name="DynamicNodeCreator",
            modules=[
                DynamicModule(
                    name="dynamic_creator",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(),
                )
            ],
            dependencies=["ParallelA", "ParallelB"],
        )
    )
    simple_agent.add_node(
        Node(
            name="MiddlewareTest",
            modules=[
                MiddlewareTestModule(
                    name="middleware_test",
                    type=ModuleType.EXECUTOR,
                    config=ModuleConfig(middlewares=[TestMiddleware()]),
                )
            ],
            dependencies=["DynamicNodeCreator"],
        )
    )
    simple_agent.add_node(
        Node(
            name="FailingNode",
            modules=[
                FailingModule(
                    name="failing", type=ModuleType.EXECUTOR, config=ModuleConfig()
                )
            ],
            dependencies=["MiddlewareTest"],
            node_config=NodeConfig(retry_on_failure=True, max_retries=2),
        )
    )
    simple_agent.add_node(
        Node(
            name="FlowControl",
            modules=[
                FlowControlModule(
                    name="flow",
                    type=ModuleType.FLOW_CONTROL,
                    config=ModuleConfig(),
                    flow_config=flow_config,
                )
            ],
            dependencies=["FailingNode"],
        )
    )
    simple_agent.add_node(
        Node(
            name="FinalNode",
            modules=[
                MockModule(
                    name="final", type=ModuleType.EXECUTOR, config=ModuleConfig()
                )
            ],
            dependencies=["FlowControl"],
        )
    )

    task = Task(type="test", objective="Test complex graph execution", params={})
    await simple_agent.run_task(task)

    result = await mock_cache.get_message(simple_agent.output_mailbox)
    assert result is not None
    assert result["status"] == TaskStatus.COMPLETE

    # Verify dynamic node creation
    assert "DynamicallyAddedNode" in result["result"]

    # Verify middleware execution
    assert (
        "Modified: Original result"
        in result["result"]["MiddlewareTest"]["results"][0]["result"]
    )

    # Verify retry mechanism
    assert (
        "Succeeded on attempt 3"
        in result["result"]["FailingNode"]["results"][0]["result"]
    )

    # Verify both DynamicallyAddedNode and FinalNode were executed
    assert (
        "DynamicallyAddedNode" in result["result"]
    ), "DynamicallyAddedNode should have been executed"
    assert "FinalNode" in result["result"], "FinalNode should have been executed"

    # Check for any error messages
    error_messages = await mock_cache.get_message(simple_agent.error_mailbox)
    assert error_messages is None, f"Unexpected errors: {error_messages}"

    # Verify final node execution
    assert result["result"]["FinalNode"]["results"][0]["status"] == "success"

    # Check for any error messages
    error_messages = await mock_cache.get_message(simple_agent.error_mailbox)
    assert error_messages is None, f"Unexpected errors: {error_messages}"

    # Verify dynamic node creation
    assert "DynamicallyAddedNode" in result["result"]

    # Verify DynamicallyAddedNode was executed twice
    assert dynamic_node.executions == 2

    # Verify FinalNode was executed
    assert "FinalNode" in result["result"], "FinalNode should have been executed"

    # Check for any error messages
    error_messages = await mock_cache.get_message(simple_agent.error_mailbox)
    assert error_messages is None, f"Unexpected errors: {error_messages}"


if __name__ == "__main__":
    pytest.main()
