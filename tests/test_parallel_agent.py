import pytest
import asyncio
from typing import Dict, Any, List

from asimov.graph import (
    Agent,
    ModuleType,
    Node,
    NodeConfig,
    FlowControlModule,
    FlowControlConfig,
    FlowDecision,
)
from asimov.graph.tasks import Task
from asimov.caches.mock_redis_cache import MockRedisCache
from examples.parallel_agent import (
    TaskSplitterModule,
    ParallelProcessorModule,
    ResultAggregatorModule
)

@pytest.fixture
def mock_cache():
    return MockRedisCache()

@pytest.fixture
def parallel_agent(mock_cache):
    # Create the agent
    agent = Agent(
        cache=mock_cache,
        max_concurrent_tasks=3,
        max_total_iterations=50
    )
    
    # Create nodes
    splitter_node = Node(
        name="splitter",
        type=ModuleType.PLANNER,
        modules=[TaskSplitterModule()],
        node_config=NodeConfig(
            parallel=False,
            max_retries=2
        )
    )
    
    # Create parallel processor nodes
    processor_nodes = []
    for i in range(3):
        processor_node = Node(
            name=f"processor_{i}",
            type=ModuleType.EXECUTOR,
            modules=[ParallelProcessorModule()],
            dependencies=["splitter"],
            node_config=NodeConfig(
                parallel=True,
                max_retries=2
            )
        )
        processor_nodes.append(processor_node)
    
    aggregator_node = Node(
        name="aggregator",
        type=ModuleType.EXECUTOR,
        modules=[ResultAggregatorModule()],
        dependencies=[node.name for node in processor_nodes],
        node_config=NodeConfig(
            parallel=False,
            condition="len(completed_subtasks) == len(subtasks)"
        )
    )
    
    flow_control = Node(
        name="flow_control",
        type=ModuleType.FLOW_CONTROL,
        modules=[FlowControlModule(
            flow_config=FlowControlConfig(
                decisions=[
                    FlowDecision(
                        next_node="processor_0",
                        condition="subtasks != null and len([st for st in subtasks if st['status'] == 'pending']) > 0"
                    ),
                    FlowDecision(
                        next_node="processor_1",
                        condition="subtasks != null and len([st for st in subtasks if st['status'] == 'pending']) > 0"
                    ),
                    FlowDecision(
                        next_node="processor_2",
                        condition="subtasks != null and len([st for st in subtasks if st['status'] == 'pending']) > 0"
                    ),
                    FlowDecision(
                        next_node="aggregator",
                        condition="len(completed_subtasks) == len(subtasks)"
                    )
                ],
                default="splitter"
            )
        )]
    )
    
    # Add all nodes to agent
    agent.add_node(splitter_node)
    for node in processor_nodes:
        agent.add_node(node)
    agent.add_node(aggregator_node)
    agent.add_node(flow_control)
    
    return agent

@pytest.mark.asyncio
async def test_parallel_agent_initialization(parallel_agent):
    """Test that the parallel agent is initialized with correct components."""
    assert len(parallel_agent.nodes) == 6  # splitter + 3 processors + aggregator + flow_control
    assert "splitter" in parallel_agent.nodes
    assert "processor_0" in parallel_agent.nodes
    assert "processor_1" in parallel_agent.nodes
    assert "processor_2" in parallel_agent.nodes
    assert "aggregator" in parallel_agent.nodes
    assert "flow_control" in parallel_agent.nodes
    
    # Verify node types
    assert parallel_agent.nodes["splitter"].type == ModuleType.PLANNER
    assert parallel_agent.nodes["processor_0"].type == ModuleType.EXECUTOR
    assert parallel_agent.nodes["aggregator"].type == ModuleType.EXECUTOR
    assert parallel_agent.nodes["flow_control"].type == ModuleType.FLOW_CONTROL

@pytest.mark.asyncio
async def test_task_splitting(parallel_agent, mock_cache):
    """Test the task splitting functionality."""
    task = Task(
        type="parallel_processing",
        objective="Test task splitting",
        params={"data": list(range(10))}  # 10 items to process
    )
    
    # Run splitter node
    splitter_node = parallel_agent.nodes["splitter"]
    result = await splitter_node.process(task.id, mock_cache)
    
    # Verify splitting result
    assert result["status"] == "success"
    
    # Verify subtasks were created and stored
    subtasks = await mock_cache.get("subtasks")
    assert isinstance(subtasks, list)
    assert len(subtasks) > 0
    
    # Verify subtask structure
    for subtask in subtasks:
        assert "id" in subtask
        assert "data" in subtask
        assert "status" in subtask
        assert subtask["status"] == "pending"
        assert isinstance(subtask["data"], list)

@pytest.mark.asyncio
async def test_parallel_processing(parallel_agent, mock_cache):
    """Test the parallel processing functionality."""
    # Setup test data
    subtasks = [
        {"id": "subtask_0", "data": [1, 2, 3], "status": "pending"},
        {"id": "subtask_1", "data": [4, 5, 6], "status": "pending"}
    ]
    await mock_cache.set("subtasks", subtasks)
    await mock_cache.set("completed_subtasks", [])
    
    # Run processor nodes
    results = []
    for i in range(2):  # Process two subtasks
        processor = parallel_agent.nodes[f"processor_{i}"]
        result = await processor.process(f"task_{i}", mock_cache)
        results.append(result)
    
    # Verify processing results
    for result in results:
        assert result["status"] == "success"
        assert "subtask_id" in result["result"]
        assert "processed_items" in result["result"]
    
    # Verify subtasks were updated in cache
    updated_subtasks = await mock_cache.get("subtasks")
    completed = [st for st in updated_subtasks if st["status"] == "completed"]
    assert len(completed) == 2

@pytest.mark.asyncio
async def test_result_aggregation(parallel_agent, mock_cache):
    """Test the result aggregation functionality."""
    # Setup completed subtasks
    completed_subtasks = [
        {
            "id": "subtask_0",
            "data": [1, 2],
            "status": "completed",
            "results": [2, 4]  # Each number multiplied by 2
        },
        {
            "id": "subtask_1",
            "data": [3, 4],
            "status": "completed",
            "results": [6, 8]
        }
    ]
    await mock_cache.set("completed_subtasks", completed_subtasks)
    
    # Run aggregator node
    aggregator_node = parallel_agent.nodes["aggregator"]
    result = await aggregator_node.process("test_task", mock_cache)
    
    # Verify aggregation results
    assert result["status"] == "success"
    assert "total_items_processed" in result["result"]
    assert "final_results" in result["result"]
    assert result["result"]["total_items_processed"] == 4
    assert result["result"]["final_results"] == [2, 4, 6, 8]

@pytest.mark.asyncio
async def test_end_to_end_processing(parallel_agent, mock_cache):
    """Test complete end-to-end parallel processing."""
    task = Task(
        type="parallel_processing",
        objective="Process numbers in parallel",
        params={
            "data": list(range(10))  # Process numbers 0-9
        }
    )
    
    # Run the task
    await parallel_agent.run_task(task)
    
    # Verify results from each stage
    splitter_result = parallel_agent.node_results.get("splitter", {})
    assert splitter_result.get("status") == "success"
    
    # Verify processor results
    for i in range(3):
        processor_result = parallel_agent.node_results.get(f"processor_{i}", {})
        if processor_result:  # Some processors might not have run if there weren't enough subtasks
            assert processor_result.get("status") == "success"
            assert "subtask_id" in processor_result.get("result", {})
    
    # Verify final aggregated results
    aggregator_result = parallel_agent.node_results.get("aggregator", {})
    assert aggregator_result.get("status") == "success"
    final_results = aggregator_result.get("result", {}).get("final_results", [])
    assert len(final_results) == 10  # Should have processed all 10 input numbers
    assert all(isinstance(x, int) for x in final_results)  # All results should be integers
    assert all(x == i * 2 for i, x in enumerate(final_results))  # Each number should be doubled

@pytest.mark.asyncio
async def test_error_handling(parallel_agent, mock_cache):
    """Test error handling in parallel processing."""
    # Create a task with invalid data
    task = Task(
        type="parallel_processing",
        objective="Process invalid data",
        params={
            "data": None  # Invalid data should be handled gracefully
        }
    )
    
    # Run the task and expect it to handle errors
    await parallel_agent.run_task(task)
    
    # Verify error handling in results
    splitter_result = parallel_agent.node_results.get("splitter", {})
    assert splitter_result.get("status") in ["success", "error"]
    
    # If splitting succeeded with empty list, processors should handle it
    for i in range(3):
        if f"processor_{i}" in parallel_agent.node_results:
            processor_result = parallel_agent.node_results[f"processor_{i}"]
            assert processor_result.get("status") in ["success", "error"]
    
    # Aggregator should handle empty or error cases
    if "aggregator" in parallel_agent.node_results:
        aggregator_result = parallel_agent.node_results["aggregator"]
        assert aggregator_result.get("status") in ["success", "error"]
        if aggregator_result.get("status") == "success":
            assert aggregator_result.get("result", {}).get("total_items_processed", 0) == 0