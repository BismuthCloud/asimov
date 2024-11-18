import pytest
import asyncio
from typing import Dict, Any

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
from examples.basic_agent import TextPlannerModule, TextExecutorModule

@pytest.fixture
def mock_cache():
    return MockRedisCache()

@pytest.fixture
def basic_agent(mock_cache):
    agent = Agent(
        cache=mock_cache,
        max_concurrent_tasks=1,
        max_total_iterations=10
    )
    
    # Create nodes
    planner_node = Node(
        name="planner",
        type=ModuleType.PLANNER,
        modules=[TextPlannerModule()],
        node_config=NodeConfig(
            parallel=False,
            max_retries=3
        )
    )
    
    executor_node = Node(
        name="executor",
        type=ModuleType.EXECUTOR,
        modules=[TextExecutorModule()],
        dependencies=["planner"]
    )
    
    flow_control = Node(
        name="flow_control",
        type=ModuleType.FLOW_CONTROL,
        modules=[FlowControlModule(
            flow_config=FlowControlConfig(
                decisions=[
                    FlowDecision(
                        next_node="executor",
                        condition="plan != null"
                    )
                ],
                default="planner"
            )
        )]
    )
    
    # Add nodes to agent
    agent.add_multiple_nodes([planner_node, executor_node, flow_control])
    return agent

@pytest.mark.asyncio
async def test_basic_agent_initialization(basic_agent):
    """Test that the basic agent is initialized with correct components."""
    assert len(basic_agent.nodes) == 3
    assert "planner" in basic_agent.nodes
    assert "executor" in basic_agent.nodes
    assert "flow_control" in basic_agent.nodes
    
    # Verify node types
    assert basic_agent.nodes["planner"].type == ModuleType.PLANNER
    assert basic_agent.nodes["executor"].type == ModuleType.EXECUTOR
    assert basic_agent.nodes["flow_control"].type == ModuleType.FLOW_CONTROL

@pytest.mark.asyncio
async def test_text_planning(basic_agent, mock_cache):
    """Test the text planning functionality."""
    task = Task(
        type="text_processing",
        objective="Test planning",
        params={"text": "Hello world"}
    )
    
    # Run just the planner node
    planner_node = basic_agent.nodes["planner"]
    await planner_node.process(task.id, mock_cache)
    
    # Verify plan was created and stored
    plan = await mock_cache.get("plan")
    assert plan is not None
    assert "operations" in plan
    assert len(plan["operations"]) == 2
    assert plan["operations"][0]["type"] == "count_words"
    assert plan["operations"][1]["type"] == "calculate_stats"

@pytest.mark.asyncio
async def test_text_execution(basic_agent, mock_cache):
    """Test the text execution functionality."""
    # Setup test data
    test_plan = {
        "operations": [
            {"type": "count_words", "text": "Hello world"},
            {"type": "calculate_stats", "text": "Hello world"}
        ]
    }
    test_task = Task(
        type="text_processing",
        objective="Test execution",
        params={"text": "Hello world"}
    )
    
    # Store test data in cache
    await mock_cache.set("plan", test_plan)
    await mock_cache.set("task", test_task)
    
    # Run executor node
    executor_node = basic_agent.nodes["executor"]
    result = await executor_node.process(test_task.id, mock_cache)
    
    # Verify results
    assert result["status"] == "success"
    assert len(result["result"]) == 2
    
    word_count_result = result["result"][0]
    assert word_count_result["operation"] == "count_words"
    assert word_count_result["result"] == 2  # "Hello world" has 2 words
    
    stats_result = result["result"][1]
    assert stats_result["operation"] == "calculate_stats"
    assert stats_result["result"]["characters"] == 11  # Length of "Hello world"
    assert stats_result["result"]["lines"] == 1

@pytest.mark.asyncio
async def test_end_to_end_processing(basic_agent, mock_cache):
    """Test complete end-to-end text processing."""
    task = Task(
        type="text_processing",
        objective="Process sample text",
        params={
            "text": "Hello world!\nThis is a sample text.\nIt demonstrates the basic agent functionality."
        }
    )
    
    # Run the task
    await basic_agent.run_task(task)
    
    # Get final results from the executor node
    result = basic_agent.node_results.get("executor", {})
    assert result.get("status") == "success"
    
    results = result.get("result", [])
    assert len(results) == 2
    
    # Verify word count operation
    word_count_result = results[0]
    assert word_count_result["operation"] == "count_words"
    assert word_count_result["result"] == 13  # Count of words in the test text
    
    # Verify stats operation
    stats_result = results[1]
    assert stats_result["operation"] == "calculate_stats"
    assert stats_result["result"]["lines"] == 3  # Number of lines in test text
    assert stats_result["result"]["characters"] > 0  # Should have characters

@pytest.mark.asyncio
async def test_error_handling(basic_agent, mock_cache):
    """Test error handling with invalid input."""
    task = Task(
        type="text_processing",
        objective="Process invalid input",
        params={}  # Missing required 'text' parameter
    )
    
    # Run the task and expect it to complete (even with empty text)
    await basic_agent.run_task(task)
    
    # Verify results still contain expected structure
    result = basic_agent.node_results.get("executor", {})
    assert result.get("status") == "success"
    
    results = result.get("result", [])
    assert len(results) == 2
    
    # Both operations should handle empty text gracefully
    word_count_result = results[0]
    assert word_count_result["result"] == 0  # Empty text has 0 words
    
    stats_result = results[1]
    assert stats_result["result"]["characters"] == 0
    assert stats_result["result"]["lines"] == 0