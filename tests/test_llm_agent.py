import pytest
import pytest_asyncio
import asyncio
import os
from typing import Dict, Any, List
from unittest.mock import MagicMock, AsyncMock, patch
import json

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
from examples.llm_agent import LLMPlannerModule, LLMExecutorModule, LLMFlowControlModule


class MockAnthropicClient:
    """Mock client for testing LLM interactions."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.default_response = {}

    async def get_generation(self, messages: List[Any]) -> str:
        # Extract prompt from messages
        prompt = messages[-1].content if messages else ""
        # Return predefined response if available, otherwise default
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        return json.dumps(self.default_response)


@pytest.fixture(autouse=True)
def setup_env():
    """Set up test environment variables."""
    os.environ["ANTHROPIC_API_KEY"] = "test-api-key"
    yield
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]


@pytest.fixture
def mock_cache():
    return MockRedisCache()


@pytest.fixture
def mock_anthropic_client():
    return MockAnthropicClient(
        {
            "Create a step-by-step plan": '{"steps": [{"description": "Gather ingredients and equipment", "requirements": "Recipe ingredients list and tools", "validation": "All ingredients and tools are present and measured"}, {"description": "Prepare cake batter", "requirements": "Ingredients and mixing equipment", "validation": "Batter has proper consistency and ingredients are well combined"}]}',
            "Execute this step": "Step executed successfully with the following results:\n1. Actions: Gathered and measured all ingredients\n2. Outcome: All ingredients and tools ready for baking\n3. Output: Ingredients measured and organized according to recipe",
            "Evaluate if the step": "success - all ingredients are properly measured and equipment is ready",
            "Analyze the execution history": "Analysis complete. Decision: continue - ingredient preparation is complete and accurate",
        }
    )


@pytest.fixture
def llm_agent(mock_cache, mock_anthropic_client):
    # Create modules with mock client
    planner = LLMPlannerModule()
    planner.client = mock_anthropic_client

    executor = LLMExecutorModule()
    executor.client = mock_anthropic_client

    flow_control_module = LLMFlowControlModule()
    flow_control_module.client = mock_anthropic_client

    # Create agent
    agent = Agent(cache=mock_cache, max_concurrent_tasks=1, max_total_iterations=20)

    # Create nodes
    planner_node = Node(
        name="planner",
        type=ModuleType.EXECUTOR,
        modules=[planner],
        node_config=NodeConfig(parallel=False, max_retries=3),
    )

    executor_node = Node(
        name="executor",
        type=ModuleType.EXECUTOR,
        modules=[executor],
        dependencies=["planner"],
    )

    flow_control_node = Node(
        name="flow_control_llm",
        type=ModuleType.FLOW_CONTROL,
        modules=[flow_control_module],
        dependencies=["executor"],
    )

    # Add nodes to agent
    agent.add_multiple_nodes([planner_node, executor_node, flow_control_node])
    return agent


@pytest.mark.asyncio
async def test_llm_agent_initialization(llm_agent):
    """Test that the LLM agent is initialized with correct components."""
    assert len(llm_agent.nodes) == 3
    assert "planner" in llm_agent.nodes
    assert "executor" in llm_agent.nodes
    assert "flow_control_llm" in llm_agent.nodes

    # Verify node types
    assert llm_agent.nodes["planner"].type == ModuleType.EXECUTOR
    assert llm_agent.nodes["executor"].type == ModuleType.EXECUTOR
    assert llm_agent.nodes["flow_control_llm"].type == ModuleType.FLOW_CONTROL


@pytest.mark.asyncio
async def test_llm_planning(llm_agent, mock_cache):
    """Test the LLM-based planning functionality."""
    task = Task(
        type="cake_baking",
        objective="Bake a chocolate cake",
        params={"cake_type": "Chocolate", "servings": "8-10", "difficulty": "intermediate"},
    )

    await mock_cache.set("task", task)

    # Run planner node
    planner_node = llm_agent.nodes["planner"]
    result = await planner_node.run(mock_cache, asyncio.Semaphore())

    # Verify planning result
    assert result["status"] == "success"

    print(result)
    assert "Plan created successfully" in result["results"][0]["result"]

    # Verify plan was stored in cache
    plan = await mock_cache.get("plan")

    print(plan)
    assert isinstance(plan, list)
    assert len(plan) > 0
    assert all(key in plan[0] for key in ["description", "requirements", "validation"])


@pytest.mark.asyncio
async def test_llm_execution(llm_agent, mock_cache):
    """Test the LLM-based execution functionality."""
    # Setup test data
    task = Task(
        type="cake_baking",
        objective="Bake a chocolate cake",
        params={"cake_type": "Chocolate", "servings": "8-10", "difficulty": "intermediate"},
    )
    plan = json.dumps(
        [
            {
                "description": "Gather ingredients and equipment",
                "requirements": "Recipe ingredients list and tools",
                "validation": "All ingredients and tools are present and measured",
            }
        ]
    )

    # Store test data in cache
    await mock_cache.set("plan", plan)
    await mock_cache.set("current_step", 0)
    await mock_cache.set("task", task)

    # Run executor node
    executor_node = llm_agent.nodes["executor"]
    result = await executor_node.run(mock_cache, asyncio.Semaphore())

    result = result["results"][0]

    # Verify execution results
    assert result["status"] == "success"
    assert "step" in result["result"]
    assert "execution_result" in result["result"]
    assert "validation" in result["result"]
    assert "success" in result["result"]["validation"].lower()

    # Verify execution history was updated
    execution_history = await mock_cache.get("execution_history")
    assert execution_history is not None
    assert len(execution_history) == 1  # Should have one entry after first execution

    history_entry = execution_history[0]
    assert "step" in history_entry
    assert "execution_result" in history_entry
    assert "validation" in history_entry
    assert "status" in history_entry
    assert "timestamp" in history_entry
    assert history_entry["status"] == "success"
    assert history_entry["step"] == "Gather ingredients and equipment"


@pytest.mark.asyncio
@pytest.mark.timeout(30)  # 30 second timeout
async def test_end_to_end_processing(llm_agent, mock_cache):
    print("Starting end-to-end LLM agent test")
    """Test complete end-to-end cake baking process."""
    task = Task(
        type="cake_baking",
        objective="Bake a chocolate cake",
        params={
            "cake_type": "Chocolate",
            "servings": "8-10",
            "difficulty": "intermediate",
            "style": "classic recipe",
        },
    )

    # Run the task
    print(f"Running task: {task.objective}")
    try:
        await asyncio.wait_for(
            llm_agent.run_task(task), timeout=25.0
        )  # 25 second timeout
        print("Task completed successfully")
    except asyncio.TimeoutError:
        print("Task execution timed out")
        raise
    except Exception as e:
        print(f"Task execution failed: {str(e)}")
        raise

    print("Verifying node results")
    # Verify results from each node
    planner_result = llm_agent.node_results.get("planner", {})
    assert planner_result.get("status") == "success"

    executor_result = llm_agent.node_results.get("executor", {})
    assert executor_result.get("status") == "success"
    result = executor_result.get("results", [])[0]["result"]
    assert "step" in result


@pytest.mark.asyncio
async def test_missing_api_key():
    """Test that modules properly handle missing API key."""
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]

    with pytest.raises(
        ValueError, match="ANTHROPIC_API_KEY environment variable must be set"
    ):
        LLMPlannerModule()

    with pytest.raises(
        ValueError, match="ANTHROPIC_API_KEY environment variable must be set"
    ):
        LLMExecutorModule()

    with pytest.raises(
        ValueError, match="ANTHROPIC_API_KEY environment variable must be set"
    ):
        LLMFlowControlModule()


@pytest.mark.asyncio
async def test_error_handling(llm_agent, mock_cache):
    """Test error handling with problematic LLM responses."""
    # Create a client that simulates errors
    error_client = MockAnthropicClient(
        {
            "Create a step-by-step plan": "Invalid JSON response",
            "Execute this step": "Error: Unable to process step",
            "Evaluate if the step": "failure - validation criteria not met",
        }
    )

    # Update modules with error client
    llm_agent.nodes["planner"].modules[0].client = error_client
    llm_agent.nodes["executor"].modules[0].client = error_client

    task = Task(
        type="cake_baking",
        objective="Bake a chocolate cake",
        params={"cake_type": "Chocolate", "servings": "8-10", "difficulty": "intermediate"},
    )

    # Run the task and expect it to handle errors gracefully
    await llm_agent.run_task(task)

    # Verify error handling in results
    planner_result = llm_agent.node_results.get("planner", {})

    print(llm_agent.node_results)
    assert planner_result.get("status") in ["success", "error"]

    # If planning succeeded despite invalid JSON, executor should handle the error
    if "executor" in llm_agent.node_results:
        executor_result = llm_agent.node_results["executor"]
        assert executor_result.get("status") in ["success", "error"]
