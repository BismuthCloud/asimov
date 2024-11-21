import pytest
import pytest_asyncio
import asyncio
from typing import Dict, Any, List
from unittest.mock import MagicMock, AsyncMock

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
from examples.llm_agent import (
    LLMPlannerModule,
    LLMExecutorModule,
    LLMFlowControlModule
)

class MockAnthropicResponse:
    def __init__(self, content):
        self.choices = [
            MagicMock(message=MagicMock(content=content))
        ]

class MockAnthropicClient:
    """Mock client for testing LLM interactions."""
    
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.default_response = MockAnthropicResponse("Default mock response")
        
    async def complete(self, prompt: str) -> MockAnthropicResponse:
        # Return predefined response if available, otherwise default
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return MockAnthropicResponse(response)
        return self.default_response

@pytest.fixture
def mock_cache():
    return MockRedisCache()

@pytest.fixture
def mock_anthropic_client():
    return MockAnthropicClient({
        "Create a step-by-step plan": '[{"description": "Research AI agents", "requirements": "Access to documentation", "validation": "Comprehensive notes available"}, {"description": "Write introduction", "requirements": "Research notes", "validation": "Clear introduction exists"}]',
        "Execute this step": "Step executed successfully with the following results:\n1. Actions: Researched AI agents\n2. Outcome: Comprehensive notes created\n3. Output: 5 pages of detailed notes",
        "Evaluate if the step": "success - all validation criteria met",
        "Analyze the execution history": "Analysis complete. Decision: continue - execution is proceeding as expected"
    })

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
    agent = Agent(
        cache=mock_cache,
        max_concurrent_tasks=1,
        max_total_iterations=20
    )
    
    # Create nodes
    planner_node = Node(
        name="planner",
        type=ModuleType.EXECUTOR,
        modules=[planner],
        node_config=NodeConfig(
            parallel=False,
            max_retries=3
        )
    )
    
    executor_node = Node(
        name="executor",
        type=ModuleType.EXECUTOR,
        modules=[executor],
        dependencies=["planner", "flow_control_llm"]
    )
    
    flow_control_node = Node(
        name="flow_control_llm",
        type=ModuleType.FLOW_CONTROL,
        modules=[flow_control_module],
        dependencies=["executor"]
    )
    
    flow_control = Node(
        name="flow_control",
        type=ModuleType.FLOW_CONTROL,
        modules=[FlowControlModule(
            name="flow_control",
            type=ModuleType.FLOW_CONTROL,
            flow_config=FlowControlConfig(
                decisions=[
                    FlowDecision(
                        next_node="executor",
                        condition="plan != null and current_step < len(plan)"
                    ),
                    FlowDecision(
                        next_node="flow_control_llm",
                        condition="execution_history != null"
                    )
                ],
                default="planner"
            )
        )]
    )
    
    # Add nodes to agent
    agent.add_multiple_nodes([planner_node, executor_node, flow_control_node, flow_control])
    return agent

@pytest.mark.asyncio
async def test_llm_agent_initialization(llm_agent):
    """Test that the LLM agent is initialized with correct components."""
    assert len(llm_agent.nodes) == 4
    assert "planner" in llm_agent.nodes
    assert "executor" in llm_agent.nodes
    assert "flow_control_llm" in llm_agent.nodes
    assert "flow_control" in llm_agent.nodes
    
    # Verify node types
    assert llm_agent.nodes["planner"].type == ModuleType.EXECUTOR
    assert llm_agent.nodes["executor"].type == ModuleType.EXECUTOR
    assert llm_agent.nodes["flow_control_llm"].type == ModuleType.FLOW_CONTROL
    assert llm_agent.nodes["flow_control"].type == ModuleType.FLOW_CONTROL

@pytest.mark.asyncio
async def test_llm_planning(llm_agent, mock_cache):
    """Test the LLM-based planning functionality."""
    task = Task(
        type="content_creation",
        objective="Write a blog post",
        params={
            "topic": "AI Testing",
            "length": "500 words"
        }
    )
    
    # Run planner node
    planner_node = llm_agent.nodes["planner"]
    result = await planner_node.process(task.id, mock_cache)
    
    # Verify planning result
    assert result["status"] == "success"
    assert "Plan created successfully" in result["result"]
    
    # Verify plan was stored in cache
    plan = await mock_cache.get("plan")
    assert isinstance(plan, list)
    assert len(plan) > 0
    assert all(key in plan[0] for key in ["description", "requirements", "validation"])

@pytest.mark.asyncio
async def test_llm_execution(llm_agent, mock_cache):
    """Test the LLM-based execution functionality."""
    # Setup test data
    task = Task(
        type="content_creation",
        objective="Write a blog post",
        params={"topic": "AI Testing"}
    )
    plan = [
        {
            "description": "Research AI agents",
            "requirements": "Access to documentation",
            "validation": "Comprehensive notes available"
        }
    ]
    
    # Store test data in cache
    await mock_cache.set("plan", plan)
    await mock_cache.set("current_step", 0)
    await mock_cache.set("task", task)
    
    # Run executor node
    executor_node = llm_agent.nodes["executor"]
    result = await executor_node.process(task.id, mock_cache)
    
    # Verify execution results
    assert result["status"] == "success"
    assert "step" in result["result"]
    assert "execution_result" in result["result"]
    assert "validation" in result["result"]
    assert "success" in result["result"]["validation"].lower()

@pytest.mark.asyncio
async def test_llm_discrimination(llm_agent, mock_cache):
    """Test the LLM-based discrimination functionality."""
    # Setup test data
    plan = [{"description": "Step 1"}, {"description": "Step 2"}]
    execution_history = ["Step 1 completed successfully"]
    
    await mock_cache.set("plan", plan)
    await mock_cache.set("current_step", 1)
    await mock_cache.set("execution_history", execution_history)
    
    # Run discriminator node
    discriminator_node = llm_agent.nodes["discriminator"]
    result = await discriminator_node.process("test_task", mock_cache)
    
    # Verify discrimination results
    assert result["status"] == "success"
    assert "decision" in result["result"]
    assert "analysis" in result["result"]

@pytest.mark.asyncio
async def test_end_to_end_processing(llm_agent, mock_cache):
    """Test complete end-to-end content creation process."""
    task = Task(
        type="content_creation",
        objective="Write a blog post about AI agents",
        params={
            "topic": "AI Agents in Production",
            "length": "1000 words",
            "style": "technical but accessible"
        }
    )
    
    # Run the task
    await llm_agent.run_task(task)
    
    # Verify results from each node
    planner_result = llm_agent.node_results.get("planner", {})
    assert planner_result.get("status") == "success"
    
    executor_result = llm_agent.node_results.get("executor", {})
    assert executor_result.get("status") == "success"
    assert "step" in executor_result.get("result", {})
    
    discriminator_result = llm_agent.node_results.get("discriminator", {})
    assert discriminator_result.get("status") == "success"
    assert "decision" in discriminator_result.get("result", {})

@pytest.mark.asyncio
async def test_error_handling(llm_agent, mock_cache):
    """Test error handling with problematic LLM responses."""
    # Create a client that simulates errors
    error_client = MockAnthropicClient({
        "Create a step-by-step plan": "Invalid JSON response",
        "Execute this step": "Error: Unable to process step",
        "Evaluate if the step": "failure - validation criteria not met"
    })
    
    # Update modules with error client
    llm_agent.nodes["planner"].modules[0].client = error_client
    llm_agent.nodes["executor"].modules[0].client = error_client
    
    task = Task(
        type="content_creation",
        objective="Write a blog post",
        params={"topic": "Error Handling"}
    )
    
    # Run the task and expect it to handle errors gracefully
    await llm_agent.run_task(task)
    
    # Verify error handling in results
    planner_result = llm_agent.node_results.get("planner", {})
    assert planner_result.get("status") in ["success", "error"]
    
    # If planning succeeded despite invalid JSON, executor should handle the error
    if "executor" in llm_agent.node_results:
        executor_result = llm_agent.node_results["executor"]
        assert executor_result.get("status") in ["success", "error"]