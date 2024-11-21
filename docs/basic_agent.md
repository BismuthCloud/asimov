# Building a Basic Agent

This walkthrough will guide you through creating a simple agent using the Asimov Agents framework. We'll build a text processing agent that demonstrates the core concepts of the framework.

## Prerequisites

- Python 3.12+
- Redis server running
- Asimov Agents package installed

## Concepts Covered

1. Agent Module Types
2. Node Configuration
3. Flow Control
4. Task Management
5. Cache Usage

## Step-by-Step Guide

### 1. Setting Up the Project Structure

Create a new Python file `basic_agent.py` with the following imports:

```python
import asyncio
from typing import Dict, Any
from asimov.graph import (
    Agent,
    AgentModule,
    ModuleType,
    Node,
    NodeConfig,
    FlowControlModule,
    FlowControlConfig,
    FlowDecision,
)
from asimov.graph.tasks import Task
from asimov.caches.redis_cache import RedisCache
```

### 2. Creating the Planning Executor Module

The planning executor module is responsible for analyzing the task and creating a plan:

```python
class TextPlannerModule(AgentModule):
    """Plans text processing operations."""
    
    name = "text_planner"
    type = ModuleType.EXECUTOR

    async def process(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        # Get the task from cache
        task = await cache.get("task")
        text = task.params.get("text", "")
        
        # Create a simple plan
        plan = {
            "operations": [
                {"type": "count_words", "text": text},
                {"type": "calculate_stats", "text": text}
            ]
        }
        
        # Store the plan in cache
        await cache.set("plan", plan)
        
        return {
            "status": "success",
            "result": "Plan created successfully"
        }
```

Key points:
- The module inherits from `AgentModule`
- It has a unique name and is an EXECUTOR type
- The `process` method contains the core logic
- Uses cache for state management
- Returns a standardized response format

### 3. Creating the Executor Module

The executor module implements the actual processing logic:

```python
class TextExecutorModule(AgentModule):
    """Executes text processing operations."""
    
    name = "text_executor"
    type = ModuleType.EXECUTOR

    async def process(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        # Get the plan and task
        plan = await cache.get("plan")
        task = await cache.get("task")
        
        results = []
        for operation in plan["operations"]:
            if operation["type"] == "count_words":
                word_count = len(operation["text"].split())
                results.append({
                    "operation": "count_words",
                    "result": word_count
                })
            elif operation["type"] == "calculate_stats":
                char_count = len(operation["text"])
                line_count = len(operation["text"].splitlines())
                results.append({
                    "operation": "calculate_stats",
                    "result": {
                        "characters": char_count,
                        "lines": line_count
                    }
                })
        
        return {
            "status": "success",
            "result": results
        }
```

Key points:
- Implements specific processing operations
- Retrieves state from cache
- Processes each operation in the plan
- Returns structured results

### 4. Setting Up Flow Control

Configure how the agent moves between nodes:

```python
flow_control = Node(
    name="flow_control",
    type=ModuleType.FLOW_CONTROL,
    modules=[FlowControlModule(
        flow_config=FlowControlConfig(
            decisions=[
                FlowDecision(
                    next_node="executor",
                    condition="plan ~= null" # Conditions are lua.
                )
            ],
            default="planner"
        )
    )]
)
```

Key points:
- Uses conditions to determine execution flow
- Provides a default node
- Can be extended with multiple decision paths

### 5. Creating and Configuring the Agent

Put everything together:

```python
async def main():
    # Create the agent
    agent = Agent(
        cache=RedisCache(),
        max_concurrent_tasks=1,
        max_total_iterations=10
    )
    
    # Create nodes
    planner_node = Node(
        name="planner",
        type=ModuleType.EXECUTOR,
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
                        condition="plan ~= null" # Conditions are lua.
                    )
                ],
                default="planner"
            )
        )]
    )
    
    # Add nodes to agent
    agent.add_multiple_nodes([planner_node, executor_node, flow_control])
    
    # Create and run a task
    task = Task(
        type="text_processing",
        objective="Process sample text",
        params={
            "text": "Hello world!\nThis is a sample text."
        }
    )
    
    # Run the task
    await agent.run_task(task)
    
    # Get the results
    results = agent.node_results.get("executor", {}).get("result", [])
    print("\nProcessing Results:")
    for result in results:
        print(f"\nOperation: {result['operation']}")
        print(f"Result: {result['result']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Key points:
- Configure agent with cache and execution limits
- Create and configure nodes
- Set up dependencies between nodes
- Create and run a task
- Handle results

## Understanding the Flow

1. The agent starts with the task in the planner node
2. The planner creates a processing plan
3. Flow control checks if a plan exists
4. If a plan exists, execution moves to the executor node
5. The executor processes the text according to the plan
6. Results are stored and can be retrieved from node_results

## Common Patterns

1. **State Management**
   - Use cache for sharing state between modules
   - Store intermediate results and plans
   - Track progress and status

2. **Module Communication**
   - Modules communicate through the cache
   - Use standardized response formats
   - Handle errors and status consistently

3. **Flow Control**
   - Use conditions to control execution flow
   - Provide default paths
   - Handle edge cases and errors

4. **Error Handling**
   - Use retry mechanisms
   - Return appropriate status codes
   - Include error details in responses

## Next Steps

1. Add more complex processing operations
2. Implement error handling and recovery
3. Add monitoring and logging
4. Explore parallel processing capabilities
5. Integrate with external services

## Best Practices

1. **Module Design**
   - Keep modules focused and single-purpose
   - Use clear naming conventions
   - Document module behavior and requirements

2. **State Management**
   - Use meaningful cache keys
   - Clean up temporary state
   - Handle cache failures gracefully

3. **Flow Control**
   - Keep conditions simple and clear
   - Plan for all possible paths
   - Use meaningful node names

4. **Testing**
   - Test modules in isolation
   - Test different execution paths
   - Verify error handling
   - Test with different input types and sizes
