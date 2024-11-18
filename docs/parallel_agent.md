# Building a Parallel Processing Agent

This walkthrough will guide you through creating an agent that processes tasks in parallel using the Asimov Agents framework. This type of agent is particularly useful for handling large-scale data processing or multiple independent tasks simultaneously.

## Prerequisites

- Python 3.7+
- Redis server running
- Asimov Agents package installed
- Basic understanding of async/await in Python

## Concepts Covered

1. Parallel Processing
2. Task Splitting
3. Result Aggregation
4. Concurrent Execution
5. State Management

## Step-by-Step Guide

### 1. Setting Up the Project

Create a new Python file `parallel_agent.py` and import the necessary modules:

```python
import asyncio
from typing import Dict, Any, List
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

### 2. Creating the Task Splitter Module

The splitter module divides the main task into smaller subtasks:

```python
class TaskSplitterModule(AgentModule):
    """Splits a large task into smaller subtasks for parallel processing."""
    
    name = "task_splitter"
    type = ModuleType.PLANNER

    async def process(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        task = await cache.get("task")
        data = task.params.get("data", [])
        
        # Split data into chunks for parallel processing
        chunk_size = 3  # Adjust based on your needs
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Create subtasks
        subtasks = []
        for i, chunk in enumerate(chunks):
            subtask = {
                "id": f"subtask_{i}",
                "data": chunk,
                "status": "pending"
            }
            subtasks.append(subtask)
        
        # Store subtasks in cache
        await cache.set("subtasks", subtasks)
        await cache.set("completed_subtasks", [])
        
        return {
            "status": "success",
            "result": f"Created {len(subtasks)} subtasks"
        }
```

Key points:
- Splits input data into manageable chunks
- Creates structured subtasks
- Uses cache for state management
- Initializes tracking for completed subtasks

### 3. Creating the Parallel Processor Module

The processor module handles individual subtasks:

```python
class ParallelProcessorModule(AgentModule):
    """Processes subtasks in parallel."""
    
    name = "parallel_processor"
    type = ModuleType.EXECUTOR

    async def process(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        subtasks = await cache.get("subtasks")
        completed = await cache.get("completed_subtasks")
        
        # Find pending subtasks
        pending = [st for st in subtasks if st["status"] == "pending"]
        
        if not pending:
            return {
                "status": "success",
                "result": "All subtasks completed"
            }
        
        # Process a subtask
        subtask = pending[0]
        
        # Process each item in the subtask
        results = []
        for item in subtask["data"]:
            # Example processing: multiply numbers by 2
            result = item * 2
            results.append(result)
        
        # Update subtask status
        subtask["status"] = "completed"
        subtask["results"] = results
        
        # Update completed subtasks
        completed.append(subtask)
        await cache.set("completed_subtasks", completed)
        
        # Update subtasks list
        for i, st in enumerate(subtasks):
            if st["id"] == subtask["id"]:
                subtasks[i] = subtask
                break
        await cache.set("subtasks", subtasks)
        
        return {
            "status": "success",
            "result": {
                "subtask_id": subtask["id"],
                "processed_items": len(results)
            }
        }
```

Key points:
- Processes one subtask at a time
- Updates task status and results
- Maintains state in cache
- Returns processing statistics

### 4. Creating the Result Aggregator Module

The aggregator module combines results from all processors:

```python
class ResultAggregatorModule(AgentModule):
    """Aggregates results from parallel processing."""
    
    name = "result_aggregator"
    type = ModuleType.EXECUTOR

    async def process(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        completed_subtasks = await cache.get("completed_subtasks")
        
        # Combine all results
        all_results = []
        for subtask in completed_subtasks:
            all_results.extend(subtask["results"])
        
        # Store final results
        await cache.set("final_results", all_results)
        
        return {
            "status": "success",
            "result": {
                "total_items_processed": len(all_results),
                "final_results": all_results
            }
        }
```

Key points:
- Combines results from all subtasks
- Provides summary statistics
- Stores final results in cache
- Returns comprehensive result set

### 5. Setting Up Parallel Flow Control

Configure flow control for parallel execution:

```python
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
```

### 6. Creating Multiple Processor Nodes

Set up multiple processor nodes for parallel execution:

```python
# Create parallel processor nodes
processor_nodes = []
for i in range(3):  # Create 3 parallel processors
    processor_node = Node(
        name=f"processor_{i}",
        type=ModuleType.EXECUTOR,
        modules=[ParallelProcessorModule()],
        dependencies=["splitter"],
        node_config=NodeConfig(
            parallel=True,  # Enable parallel execution
            max_retries=2
        )
    )
    processor_nodes.append(processor_node)
```

### 7. Putting It All Together

Create and run the parallel processing agent:

```python
async def main():
    # Create the agent
    agent = Agent(
        cache=RedisCache(),
        max_concurrent_tasks=3,  # Allow multiple concurrent tasks
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
    
    # Add all nodes to agent
    agent.add_node(splitter_node)
    for node in processor_nodes:
        agent.add_node(node)
    agent.add_node(aggregator_node)
    agent.add_node(flow_control)
    
    # Create and run a task
    task = Task(
        type="parallel_processing",
        objective="Process a list of numbers in parallel",
        params={
            "data": list(range(20))  # Process numbers 0-19
        }
    )
    
    # Run the task
    await agent.run_task(task)

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices for Parallel Agents

1. **Task Splitting**
   - Choose appropriate chunk sizes
   - Consider data dependencies
   - Balance load across processors
   - Handle edge cases

2. **State Management**
   - Use atomic operations
   - Handle race conditions
   - Track progress accurately
   - Clean up temporary state

3. **Resource Management**
   - Control concurrency levels
   - Monitor memory usage
   - Handle timeouts
   - Implement backpressure

4. **Error Handling**
   - Handle partial failures
   - Implement retries
   - Track failed subtasks
   - Provide failure summaries

## Common Patterns

1. **Split-Process-Aggregate**
   - Split large tasks into subtasks
   - Process subtasks independently
   - Aggregate results systematically
   - Handle failures gracefully

2. **Progress Tracking**
   - Track individual subtasks
   - Monitor overall progress
   - Report completion status
   - Handle stragglers

3. **Resource Optimization**
   - Balance load across processors
   - Adjust chunk sizes dynamically
   - Monitor resource usage
   - Implement throttling

4. **Result Management**
   - Combine results efficiently
   - Handle partial results
   - Provide progress updates
   - Clean up temporary data

## Performance Considerations

1. **Chunk Size**
   - Too small: overhead dominates
   - Too large: poor load balancing
   - Consider memory constraints
   - Monitor processing times

2. **Concurrency Level**
   - Match available resources
   - Consider I/O vs CPU bound tasks
   - Monitor system load
   - Implement adaptive scaling

3. **Memory Management**
   - Control memory usage
   - Clean up completed tasks
   - Monitor cache size
   - Implement garbage collection

4. **Cache Usage**
   - Use appropriate data structures
   - Implement TTL for temporary data
   - Monitor cache size
   - Handle cache failures

## Next Steps

1. Implement dynamic chunk sizing
2. Add progress monitoring
3. Implement failure recovery
4. Add performance metrics
5. Optimize resource usage
6. Add support for distributed processing
7. Implement advanced load balancing