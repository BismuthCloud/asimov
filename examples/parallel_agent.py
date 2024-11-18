"""
Parallel Processing Agent Example

This example demonstrates how to create an agent that processes multiple tasks
in parallel using the framework's parallel execution capabilities. The agent will:
1. Split a large task into subtasks
2. Process subtasks in parallel
3. Aggregate results from parallel processing
"""

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
        
        # Simulate processing with some async work
        await asyncio.sleep(1)  # Simulate work
        
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
    
    aggregator_node = Node(
        name="aggregator",
        type=ModuleType.EXECUTOR,
        modules=[ResultAggregatorModule()],
        dependencies=[node.name for node in processor_nodes],
        node_config=NodeConfig(
            parallel=False,
            condition="len(completed_subtasks) == len(subtasks)"  # Only run when all subtasks are done
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
    
    # Get final results
    final_results = agent.node_results.get("aggregator", {}).get("result", {})
    print("\nProcessing Results:")
    print(f"Total items processed: {final_results.get('total_items_processed', 0)}")
    print(f"Results: {final_results.get('final_results', [])}")


if __name__ == "__main__":
    asyncio.run(main())