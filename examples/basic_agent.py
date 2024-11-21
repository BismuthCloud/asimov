"""
Basic Agent Example

This example demonstrates how to create a simple agent that processes text using
a planner and executor pattern. The agent will:
1. Plan how to process the text (e.g., identify operations needed)
2. Execute the planned operations
3. Use flow control to manage the execution flow
"""

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
    Cache,
)
from asimov.graph.tasks import Task
from asimov.caches.redis_cache import RedisCache


class TextPlannerModule(AgentModule):
    """Plans text processing operations."""
    
    name: str = "text_planner"
    type: ModuleType = ModuleType.EXECUTOR

    async def process(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        print(f"{self.name}: Starting planning process")
        # Get the task from cache
        task = await cache.get("task")
        text = task.params.get("text", "")
        print(f"{self.name}: Retrieved task with text length {len(text)}")
        
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


class TextExecutorModule(AgentModule):
    """Executes text processing operations."""
    
    name: str = "text_executor"
    type: ModuleType = ModuleType.EXECUTOR

    async def process(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        print(f"{self.name}: Starting execution process")
        # Get the plan and task
        plan = await cache.get("plan")
        task = await cache.get("task")
        print(f"{self.name}: Retrieved plan with {len(plan['operations'])} operations")
        
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


async def main():
    print("Starting basic agent example")
    # Create the agent
    agent = Agent(
        cache=RedisCache(),
        max_concurrent_tasks=1,
        max_total_iterations=10
    )
    print("Agent created with Redis cache")
    
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
                        condition="plan != null"
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
            "text": "Hello world!\nThis is a sample text.\nIt demonstrates the basic agent functionality."
        }
    )
    
    # Run the task
    await agent.run_task(task)
    
    # Get the final results
    results = agent.node_results.get("executor", {}).get("result", [])
    print("\nProcessing Results:")
    for result in results:
        print(f"\nOperation: {result['operation']}")
        print(f"Result: {result['result']}")


if __name__ == "__main__":
    asyncio.run(main())