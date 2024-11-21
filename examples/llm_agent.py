"""
LLM-Based Agent Example

This example demonstrates how to create an agent that uses Large Language Models (LLMs)
for task planning and execution. The agent will:
1. Use an LLM to analyze and plan a task
2. Execute the plan using LLM-guided steps
3. Use flow control to manage the execution flow and handle LLM responses
"""

import json
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
from asimov.services.inference_clients import AnthropicInferenceClient, ChatMessage, ChatRole


class LLMPlannerModule(AgentModule):
    """Uses LLM to plan task execution."""

    name: str = "llm_planner"
    type: ModuleType = ModuleType.EXECUTOR

    client: AnthropicInferenceClient = None

    def __init__(self):
        super().__init__()
        self.client = AnthropicInferenceClient(
            model="claude-3", api_key="your-api-key"  # Replace with your API key
        )

    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        print(f"{self.name}: Starting planning process")
        task = await cache.get("task")
        print(f"{self.name}: Retrieved task: {task.objective}")

        # Create a planning prompt
        prompt = f"""
        Task Objective: {task.objective}
        Parameters: {task.params}

        Create a step-by-step plan to accomplish this task.
        Format the response as a JSON array of steps, where each step has:
        - description: what needs to be done
        - requirements: any input needed
        - validation: how to verify the step was successful
        """

        # Get plan from LLM
        try:
            print(f"{self.name}: Sending planning request to LLM")
            response = await asyncio.wait_for(
                self.client.get_generation([ChatMessage(role=ChatRole.USER, content=prompt)]), timeout=30.0
            )
            print(f"{self.name}: Received plan from LLM")
        except asyncio.TimeoutError:
            print(f"{self.name}: Timeout waiting for LLM response")
            raise

        # Store the plan
        await cache.set("plan", json.loads(response))
        await cache.set("current_step", 0)

        return {"status": "success", "result": "Plan created successfully"}


class LLMExecutorModule(AgentModule):
    """Executes steps using LLM guidance."""

    name: str = "llm_executor"
    type: ModuleType = ModuleType.EXECUTOR

    client: AnthropicInferenceClient = None

    def __init__(self):
        super().__init__()
        self.client = AnthropicInferenceClient(
            model="claude-3", api_key="your-api-key"  # Replace with your API key
        )

    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        print(f"{self.name}: Starting execution process")
        try:
            # Note the real cache uses jsonpickle and handles serialization and deserialization so you don't need to do this.
            plan = json.loads(await cache.get("plan"))
            current_step = await cache.get("current_step")
            task = await cache.get("task")
            print(f"{self.name}: Retrieved plan and current step {current_step}")
        except asyncio.TimeoutError:
            print(f"{self.name}: Timeout retrieving task data")
            raise

        if current_step >= len(plan):
            return {"status": "success", "result": "All steps completed"}

        step = plan[current_step]

        print(step)

        # Create execution prompt
        prompt = f"""
        Task: {task.objective}
        Current Step: {step['description']}
        Requirements: {step['requirements']}

        Execute this step and provide the results.
        Include:
        1. The actions taken
        2. The outcome
        3. Any relevant output or artifacts
        """

        # Execute step with LLM
        try:
            print(f"{self.name}: Sending execution request to LLM")
            result = await asyncio.wait_for(
                self.client.get_generation([ChatMessage(role=ChatRole.USER, content=prompt)]), timeout=30.0
            )
            print(f"{self.name}: Received execution result from LLM")
        except asyncio.TimeoutError:
            print(f"{self.name}: Timeout waiting for LLM execution response")
            raise

        # Validate step
        validation_prompt = f"""
        Step: {step['description']}
        Validation Criteria: {step['validation']}
        Result: {result}

        Evaluate if the step was completed successfully.
        Return either "success" or "failure" with a brief explanation.
        """

        try:
            print(f"{self.name}: Sending validation request to LLM")
            validation_result = await asyncio.wait_for(
                self.client.get_generation([ChatMessage(role=ChatRole.USER, content=validation_prompt)]), timeout=30.0
            )
            print(f"{self.name}: Received validation result from LLM")
        except asyncio.TimeoutError:
            print(f"{self.name}: Timeout waiting for LLM validation response")
            raise

        if "success" in validation_result.lower():
            current_step += 1
            await cache.set("current_step", current_step)
            status = "success"
        else:
            status = "error"

        return {
            "status": status,
            "result": {
                "step": step["description"],
                "execution_result": result,
                "validation": validation_result,
            },
        }


class LLMFlowControlModule(AgentModule):
    """Makes decisions about execution flow based on LLM analysis."""

    name: str = "llm_flow_control"
    type: ModuleType = ModuleType.FLOW_CONTROL

    client: AnthropicInferenceClient = None

    def __init__(self):
        super().__init__()
        self.client = AnthropicInferenceClient(
            model="claude-3", api_key="your-api-key"  # Replace with your API key
        )

    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        print(f"{self.name}: Starting flow control process")
        plan = await cache.get("plan")
        current_step = await cache.get("current_step")
        execution_history = await cache.get("execution_history", [])
        print(
            f"{self.name}: Retrieved plan and history with {len(execution_history)} entries"
        )

        if not execution_history:
            return {
                "status": "success",
                "result": {"decision": "continue", "reason": "No history to analyze"},
            }

        # Create analysis prompt
        prompt = f"""
        Execution History: {execution_history}
        Current Step: {current_step} of {len(plan)} steps

        Analyze the execution history and determine if we should:
        1. continue: proceed with the next step
        2. retry: retry the current step
        3. replan: create a new plan
        4. abort: stop execution

        Provide your decision and reasoning.
        """

        try:
            print(f"{self.name}: Sending analysis request to LLM")
            response = await asyncio.wait_for(
                self.client.complete(prompt), timeout=30.0
            )
            analysis = response.choices[0].message.content
            print(f"{self.name}: Received analysis result from LLM")
        except asyncio.TimeoutError:
            print(f"{self.name}: Timeout waiting for LLM analysis response")
            raise

        return {
            "status": "success",
            "result": {
                "analysis": analysis,
                "decision": "continue",  # Extract actual decision from analysis
            },
        }


async def main():
    print("Starting LLM agent example")
    # Create the agent
    agent = Agent(cache=RedisCache(), max_concurrent_tasks=1, max_total_iterations=20)
    print("Agent created with Redis cache")

    # Create nodes
    planner_node = Node(
        name="planner",
        type=ModuleType.EXECUTOR,
        modules=[LLMPlannerModule()],
        node_config=NodeConfig(parallel=False, max_retries=3),
    )

    executor_node = Node(
        name="executor",
        type=ModuleType.EXECUTOR,
        modules=[LLMExecutorModule()],
        dependencies=["planner"],
    )

    flow_control = Node(
        name="flow_control",
        type=ModuleType.FLOW_CONTROL,
        modules=[
            FlowControlModule(
                name="flow_control",
                type=ModuleType.FLOW_CONTROL,
                flow_config=FlowControlConfig(
                    decisions=[
                        FlowDecision(
                            next_node="executor",
                            condition="plan ~= nil and current_step < #plan",
                        ),
                        FlowDecision(
                            next_node="flow_control",
                            condition="execution_history ~= nil",
                        ),
                    ],
                    default="planner",
                ),
            )
        ],
    )

    # Add nodes to agent
    agent.add_multiple_nodes(
        [planner_node, executor_node, flow_control]
    )

    # Create and run a task
    task = Task(
        type="content_creation",
        objective="Write a blog post about AI agents",
        params={
            "topic": "AI Agents in Production",
            "length": "1000 words",
            "style": "technical but accessible",
            "key_points": [
                "Definition of AI agents",
                "Common architectures",
                "Real-world applications",
                "Future trends",
            ],
        },
    )

    # Run the task
    await agent.run_task(task)

    # Print results
    print("\nTask Execution Results:")
    for node, result in agent.node_results.items():
        print(f"\nNode: {node}")
        print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
