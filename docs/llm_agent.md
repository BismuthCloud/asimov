# Building an LLM-Based Agent

This walkthrough will guide you through creating an agent that uses Large Language Models (LLMs) for task planning and execution. This type of agent is particularly useful for complex tasks that require natural language understanding and generation.

## Prerequisites

- Python 3.12+
- Redis server running
- Asimov Agents package installed
- API key for an LLM provider (e.g., Anthropic, AWS Bedrock)

## Concepts Covered

1. LLM Integration
2. Complex Task Planning
3. Dynamic Execution
4. Result Validation
5. Decision Making with LLMs

## Step-by-Step Guide

### 1. Setting Up the Project

Create a new Python file `llm_agent.py` and import the necessary modules:

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
from asimov.services.inference_clients import AnthropicInferenceClient
```

### 2. Creating the Planning Executor Module

The planning executor module uses an LLM to analyze tasks and create execution plans:

```python
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
        print(await cache.keys())
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
```

Key points:
- Initializes LLM client in constructor
- Creates structured prompts for the LLM
- Stores plan and progress in cache
- Uses standardized response format

### 3. Creating the LLM Executor Module

The executor module uses LLM to perform task steps:

```python
class LLMExecutorModule(AgentModule):
    """Executes steps using LLM guidance."""
    
    name = "llm_executor"
    type = ModuleType.EXECUTOR

    def __init__(self):
        super().__init__()
        self.client = AnthropicInferenceClient(
            model="claude-3",
            api_key="your-api-key"
        )

    async def process(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        plan = await cache.get("plan")
        current_step = await cache.get("current_step")
        task = await cache.get("task")
        
        if current_step >= len(plan):
            return {
                "status": "success",
                "result": "All steps completed"
            }
        
        step = plan[current_step]
        
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
        response = await self.client.complete(prompt)
        result = response.choices[0].message.content
        
        # Validate step
        validation_prompt = f"""
        Step: {step['description']}
        Validation Criteria: {step['validation']}
        Result: {result}
        
        Evaluate if the step was completed successfully.
        Return either "success" or "failure" with a brief explanation.
        """
        
        validation = await self.client.complete(validation_prompt)
        validation_result = validation.choices[0].message.content
        
        if "success" in validation_result.lower():
            current_step += 1
            await cache.set("current_step", current_step)
            status = "success"
        else:
            status = "error"
        
        return {
            "status": status,
            "result": {
                "step": step['description'],
                "execution_result": result,
                "validation": validation_result
            }
        }
```

Key points:
- Executes one step at a time
- Uses LLM for both execution and validation
- Maintains progress through steps
- Includes detailed results and validation

### 4. Adding a Flow Control Module

The flow control module makes decisions about execution flow:

```python
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
```

Key points:
- Uses LLM for decision making
- Analyzes execution history
- Provides reasoning with decisions
- Supports multiple decision types

### 5. Configuring Flow Control

Set up flow control with LLM-aware conditions:

```python
flow_control = Node(
    name="flow_control",
    type=ModuleType.FLOW_CONTROL,
    modules=[FlowControlModule(
        flow_config=FlowControlConfig(
            decisions=[
                FlowDecision(
                    next_node="executor",
                    condition="plan ~= nil and current_step < #plan" # Conditions are lua
                ),
                FlowDecision(
                    next_node="flow_control",
                    condition="execution_history ~= nil" # Conditions are lua 
                )
            ],
            default="planner"
        )
    )]
)
```

### 6. Putting It All Together

Create and run the agent:

```python
async def main():
    # Create the agent
    agent = Agent(
        cache=RedisCache(),
        max_concurrent_tasks=1,
        max_total_iterations=20
    )
    
    # Create and add nodes
    planner_node = Node(
        name="planner",
        type=ModuleType.EXECUTOR,
        modules=[LLMPlannerModule()],
        node_config=NodeConfig(
            parallel=False,
            max_retries=3
        )
    )
    
    executor_node = Node(
        name="executor",
        type=ModuleType.EXECUTOR,
        modules=[LLMExecutorModule()],
        dependencies=["planner", "flow_control"]
    )
    
    flow_control_node = Node(
        name="flow_control",
        type=ModuleType.FLOW_CONTROL,
        modules=[LLMFlowControlModule()],
        dependencies=["executor"]
    )
    
    # Add nodes to agent
    agent.add_multiple_nodes([
        planner_node,
        executor_node,
        flow_control_node,
        flow_control
    ])
    
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
                "Future trends"
            ]
        }
    )
    
    # Run the task
    await agent.run_task(task)

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices for LLM Agents

1. **Prompt Engineering**
   - Be specific and structured in prompts
   - Include relevant context
   - Request structured outputs
   - Validate LLM responses

2. **Error Handling**
   - Handle LLM API errors
   - Implement retries for failed requests
   - Validate LLM outputs
   - Have fallback strategies

3. **State Management**
   - Track progress through steps
   - Store intermediate results
   - Maintain execution history
   - Clean up temporary state

4. **Cost Management**
   - Cache LLM responses when appropriate
   - Use smaller models for simpler tasks
   - Batch similar requests
   - Monitor token usage

5. **Performance**
   - Use streaming for long responses
   - Implement timeouts
   - Consider parallel processing
   - Cache frequently used prompts

## Common Patterns

1. **Plan-Execute-Validate**
   - Create detailed plans
   - Execute steps systematically
   - Validate results
   - Adjust plans based on feedback

2. **Progressive Refinement**
   - Start with high-level plans
   - Refine details progressively
   - Validate intermediate results
   - Adjust based on outcomes

3. **Decision Making**
   - Use LLMs for complex decisions
   - Provide context and history
   - Include decision criteria
   - Document reasoning

4. **Error Recovery**
   - Detect errors early
   - Implement retry strategies
   - Consider alternative approaches
   - Learn from failures

## Next Steps

1. Implement more sophisticated planning strategies
2. Add support for multiple LLM providers
3. Implement caching for LLM responses
4. Add monitoring and logging
5. Implement cost tracking and optimization
6. Add support for streaming responses
7. Implement parallel processing for suitable tasks
