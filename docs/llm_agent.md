# Building an LLM-Based Agent

This walkthrough will guide you through creating an agent that uses Large Language Models (LLMs) for task planning and execution. This type of agent is particularly useful for complex tasks that require natural language understanding and generation.

## Prerequisites

- Python 3.12+
- Redis server running (see Redis Setup below)
- Asimov Agents package installed
- API key for an LLM provider (e.g., Anthropic, AWS Bedrock) set in the ANTHROPIC_API_KEY environment variable (see Environment Setup below)

## Environment Setup

### API Key Configuration

Before running the agent, you need to set up your Anthropic API key as an environment variable:

```bash
# Linux/macOS
export ANTHROPIC_API_KEY="your-api-key"

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=your-api-key

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="your-api-key"
```

The agent will automatically use this environment variable for authentication with the Anthropic API.

> **Security Note**: Never hardcode API keys in your source code. Always use environment variables or secure configuration management systems to handle sensitive credentials. This helps prevent accidental exposure of API keys in version control systems or logs.

### Redis Setup with Docker

To run Redis using Docker:

```bash
# Pull the official Redis image
docker pull redis:latest

# Run Redis container
docker run --name asimov-redis -d -p 6379:6379 redis:latest

# Verify Redis is running
docker ps | grep asimov-redis
```

This will start Redis on the default port 6379. The container will run in the background and restart automatically unless explicitly stopped.

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
import os
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
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
        self.client = AnthropicInferenceClient(
            model="claude-3-5-sonnet-20241022", api_key=api_key
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
- Initializes LLM client in constructor with API key from environment variable
- Creates structured prompts for the LLM
- Stores plan and progress in cache
- Uses standardized response format
- Implements secure credential handling

### 3. Creating the LLM Executor Module

The executor module uses LLM to perform task steps:

```python
class LLMExecutorModule(AgentModule):
    """Executes steps using LLM guidance."""
    
    name = "llm_executor"
    type = ModuleType.EXECUTOR

    def __init__(self):
        super().__init__()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
        self.client = AnthropicInferenceClient(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key
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

### 4. Flow Control Options

The framework provides two types of flow control modules:

1. Basic Flow Control Module
2. Agent-Directed Flow Control Module

#### Basic Flow Control Module

The basic flow control module uses Lua conditions to make routing decisions:

```python
flow_control = Node(
    name="flow_control",
    type=ModuleType.FLOW_CONTROL,
    modules=[FlowControlModule(
        flow_config=FlowControlConfig(
            decisions=[
                FlowDecision(
                    next_node="executor",
                    condition="plan ~= nil and current_step < #plan" # Lua conditions
                ),
                FlowDecision(
                    next_node="flow_control",
                    condition="execution_history ~= nil" 
                )
            ],
            default="planner"
        )
    )]
)
```

Key points:
- Uses Lua conditions for decision making
- Supports multiple decision paths
- Can access cache variables in conditions
- Has a default fallback path

#### Agent-Directed Flow Control Module

The agent-directed flow control module uses LLMs to make intelligent routing decisions:

```python
from asimov.graph.agent_directed_flow import (
    AgentDirectedFlowControl,
    AgentDrivenFlowControlConfig,
    AgentDrivenFlowDecision,
    Example
)

flow_control = Node(
    name="flow_control",
    type=ModuleType.FLOW_CONTROL,
    modules=[AgentDirectedFlowControl(
        inference_client=AnthropicInferenceClient(...),
        system_description="A system that handles various content creation tasks",
        flow_config=AgentDrivenFlowControlConfig(
            decisions=[
                AgentDrivenFlowDecision(
                    next_node="blog_writer",
                    metadata={"description": "Writes blog posts on technical topics"},
                    examples=[
                        Example(
                            message="Write a blog post about AI agents",
                            choices=[
                                {"choice": "blog_writer", "description": "Writes blog posts"},
                                {"choice": "code_writer", "description": "Writes code"}
                            ],
                            choice="blog_writer",
                            reasoning="The request is specifically for blog content"
                        )
                    ]
                ),
                AgentDrivenFlowDecision(
                    next_node="code_writer",
                    metadata={"description": "Writes code examples and tutorials"},
                    examples=[
                        Example(
                            message="Create a Python script for data processing",
                            choices=[
                                {"choice": "blog_writer", "description": "Writes blog posts"},
                                {"choice": "code_writer", "description": "Writes code"}
                            ],
                            choice="code_writer",
                            reasoning="The request is for code creation"
                        )
                    ]
                )
            ],
            default="error_handler"
        )
    )]
)
```

Key points:
- Uses LLM for intelligent decision making
- Supports example-based learning
- Can include reasoning for decisions
- Maintains context awareness
- Handles complex routing scenarios
- Supports metadata for rich decision context

### 5. Configuring Flow Control

Choose the appropriate flow control based on your needs:

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

### 6. Example Implementations

#### Basic Task Planning Agent

Create a basic agent that uses LLM for task planning and execution:

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
        flow_control_node
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
    
    await agent.run_task(task)
```

#### Intelligent Content Router

Create an agent that uses AgentDirectedFlow to route content creation tasks:

```python
async def main():
    # Create the agent
    agent = Agent(
        cache=RedisCache(default_prefix=str(uuid4())),
        max_concurrent_tasks=1,
        max_total_iterations=20
    )
    
    # Set up specialized content nodes
    blog_writer = Node(
        name="blog_writer",
        type=ModuleType.EXECUTOR,
        modules=[BlogWriterModule()]
    )
    
    code_writer = Node(
        name="code_writer",
        type=ModuleType.EXECUTOR,
        modules=[CodeWriterModule()]
    )
    
    error_handler = Node(
        name="error_handler",
        type=ModuleType.EXECUTOR,
        modules=[ErrorHandlerModule()]
    )
    
    # Create flow control with agent-directed decisions
    flow_control = Node(
        name="flow_control",
        type=ModuleType.FLOW_CONTROL,
        modules=[AgentDirectedFlowControl(
            inference_client=AnthropicInferenceClient(...),
            system_description="A system that handles various content creation tasks",
            flow_config=AgentDrivenFlowControlConfig(
                decisions=[
                    AgentDrivenFlowDecision(
                        next_node="blog_writer",
                        metadata={"description": "Writes blog posts"},
                        examples=[
                            Example(
                                message="Write a blog post about AI",
                                choices=[
                                    {"choice": "blog_writer", "description": "Writes blogs"},
                                    {"choice": "code_writer", "description": "Writes code"}
                                ],
                                choice="blog_writer",
                                reasoning="Request is for blog content"
                            )
                        ]
                    ),
                    AgentDrivenFlowDecision(
                        next_node="code_writer",
                        metadata={"description": "Writes code"},
                        examples=[
                            Example(
                                message="Create a sorting algorithm",
                                choices=[
                                    {"choice": "blog_writer", "description": "Writes blogs"},
                                    {"choice": "code_writer", "description": "Writes code"}
                                ],
                                choice="code_writer",
                                reasoning="Request is for code implementation"
                            )
                        ]
                    )
                ],
                default="error_handler"
            )
        )]
    )
    
    # Add all nodes to agent
    agent.add_multiple_nodes([
        blog_writer,
        code_writer,
        error_handler,
        flow_control
    ])
    
    # Example tasks
    tasks = [
        Task(
            type="content_request",
            objective="Write about AI",
            params={"input_message": "Write a blog post about AI"}
        ),
        Task(
            type="content_request",
            objective="Implement quicksort",
            params={"input_message": "Create a quicksort implementation"}
        )
    ]
    
    # Run tasks
    for task in tasks:
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
