"""
Agent-Directed Flow Example

This example demonstrates how to use the AgentDirectedFlow module to create an intelligent
routing system for content creation tasks. The system will:
1. Analyze incoming requests using LLM
2. Route tasks to appropriate specialized modules
3. Handle multiple types of content creation tasks
"""

import os
import asyncio
from typing import Dict, Any
from uuid import uuid4

from asimov.graph import (
    Agent,
    AgentModule,
    ModuleType,
    Node,
    NodeConfig,
    Cache,
)
from asimov.graph.agent_directed_flow import (
    AgentDirectedFlowControl,
    AgentDrivenFlowControlConfig,
    AgentDrivenFlowDecision,
    Example,
)
from asimov.graph.tasks import Task
from asimov.caches.redis_cache import RedisCache
from asimov.services.inference_clients import (
    AnthropicInferenceClient,
    ChatMessage,
    ChatRole,
)


class BlogWriterModule(AgentModule):
    """Specialized module for writing blog posts."""

    name: str = "blog_writer"
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
        message = await cache.get("input_message")

        prompt = f"""
        Write a blog post based on the following request:
        {message}
        
        Format the post with:
        1. A compelling title
        2. An introduction
        3. Main content sections
        4. A conclusion
        """

        response = await self.client.get_generation(
            [ChatMessage(role=ChatRole.USER, content=prompt)]
        )

        return {"status": "success", "result": response}


class CodeWriterModule(AgentModule):
    """Specialized module for writing code examples."""

    name: str = "code_writer"
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
        message = await cache.get("input_message")

        prompt = f"""
        Create code based on the following request:
        {message}
        
        Provide:
        1. The complete code implementation
        2. Comments explaining key parts
        3. Usage examples
        4. Any necessary setup instructions
        """

        response = await self.client.get_generation(
            [ChatMessage(role=ChatRole.USER, content=prompt)]
        )

        return {"status": "success", "result": response}


class ErrorHandlerModule(AgentModule):
    """Handles cases where no other module is appropriate."""

    name: str = "error_handler"
    type: ModuleType = ModuleType.EXECUTOR

    async def process(
        self, cache: Cache, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        message = await cache.get("input_message")

        return {
            "status": "error",
            "result": f"Unable to process request: {message}. Please try a different type of request.",
        }


async def main():
    # Create the agent
    cache = RedisCache(default_prefix=str(uuid4()))
    agent = Agent(
        cache=cache,
        max_concurrent_tasks=1,
        max_total_iterations=20,
    )

    # Set up the inference client for flow control
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
    inference_client = AnthropicInferenceClient(
        model="claude-3-5-sonnet-20241022", api_key=api_key
    )

    # Create specialized content nodes
    blog_writer = Node(
        name="blog_writer",
        type=ModuleType.EXECUTOR,
        modules=[BlogWriterModule()],
        dependencies=["flow_control"],
    )

    code_writer = Node(
        name="code_writer",
        type=ModuleType.EXECUTOR,
        modules=[CodeWriterModule()],
        dependencies=["flow_control"],
    )

    error_handler = Node(
        name="error_handler",
        type=ModuleType.EXECUTOR,
        modules=[ErrorHandlerModule()],
        dependencies=["flow_control"],
    )

    # Create flow control node with agent-directed decisions
    flow_control = Node(
        name="flow_control",
        type=ModuleType.FLOW_CONTROL,
        modules=[
            AgentDirectedFlowControl(
                name="ContentFlowControl",
                type=ModuleType.FLOW_CONTROL,
                voters=3,
                inference_client=inference_client,
                system_description="A system that handles various content creation tasks",
                flow_config=AgentDrivenFlowControlConfig(
                    decisions=[
                        AgentDrivenFlowDecision(
                            next_node="blog_writer",
                            metadata={
                                "description": "Writes blog posts on technical topics"
                            },
                            examples=[
                                Example(
                                    message="Write a blog post about AI agents",
                                    choices=[
                                        {
                                            "choice": "blog_writer",
                                            "description": "Writes blog posts",
                                        },
                                        {
                                            "choice": "code_writer",
                                            "description": "Writes code",
                                        },
                                    ],
                                    choice="blog_writer",
                                    reasoning="The request is specifically for blog content",
                                ),
                                Example(
                                    message="Create an article explaining machine learning basics",
                                    choices=[
                                        {
                                            "choice": "blog_writer",
                                            "description": "Writes blog posts",
                                        },
                                        {
                                            "choice": "code_writer",
                                            "description": "Writes code",
                                        },
                                    ],
                                    choice="blog_writer",
                                    reasoning="The request is for an explanatory article",
                                ),
                            ],
                        ),
                        AgentDrivenFlowDecision(
                            next_node="code_writer",
                            metadata={
                                "description": "Writes code examples and tutorials"
                            },
                            examples=[
                                Example(
                                    message="Create a Python script for data processing",
                                    choices=[
                                        {
                                            "choice": "blog_writer",
                                            "description": "Writes blog posts",
                                        },
                                        {
                                            "choice": "code_writer",
                                            "description": "Writes code",
                                        },
                                    ],
                                    choice="code_writer",
                                    reasoning="The request is for code creation",
                                ),
                                Example(
                                    message="Show me how to implement a binary search tree",
                                    choices=[
                                        {
                                            "choice": "blog_writer",
                                            "description": "Writes blog posts",
                                        },
                                        {
                                            "choice": "code_writer",
                                            "description": "Writes code",
                                        },
                                    ],
                                    choice="code_writer",
                                    reasoning="The request is for a code implementation",
                                ),
                            ],
                        ),
                    ],
                    default="error_handler",
                ),
            )
        ],
    )

    # Add all nodes to the agent
    agent.add_multiple_nodes([blog_writer, code_writer, error_handler, flow_control])

    # Example tasks to demonstrate routing
    tasks = [
        Task(
            type="content_request",
            objective="Write a blog post about the future of AI",
            params={"input_message": "Write a blog post about the future of AI"},
        ),
        Task(
            type="content_request",
            objective="Create a Python implementation of quicksort",
            params={"input_message": "Create a Python implementation of quicksort"},
        ),
        Task(
            type="content_request",
            objective="Generate a haiku about programming",
            params={"input_message": "Generate a haiku about programming"},
        ),
    ]

    # Run each task and show results
    for task in tasks:
        await cache.set("input_message", task.params["input_message"])
        print(f"\nProcessing task: {task.objective}")
        result = await agent.run_task(task)
        print(f"Result: {result}")
        print("Node results:")
        for node, node_result in agent.node_results.items():
            print(f"{node}: {node_result}")


if __name__ == "__main__":
    asyncio.run(main())
