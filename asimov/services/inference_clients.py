import asyncio
from pydantic import Field
import json
from typing import Awaitable, Callable, List, Dict, Any, Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod

import aioboto3
import httpx
import opentelemetry.instrumentation.httpx
import opentelemetry.trace
import pydantic_core
import vertexai.generative_models
import google.api_core.exceptions

from asimov.asimov_base import AsimovBase

tracer = opentelemetry.trace.get_tracer(__name__)
opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor().instrument()


class ChatRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


class ChatMessage(AsimovBase):
    role: ChatRole
    content: str
    cache_marker: bool = False
    model_families: List[str] = Field(default_factory=list)


class AnthropicRequest(AsimovBase):
    anthropic_version: str
    system: str
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    tool_choice: Dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = 512
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 50


class InferenceClient(ABC):
    model: str

    _input_tokens: int = 0
    _output_tokens: int = 0

    def _account_input_tokens(self, tokens):
        self._input_tokens += tokens
        opentelemetry.trace.get_current_span().set_attribute(
            "inference.usage.input_tokens", tokens
        )

    def _account_output_tokens(self, tokens):
        self._output_tokens += tokens
        opentelemetry.trace.get_current_span().set_attribute(
            "inference.usage.output_tokens", tokens
        )

    @abstractmethod
    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.5, temperature=0.5
    ):
        pass

    @abstractmethod
    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.5, temperature=0.5
    ):
        pass

    @abstractmethod
    async def _tool_chain_stream(
        self,
        serialized_messages: List[Dict[str, Any]],
        tools: List[Tuple[Callable, Dict[str, Any]]],
        system: Optional[str] = None,
        max_tokens=8192,
        top_p=0.5,
        temperature=0.5,
        tool_choice="any",
        middlewares: List[Callable[[dict[str, Any]], Awaitable[None]]] = [],
    ) -> List[Dict[str, Any]]:
        pass

    @tracer.start_as_current_span(name="InferenceClient.tool_chain")
    async def tool_chain(
        self,
        messages: List[ChatMessage],
        tools: List[Tuple[Callable, Dict[str, Any]]],
        max_tokens=8192,
        top_p=0.5,
        temperature=0.5,
        max_iterations=10,
        tool_choice="any",
        middlewares: List[Callable[[dict[str, Any]], Awaitable[None]]] = [],
    ):
        tools[-1][1]["cache_control"] = {"type": "ephemeral"}

        tool_funcs = {tool[1]["name"]: tool[0] for tool in tools}

        system = None
        if messages[0].role == ChatRole.SYSTEM:
            system = messages[0].content
            messages = messages[1:]

        serialized_messages: list[dict[str, Any]] = [
            {
                "role": msg.role.value,
                "content": [{"type": "text", "text": msg.content}],
            }
            for msg in messages
        ]

        for _ in range(max_iterations):
            resp = await self._tool_chain_stream(
                serialized_messages,
                tools,
                system=system,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                tool_choice=tool_choice,
                middlewares=middlewares,
            )

            serialized_messages.append(
                {
                    "role": "assistant",
                    "content": resp,
                }
            )

            call = next((c for c in resp if c["type"] == "tool_use"), None)

            if not call:
                print("no call, returning", resp)
                return serialized_messages

            try:
                result = await tool_funcs[call["name"]](call["input"])
            except StopAsyncIteration:
                return serialized_messages

            serialized_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": result,
                        }
                    ],
                }
            )

            if len(serialized_messages) > 5:
                serialized_messages[-5]["content"][0].pop("cache_control", None)
                serialized_messages[-1]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }

            # Artificially slow things down just a bit because perf with caching makes it easy to hit TPM limits
            await asyncio.sleep(2)

        return serialized_messages


class BedrockInferenceClient(InferenceClient):
    def __init__(self, model: str, region_name="us-east-1"):
        self.model = model
        self.region_name = region_name
        self.session = aioboto3.Session()
        self.anthropic_version = "bedrock-2023-05-31"

    @tracer.start_as_current_span(name="BedrockInferenceClient.get_generation")
    async def get_generation(
        self,
        messages: List[ChatMessage],
        max_tokens=8192,
        top_p=0.5,
        temperature=0.5,
    ):

        system = ""
        if messages[0].role == ChatRole.SYSTEM:
            system = messages[0].content
            messages = messages[1:]

        request = AnthropicRequest(
            anthropic_version=self.anthropic_version,
            system=system,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": msg.role.value,
                    "content": [{"type": "text", "text": msg.content}],
                }
                for msg in messages
            ],
        )

        async with self.session.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
        ) as client:
            response = await client.invoke_model(
                body=request.model_dump_json(exclude={"tools", "tool_choice"}),
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
            )

            body: dict = json.loads(await response["body"].read())

            self._account_input_tokens(body["usage"]["input_tokens"])
            self._account_output_tokens(body["usage"]["output_tokens"])

            return body["content"][0]["text"]

    @tracer.start_as_current_span(name="BedrockInferenceClient.tool_chain")
    async def _tool_chain_stream(
        self,
        serialized_messages: List[Dict[str, Any]],
        tools: List[Tuple[Callable, Dict[str, Any]]],
        system: Optional[str] = None,
        max_tokens=8192,
        top_p=0.5,
        temperature=0.5,
        tool_choice="any",
        middlewares: List[Callable[[dict[str, Any]], Awaitable[None]]] = [],
    ):
        for msg in serialized_messages:
            msg["content"][0].pop("cache_control", None)
        for _, tool in tools:
            tool.pop("cache_control", None)

        async with self.session.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
        ) as client:
            request = {
                "anthropic_version": self.anthropic_version,
                "top_p": top_p,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": serialized_messages,
                "tools": [x[1] for x in tools],
                "tool_choice": {"type": tool_choice},
            }
            if system:
                request["system"] = system

            response = await client.invoke_model_with_response_stream(
                body=json.dumps(request),
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
            )

            current_content = []
            current_block: dict[str, Any] = {
                "type": None,
            }
            current_json = ""

            async for chunk in response["body"]:
                chunk_json = json.loads(chunk["chunk"]["bytes"].decode())
                chunk_type = chunk_json["type"]

                if chunk_type == "message_start":
                    self._account_input_tokens(
                        chunk_json["message"]["usage"]["input_tokens"]
                    )
                elif chunk_type == "content_block_start":
                    block_type = chunk_json["content_block"]["type"]
                    if block_type == "text":
                        current_block = {
                            "type": "text",
                            "text": "",
                        }
                    elif block_type == "tool_use":
                        current_block["tool_use"] = {
                            "type": "tool_use",
                            "id": chunk_json["content_block"]["id"],
                            "name": chunk_json["content_block"]["name"],
                            "input": {},
                        }
                        current_json = ""
                elif chunk_type == "content_block_delta":
                    if chunk_json["delta"]["type"] == "text_delta":
                        current_block["text"] += chunk_json["delta"]["text"]
                        for middleware in middlewares:
                            await middleware(current_block)
                    elif chunk_json["delta"]["type"] == "input_json_delta":
                        current_json += chunk_json["delta"]["partial_json"]
                        try:
                            current_block["input"] = pydantic_core.from_json(
                                current_json, allow_partial=True
                            )
                            for middleware in middlewares:
                                await middleware(current_block)
                        except ValueError:
                            pass
                elif chunk_type == "content_block_stop":
                    current_content.append(current_block)

                    current_block = {
                        "type": None,
                    }
                elif chunk_type == "message_delta":
                    if chunk_json["delta"].get("stop_reason") == "tool_use":
                        self._account_output_tokens(
                            chunk_json["usage"]["output_tokens"]
                        )
                        break

            return current_content

        raise Exception("Max retries exceeded")

    @tracer.start_as_current_span(name="BedrockInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self,
        messages: List[ChatMessage],
        max_tokens=8192,
        top_p=0.5,
        temperature=0.5,
    ):
        system = ""
        if messages[0].role == ChatRole.SYSTEM:
            system = messages[0].content
            messages = messages[1:]

        request = AnthropicRequest(
            anthropic_version=self.anthropic_version,
            system=system,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": msg.role.value,
                    "content": [{"type": "text", "text": msg.content}],
                }
                for msg in messages
            ],
        )

        async with self.session.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
        ) as client:
            response = await client.invoke_model_with_response_stream(
                body=request.model_dump_json(exclude={"tools", "tool_choice"}),
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
            )

            async for chunk in response["body"]:
                chunk_json = json.loads(chunk["chunk"]["bytes"].decode())
                chunk_type = chunk_json["type"]

                if chunk_type == "content_block_delta":
                    content_type = chunk_json["delta"]["type"]
                    text = (
                        chunk_json["delta"]["text"]
                        if content_type == "text_delta"
                        else ""
                    )
                    yield text
                elif chunk_type == "message_start":
                    self._account_input_tokens(
                        chunk_json["message"]["usage"]["input_tokens"]
                    )
                elif chunk_type == "message_delta":
                    self._account_output_tokens(chunk_json["usage"]["output_tokens"])


class AnthropicInferenceClient(InferenceClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        api_url: str = "https://api.anthropic.com/v1/messages",
    ):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key

    @tracer.start_as_current_span(name="AnthropicInferenceClient.get_generation")
    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=8192, top_p=0.5, temperature=0.5
    ):
        system = None
        if messages[0].role == ChatRole.SYSTEM:
            system = {
                "system": [
                    {"type": "text", "text": messages[0].content}
                    | (
                        {"cache_control": {"type": "ephemeral"}}
                        if messages[0].cache_marker
                        else {}
                    )
                ]
            }
            messages = messages[1:]

        request = {
            "model": self.model,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": msg.role.value, "content": msg.content}
                | ({"cache_control": {"type": "ephemeral"}} if msg.cache_marker else {})
                for msg in messages
            ],
            "stream": False,
        }

        if system:
            request.update(system)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                timeout=300000,
                json=request,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "prompt-caching-2024-07-31",
                },
            )

            if response.status_code != 200:
                print(await response.aread())
                response.raise_for_status()

            body: dict = response.json()

            self._account_input_tokens(
                # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
                body["usage"]["input_tokens"]
                + body["usage"].get("cache_read_input_tokens", 0)
                + body["usage"].get("cache_creation_input_tokens", 0),
            )
            opentelemetry.trace.get_current_span().set_attribute(
                "inference.usage.cached_input_tokens",
                body["usage"].get("cache_read_input_tokens", 0),
            )
            self._account_output_tokens(body["usage"]["output_tokens"])

            return body["content"][0]["text"]

    @tracer.start_as_current_span(name="AnthropicInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=8192, top_p=0.5, temperature=0.5
    ):

        system = None
        if messages[0].role == ChatRole.SYSTEM:
            system = {
                "system": [
                    {"type": "text", "text": messages[0].content}
                    | (
                        {"cache_control": {"type": "ephemeral"}}
                        if messages[0].cache_marker
                        else {}
                    )
                ]
            }
            messages = messages[1:]

        request = {
            "model": self.model,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": [
                        {"type": "text", "text": msg.content}
                        | (
                            {"cache_control": {"type": "ephemeral"}}
                            if msg.cache_marker
                            else {}
                        )
                    ],
                }
                for msg in messages
            ],
            "stream": True,
        }

        if system:
            request.update(system)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.api_url,
                timeout=300000,
                json=request,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "prompt-caching-2024-07-31",
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_json = json.loads(line[6:])
                        chunk_type = chunk_json["type"]

                        if chunk_type == "content_block_delta":
                            content_type = chunk_json["delta"]["type"]
                            text = (
                                chunk_json["delta"]["text"]
                                if content_type == "text_delta"
                                else ""
                            )
                            yield text
                        elif chunk_type == "message_start":
                            self._account_input_tokens(
                                chunk_json["message"]["usage"]["input_tokens"]
                                + chunk_json["message"]["usage"].get(
                                    "cache_read_input_tokens", 0
                                )
                                + chunk_json["message"]["usage"].get(
                                    "cache_creation_input_tokens", 0
                                ),
                            )
                            opentelemetry.trace.get_current_span().set_attribute(
                                "inference.usage.cached_input_tokens",
                                chunk_json["message"]["usage"].get(
                                    "cache_read_input_tokens", 0
                                ),
                            )
                        elif chunk_type == "message_delta":
                            self._account_output_tokens(
                                chunk_json["usage"]["output_tokens"],
                            )

    async def _tool_chain_stream(
        self,
        serialized_messages,
        tools,
        system=None,
        max_tokens=8192,
        top_p=0.5,
        temperature=0.5,
        tool_choice="any",
        middlewares=[],
    ):
        request = {
            "model": self.model,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": serialized_messages,
            "stream": True,
            "tools": [x[1] for x in tools],
            "tool_choice": {"type": tool_choice},
        }
        if system:
            request["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        current_content = []
        current_block: dict[str, Any] = {
            "type": None,
        }
        current_json = ""

        async with httpx.AsyncClient() as client:
            for retry in range(5):
                try:
                    async with client.stream(
                        "POST",
                        self.api_url,
                        timeout=300000,
                        json=request,
                        headers={
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                            "anthropic-beta": "prompt-caching-2024-07-31",
                        },
                    ) as response:
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue

                            chunk_json = json.loads(line[6:])
                            chunk_type = chunk_json["type"]

                            if chunk_type == "message_start":
                                self._account_input_tokens(
                                    chunk_json["message"]["usage"]["input_tokens"]
                                    + chunk_json["message"]["usage"].get(
                                        "cache_read_input_tokens", 0
                                    )
                                    + chunk_json["message"]["usage"].get(
                                        "cache_creation_input_tokens", 0
                                    )
                                )
                                opentelemetry.trace.get_current_span().set_attribute(
                                    "inference.usage.cached_input_tokens",
                                    chunk_json["message"]["usage"].get(
                                        "cache_read_input_tokens", 0
                                    ),
                                )
                            elif chunk_type == "content_block_start":
                                block_type = chunk_json["content_block"]["type"]
                                if block_type == "text":
                                    current_block = {
                                        "type": "text",
                                        "text": "",
                                    }
                                elif block_type == "tool_use":
                                    current_block = {
                                        "type": "tool_use",
                                        "id": chunk_json["content_block"]["id"],
                                        "name": chunk_json["content_block"]["name"],
                                        "input": {},
                                    }
                                    current_json = ""
                            elif chunk_type == "content_block_delta":
                                if chunk_json["delta"]["type"] == "text_delta":
                                    current_block["text"] += chunk_json["delta"]["text"]
                                    for middleware in middlewares:
                                        await middleware(current_block)
                                elif chunk_json["delta"]["type"] == "input_json_delta":
                                    current_json += chunk_json["delta"]["partial_json"]
                                    try:
                                        current_block["input"] = (
                                            pydantic_core.from_json(
                                                current_json, allow_partial=True
                                            )
                                        )
                                        for middleware in middlewares:
                                            await middleware(current_block)
                                    except ValueError:
                                        pass
                            elif chunk_type == "content_block_stop":
                                current_content.append(current_block)

                                current_block = {
                                    "type": None,
                                }
                            elif chunk_type == "message_delta":
                                if chunk_json["delta"].get("stop_reason") == "tool_use":
                                    self._account_output_tokens(
                                        chunk_json["usage"]["output_tokens"]
                                    )
                                    break
                    return current_content
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        print("429 backoff")
                        await asyncio.sleep(3**retry)
                        continue
                    raise
                except (httpx.RequestError, httpx.HTTPError) as e:
                    if retry < 4:  # Allow retrying on connection errors
                        await asyncio.sleep(3**retry)
                        continue
                    raise


class OAIRequest(AsimovBase):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stream_options: Dict[str, Any] = {}


class OAIInferenceClient(InferenceClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        api_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key

    @tracer.start_as_current_span(name="OAIInferenceClient.get_generation")
    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=1.0, temperature=0.5
    ):
        if self.model in ["o1-preview", "o1-mini"]:
            request = {
                "model": self.model,
                "messages": messages,
            }
        else:
            request = OAIRequest(
                model=self.model,
                messages=[
                    {
                        "role": m.role.value,
                        "content": m.content,
                    }
                    for m in messages
                ],
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                stream=False,
            ).model_dump(exclude={"stream_options"})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                json=request,
                timeout=180,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

            response.raise_for_status()

            body: dict = response.json()

            self._account_input_tokens(body["usage"]["prompt_tokens"])
            opentelemetry.trace.get_current_span().set_attribute(
                "inference.usage.cached_input_tokens",
                body["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0),
            )
            self._account_output_tokens(body["usage"]["completion_tokens"])

            return body["choices"][0]["message"]["content"]

    @tracer.start_as_current_span(name="OAIInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=1.0, temperature=0.5
    ):
        request = OAIRequest(
            model=self.model,
            messages=[
                {
                    "role": msg.role.value,
                    "content": msg.content,
                }
                for msg in messages
            ],
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.api_url,
                json=request.__dict__,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=180,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break
                        data = json.loads(line[6:])
                        if data["choices"] and data["choices"][0]["delta"].get(
                            "content"
                        ):
                            yield data["choices"][0]["delta"]["content"]
                        elif data.get("usage"):
                            self._account_input_tokens(data["usage"]["prompt_tokens"])
                            # No reference to cached tokens in the docs for the streaming API response objects...
                            opentelemetry.trace.get_current_span().set_attribute(
                                "inference.usage.cached_input_tokens",
                                data["usage"]
                                .get("prompt_tokens_details", {})
                                .get("cached_tokens", 0),
                            )
                            self._account_output_tokens(
                                data["usage"]["completion_tokens"]
                            )


class OpenRouterInferenceClient(OAIInferenceClient):
    def __init__(self, model: str, api_key: str):
        super().__init__(
            model,
            api_key,
            api_url="https://openrouter.ai/api/v1/chat/completions",
        )

    async def _tool_chain_stream(
        self,
        serialized_messages,
        tools,
        system=None,
        max_tokens=8192,
        top_p=0.5,
        temperature=0.5,
        tool_choice="any",
        middlewares=[],
    ):
        if system:
            serialized_messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system}],
                }
            ] + serialized_messages

        openrouter_messages = []
        for message in serialized_messages:
            if (
                message["role"] == "user"
                and isinstance(message["content"], list)
                and message["content"][0]["type"] == "tool_result"
            ):
                openrouter_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["content"][0]["tool_use_id"],
                        "content": message["content"][0]["content"],
                    }
                )
            else:
                tc = next(
                    (
                        content
                        for content in message["content"]
                        if content["type"] == "tool_use"
                    ),
                    None,
                )
                if tc:
                    # OpenRouter hangs if there is message content here?
                    message = {
                        "role": message["role"],
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["input"]),
                                },
                            }
                        ],
                    }
                openrouter_messages.append(message)

        openrouter_tools = []
        for _, tool in tools:
            openrouter_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description"),
                        "parameters": tool["input_schema"],
                    },
                }
            )

        request = {
            "model": self.model,
            "messages": openrouter_messages,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "stream": True,
            "tool_choice": tool_choice,
            "tools": openrouter_tools,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.api_url,
                json=request,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=180,
            ) as response:
                response.raise_for_status()

                text_block = None
                tool_call_blocks = []

                async for line in response.aiter_lines():
                    if line == "data: [DONE]":
                        return ([text_block] if text_block else []) + tool_call_blocks
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data["choices"]:
                            d = data["choices"][0]["delta"]
                            if d.get("content"):
                                if not text_block:
                                    text_block = {
                                        "type": "text",
                                        "text": d["content"],
                                    }
                                else:
                                    text_block["text"] += d["content"]
                            for tc in d.get("tool_calls", []):
                                if "id" in tc:
                                    tool_call_blocks.append(
                                        {
                                            "type": "tool_use",
                                            "id": tc["id"],
                                            "name": tc["function"]["name"],
                                            "raw_input": "",
                                            "input": {},
                                        }
                                    )
                                tool_call_blocks[tc["index"]]["raw_input"] += tc[
                                    "function"
                                ]["arguments"]
                                try:
                                    tool_call_blocks[tc["index"]]["input"] = (
                                        pydantic_core.from_json(
                                            tool_call_blocks[tc["index"]]["raw_input"],
                                            allow_partial=True,
                                        )
                                    )
                                    for middleware in middlewares:
                                        await middleware(tool_call_blocks[tc["index"]])
                                except ValueError:
                                    pass
                        if data.get("usage"):
                            self._account_input_tokens(data["usage"]["prompt_tokens"])
                            self._account_output_tokens(
                                data["usage"]["completion_tokens"]
                            )


class VertexInferenceClient(InferenceClient):
    def __init__(self, model: str):
        self.model = model

    @tracer.start_as_current_span(name="VertexInferenceClient.get_generation")
    async def get_generation(
        self,
        messages: List[ChatMessage],
        max_tokens=4096,
        top_p=0.5,
        temperature=0.5,
        schema=None,
    ):
        if messages[0].role == ChatRole.SYSTEM:
            model = vertexai.generative_models.GenerativeModel(
                self.model, system_instruction=messages[0].content
            )
            messages = messages[1:]
        else:
            model = vertexai.generative_models.GenerativeModel(self.model)

        prefill = ""
        if schema and messages[-1].role == ChatRole.ASSISTANT:
            prefill = messages[-1].content
            messages = messages[:-1]

        for retry in range(4):
            try:
                resp = await model.generate_content_async(
                    [
                        vertexai.generative_models.Content(
                            role=msg.role.value,
                            parts=[
                                vertexai.generative_models.Part.from_text(msg.content)
                            ],
                        )
                        for msg in messages
                    ],
                    generation_config=vertexai.generative_models.GenerationConfig(
                        max_output_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature,
                        response_mime_type="application/json" if schema else None,
                        response_schema=schema,
                    ),
                )
                if schema:
                    resp.text = resp.text[len(prefill) :]
                break
            except google.api_core.exceptions.ResourceExhausted:
                print("backoff retry")
                await asyncio.sleep(2**retry)
        else:
            raise

        self._account_input_tokens(resp.usage_metadata.prompt_token_count)
        self._account_output_tokens(resp.usage_metadata.candidates_token_count)

        return resp.text

    @tracer.start_as_current_span(name="VertexInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.5, temperature=0.5
    ):
        if messages[0].role == ChatRole.SYSTEM:
            model = vertexai.generative_models.GenerativeModel(
                self.model, system_instruction=messages[0].content
            )
            messages = messages[1:]
        else:
            model = vertexai.generative_models.GenerativeModel(self.model)

        async for chunk in model.generate_content_async(
            [
                vertexai.generative_models.Content(
                    role=msg.role.value,
                    parts=[vertexai.generative_models.Part.from_text(msg.content)],
                )
                for msg in messages
            ],
            generation_config=vertexai.generative_models.GenerationConfig(
                max_output_tokens=max_tokens, top_p=top_p, temperature=temperature
            ),
            stream=True,
        ):
            yield chunk.text
            if chunk.usage_metadata:
                self._account_input_tokens(chunk.usage_metadata.prompt_token_count)
                self._account_output_tokens(chunk.usage_metadata.candidates_token_count)
