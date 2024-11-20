import asyncio
from pydantic import Field
import json
from typing import Callable, List, Dict, Any, Tuple
from enum import Enum
from abc import ABC, abstractmethod

import aioboto3
import httpx
import opentelemetry.instrumentation.httpx
import opentelemetry.trace
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
    async def tool_chain(
        messages: List[ChatMessage],
        tools: List[Tuple[Callable, Dict[str, Any]]],
        max_tokens=8192,
        top_p=0.5,
        temperature=0.5,
        max_iterations=10,
        tool_choice="any",
    ):
        pass


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
    async def tool_chain(
        self,
        messages: List[ChatMessage],
        tools: List[Tuple[Callable, Dict[str, Any]]],
        max_tokens=8192,
        top_p=0.5,
        temperature=0.5,
        max_iterations=10,
        tool_choice="any",
        keep_n_states=None,
    ):
        tool_funcs = {tool[1]["name"]: tool[0] for tool in tools}

        system = ""
        if messages[0].role == ChatRole.SYSTEM:
            system = messages[0].content
            messages = messages[1:]

        serialized_messages = [
            {
                "role": msg.role.value,
                "content": [{"type": "text", "text": msg.content}],
            }
            for msg in messages
        ]

        async with self.session.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
        ) as client:
            for _ in range(max_iterations):
                request = AnthropicRequest(
                    anthropic_version=self.anthropic_version,
                    system=system,
                    top_p=top_p,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=serialized_messages,
                    tools=[x[1] for x in tools],
                    tool_choice={"type": tool_choice},
                )

                response = await client.invoke_model(
                    body=request.model_dump_json(),
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                )

                body: dict = json.loads(await response["body"].read())

                self._account_input_tokens(body["usage"]["input_tokens"])
                self._account_output_tokens(body["usage"]["output_tokens"])

                serialized_messages.append(
                    {
                        "role": "assistant",
                        "content": body["content"],
                    }
                )

                call = next(
                    (c for c in body["content"] if c["type"] == "tool_use"), None
                )
                if not call:
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

                if keep_n_states:
                    serialized_messages = [
                        serialized_messages[0]
                    ] + serialized_messages[1:][-keep_n_states:]
        return serialized_messages

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

    @tracer.start_as_current_span(name="AnthropicInferenceClient.tool_chain")
    async def tool_chain(
        self,
        messages: List[ChatMessage],
        tools: List[Tuple[Callable, Dict[str, Any]]],
        max_tokens=8192,
        top_p=0.5,
        temperature=0.5,
        max_iterations=10,
        tool_choice="any",
        keep_n_states=None,
    ):
        tool_funcs = {tool[1]["name"]: tool[0] for tool in tools}

        system = None
        if messages[0].role == ChatRole.SYSTEM:
            system = {
                "system": [
                    {
                        "type": "text",
                        "text": messages[0].content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            }
            messages = messages[1:]

        serialized_messages = [
            {
                "role": msg.role.value,
                "content": [{"type": "text", "text": msg.content}],
            }
            for msg in messages
        ]

        async with httpx.AsyncClient() as client:
            for _ in range(max_iterations):
                request = {
                    "model": self.model,
                    "top_p": top_p,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": serialized_messages,
                    "stream": False,
                    "tools": [x[1] for x in tools],
                    "tool_choice": {"type": tool_choice},
                }
                if system:
                    request.update(system)

                for retry in range(5):
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

                    if response.status_code == 429:
                        print("429 backoff")
                        await asyncio.sleep(3**retry)
                        continue

                    break

                response.raise_for_status()

                body: dict = response.json()

                self._account_input_tokens(
                    # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
                    body["usage"]["input_tokens"]
                    + body["usage"].get("cache_read_input_tokens", 0)
                    + body["usage"].get("cache_creation_input_tokens", 0),
                )
                self._account_output_tokens(body["usage"]["output_tokens"])

                serialized_messages.append(
                    {
                        "role": "assistant",
                        "content": body["content"],
                    }
                )

                call = next(
                    (c for c in body["content"] if c["type"] == "tool_use"), None
                )
                if not call:
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

                if keep_n_states:
                    serialized_messages = [
                        serialized_messages[0]
                    ] + serialized_messages[1:][-keep_n_states:]

                # Artificially slow things down just a bit because perf with caching makes it easy to hit TPM limits
                await asyncio.sleep(2)

        return serialized_messages


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
                messages=[m.model_dump(exclude={"cache_marker"}) for m in messages],
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
            messages=[msg.model_dump(exclude={"cache_marker"}) for msg in messages],
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
