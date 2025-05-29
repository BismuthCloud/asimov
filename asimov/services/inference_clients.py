import asyncio
from collections.abc import Hashable
import botocore.exceptions
from pydantic import Field
import json
from typing import Awaitable, Callable, List, Dict, Any, Optional, Tuple, AsyncGenerator
from enum import Enum
from abc import ABC, abstractmethod
import logging

import aioboto3
import httpx
import backoff
import opentelemetry.instrumentation.httpx
import opentelemetry.trace
import pydantic_core
import vertexai.generative_models
import google.api_core.exceptions
from google import genai
from google.genai import types
import google.auth

from asimov.asimov_base import AsimovBase
from asimov.utils.token_counter import approx_tokens_from_serialized_messages
from asimov.graph import NonRetryableException

logger = logging.getLogger(__name__)
tracer = opentelemetry.trace.get_tracer(__name__)
opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor().instrument()


class InferenceException(Exception):
    """
    A generic exception for inference errors.
    Should be safe to retry.
    ValueError is raised if the request is un-retryable (e.g. parameters are malformed).
    """

    pass


class ContextLengthExceeded(InferenceException, ValueError):
    pass


class RetriesExceeded(InferenceException):
    """
    Raised when the maximum number of retries is exceeded.
    """

    pass


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


class InferenceCost(AsimovBase):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    # Dollar amount that should be added on to the cost of the request
    dollar_adjust: float = 0.0


class InferenceClient(ABC):
    model: str
    _trace_id: int = 0
    trace_cb: Optional[
        Callable[
            [int, list[dict[str, Any]], list[dict[str, Any]], InferenceCost],
            Awaitable[None],
        ]
    ] = None
    _cost: InferenceCost
    _last_cost: InferenceCost

    def __init__(self):
        self._cost = InferenceCost()
        self._last_cost = InferenceCost()

    async def _trace(self, request, response):
        opentelemetry.trace.get_current_span().set_attribute(
            "inference.usage.input_tokens", self._cost.input_tokens
        )
        opentelemetry.trace.get_current_span().set_attribute(
            "inference.usage.cached_input_tokens",
            self._cost.cache_read_input_tokens,
        )
        opentelemetry.trace.get_current_span().set_attribute(
            "inference.usage.output_tokens", self._cost.output_tokens
        )

        if self.trace_cb:
            logger.debug(f"Request {self._trace_id} cost {self._cost}")
            await self.trace_cb(self._trace_id, request, response, self._cost)
            self._last_cost = self._cost
            self._cost = InferenceCost()
            self._trace_id += 1

    @abstractmethod
    def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.9, temperature=0.5
    ) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.9, temperature=0.5
    ):
        pass

    @abstractmethod
    async def _tool_chain_stream(
        self,
        serialized_messages: List[Dict[str, Any]],
        tools: List[Tuple[Callable, Dict[str, Any]]],
        system: Optional[str] = None,
        max_tokens=1024,
        top_p=0.9,
        temperature=0.5,
        tool_choice="any",
        middlewares: List[Callable[[dict[str, Any]], Awaitable[None]]] = [],
    ) -> List[Dict[str, Any]]:
        pass

    @tracer.start_as_current_span(name="InferenceClient.tool_chain")
    async def tool_chain(
        self,
        messages: List[ChatMessage],
        tools: List[Tuple[Callable, Dict[str, Any]]] = [],
        max_tokens=1024,
        top_p=0.9,
        temperature=0.5,
        max_iterations=10,
        tool_choice="any",
        middlewares: List[Callable[[dict[str, Any]], Awaitable[None]]] = [],
        mode_swap_callback: Optional[
            Callable[
                [],
                Awaitable[tuple[str, List[Tuple[Callable, Dict[str, Any]]], Hashable]],
            ]
        ] = None,
        fifo_ratio: Optional[float] = None,
        tool_parser: Optional[Callable[[str, ...], list[Callable[[any], str]]]] = None,
        tool_result_reducer: Optional[Callable[[list[dict[str, Any]]], str]] = None,
    ):
        mode = None
        if mode_swap_callback:
            _, _, mode = await mode_swap_callback()

        if tool_parser and not tool_result_reducer:
            raise ValueError(
                "If tool_parser is set then tool_result_reducer must be set."
            )

        last_mode_cached_message: dict[Hashable, int] = {}

        if not tool_parser:
            tools[-1][1]["cache_control"] = {"type": "ephemeral"}

        tool_funcs = {tool[1]["name"]: tool[0] for tool in tools}

        system = None
        if messages[0].role == ChatRole.SYSTEM and not tool_parser:
            system = messages[0].content
            messages = messages[1:]

        serialized_messages: list[dict[str, Any]] = [
            {"role": msg.role.value, "content": [{"type": "text", "text": msg.content}]}
            for msg in messages
        ]

        for _ in range(max_iterations):
            for msg in serialized_messages:
                msg["content"][-1].pop("cache_control", None)
            if mode_swap_callback and len(serialized_messages) > 2:
                if not tool_parser:
                    tools[-1][1].pop("cache_control", None)

                prompt, tools, mode = await mode_swap_callback()

                if not tool_parser:
                    tools[-1][1]["cache_control"] = {"type": "ephemeral"}

                tool_funcs = {tool[1]["name"]: tool[0] for tool in tools}

                if prompt:
                    serialized_messages[0]["content"] = [
                        {"type": "text", "text": prompt}
                    ]

            if mode in last_mode_cached_message:
                serialized_messages[last_mode_cached_message[mode]]["content"][-1][
                    "cache_control"
                ] = {"type": "ephemeral"}

            serialized_messages[-1]["content"][-1]["cache_control"] = {
                "type": "ephemeral"
            }

            last_mode_cached_message[mode] = len(serialized_messages) - 1

            if (
                fifo_ratio
                and (
                    (
                        self._last_cost.cache_read_input_tokens
                        + self._last_cost.input_tokens
                    )
                    / 200000
                )
                > fifo_ratio
            ):
                logger.info(f"cache threshold hit, tossing early messages and retrying")
                # If we hit context length, remove a handful of assistant,user message pairs from the middle
                # A handful so that we can hopefully get at least a couple cache hits with this
                # truncated history before having to drop messages again.

                # We want the earliest thing we remove to be an assistant message (requesting the next tool call), which have odd indices
                start_remove = int(len(serialized_messages) / 3)
                if start_remove % 2 != 1:
                    start_remove += 1
                # And the last thing we remove should be a user message (with tool response), which have even indices
                end_remove = int(len(serialized_messages) * 2 / 3)
                if end_remove % 2 != 0:
                    end_remove -= 1
                logger.debug(
                    f"Removing messages {start_remove} through {end_remove} from serialized messages"
                )
                end_remove += 1  # inclusive
                serialized_messages = (
                    serialized_messages[:start_remove]
                    + serialized_messages[end_remove:]
                )
                for mode in last_mode_cached_message.keys():
                    # Delete markers if they are in the removed range
                    if start_remove <= last_mode_cached_message[mode] < end_remove:
                        del last_mode_cached_message[mode]
                    # And adjust indices of anything that got "slid" back
                    elif last_mode_cached_message[mode] >= end_remove:
                        last_mode_cached_message[mode] -= end_remove - start_remove

            for retry in range(1, 5):
                try:
                    if not tool_parser:
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

                        if not resp:
                            raise InferenceException("no response blocks returned")

                        calls = [c for c in resp if c["type"] == "tool_use"]

                        await self._trace(serialized_messages, resp)
                    else:
                        buf = ""
                        calls = None

                        # this is a little dumb since we serialize them above, but we do the caching updates so whatever
                        unserialized_messages = [
                            ChatMessage(
                                role=ChatRole(msg["role"]),
                                content=msg["content"][0]["text"],
                                cache_marker=msg["content"][0]
                                .get("cache_control", {})
                                .get("type", None)
                                == "ephemeral",
                            )
                            for msg in serialized_messages
                        ]

                        async for token in self.connect_and_listen(
                            unserialized_messages,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            temperature=temperature,
                        ):
                            buf += token

                            calls = await tool_parser(buf, mode)

                        if type(calls) is not list:
                            calls = [calls]

                        resp = [{"type": "text", "text": buf}]

                    break
                except ContextLengthExceeded as e:
                    logger.info("Non-retryable exception hit (context length), bailing")
                    return serialized_messages
                except NonRetryableException as e:
                    logger.info(f"Non-retryable exception hit ({e}), bailing")
                    raise
                except ValueError as e:
                    logger.info(f"ValueError hit ({e}), bailing")
                    import traceback

                    traceback.print_exc()
                    return serialized_messages
                except InferenceException as e:
                    logger.info("inference exception %s", e)
                    await asyncio.sleep(3**retry)
                    if retry > 3:
                        # Modify messages to try and cache bust in case we have a poison message or similar
                        serialized_messages[0]["content"][0][
                            "text"
                        ] += "\n\nTFJeD9K6smAnr6sUcllj"
                    continue
                except Exception:
                    logger.warning("generic inference exception", exc_info=True)
                    await asyncio.sleep(3**retry)
                    continue
            else:
                logger.info("Retries exceeded, bailing!")
                raise RetriesExceeded()

            serialized_messages.append(
                {
                    "role": "assistant",
                    "content": resp,
                }
            )

            if not calls:
                serialized_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "You did not correctly make a tool call. Please make a tool call from your available tools.",
                            }
                        ],
                    }
                )
                continue

            content_blocks = []
            for call in calls:
                try:
                    func = tool_funcs.get(call["name"])

                    if not func:
                        result = "This tool is not available please select and use an available tool."
                    else:
                        result = await func(call["input"])
                except StopAsyncIteration:
                    return serialized_messages

                if tool_parser:
                    content_blocks.append(
                        {
                            "type": "text",
                            "content": str(result),
                        }
                    )
                else:
                    content_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": str(result),
                        }
                    )

            if tool_parser and tool_result_reducer:
                content_blocks = [
                    {"type": "text", "text": tool_result_reducer(calls, content_blocks)}
                ]

            serialized_messages.append(
                {
                    "role": "user",
                    "content": content_blocks,
                }
            )
        return serialized_messages


class BedrockInferenceClient(InferenceClient):
    def __init__(self, model: str, region_name="us-east-1"):
        super().__init__()
        self.model = model
        self.region_name = region_name
        self.session = aioboto3.Session()
        self.anthropic_version = "bedrock-2023-05-31"

    @tracer.start_as_current_span(name="BedrockInferenceClient.get_generation")
    @backoff.on_exception(backoff.expo, InferenceException, max_time=60)
    async def get_generation(
        self,
        messages: List[ChatMessage],
        max_tokens=1024,
        top_p=0.9,
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
            try:
                response = await client.invoke_model(
                    body=request.model_dump_json(exclude={"tools", "tool_choice"}),
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                )
            except client.exceptions.ValidationException as e:
                raise ValueError(str(e))
            except botocore.exceptions.ClientError as e:
                raise InferenceException(str(e))

            body: dict = json.loads(await response["body"].read())

            self._cost.input_tokens += body["usage"]["input_tokens"]
            self._cost.output_tokens += body["usage"]["output_tokens"]

            await self._trace(request.messages, body["content"])

            return body["content"][0]["text"]

    @tracer.start_as_current_span(name="BedrockInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self,
        messages: List[ChatMessage],
        max_tokens=1024,
        top_p=0.9,
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
            try:
                response = await client.invoke_model_with_response_stream(
                    body=request.model_dump_json(exclude={"tools", "tool_choice"}),
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                )
            except client.exceptions.ValidationException as e:
                raise ValueError(str(e))
            except botocore.exceptions.ClientError as e:
                raise InferenceException(str(e))

            out = ""

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
                    out += text
                    yield text
                elif chunk_type == "message_start":
                    self._cost.input_tokens += chunk_json["message"]["usage"][
                        "input_tokens"
                    ]
                elif chunk_type == "message_delta":
                    self._cost.output_tokens += chunk_json["usage"]["output_tokens"]

            await self._trace(request.messages, [{"text": out}])

    @tracer.start_as_current_span(name="BedrockInferenceClient._tool_chain_stream")
    async def _tool_chain_stream(
        self,
        serialized_messages: List[Dict[str, Any]],
        tools: List[Tuple[Callable, Dict[str, Any]]],
        system: Optional[str] = None,
        max_tokens=1024,
        top_p=0.9,
        temperature=0.5,
        tool_choice="any",
        middlewares: List[Callable[[dict[str, Any]], Awaitable[None]]] = [],
    ):
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
                request["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            try:
                response = await client.invoke_model_with_response_stream(
                    body=json.dumps(request),
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                )
            except client.exceptions.ValidationException as e:
                raise ValueError(str(e))
            except botocore.exceptions.ClientError as e:
                raise InferenceException(str(e))

            current_content = []
            current_block: dict[str, Any] = {
                "type": None,
            }
            current_json = ""

            async for chunk in response["body"]:
                chunk_json = json.loads(chunk["chunk"]["bytes"].decode())
                chunk_type = chunk_json["type"]

                if chunk_type == "message_start":
                    self._cost.input_tokens += chunk_json["message"]["usage"][
                        "input_tokens"
                    ]
                    self._cost.cache_read_input_tokens += chunk_json["message"][
                        "usage"
                    ].get("cache_read_input_tokens", 0)
                    self._cost.cache_write_input_tokens += chunk_json["message"][
                        "usage"
                    ].get("cache_creation_input_tokens", 0)
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
                    elif block_type == "thinking":
                        current_block = {
                            "type": "thinking",
                            "thinking": chunk_json["content_block"]["thinking"],
                        }
                    elif block_type == "redacted_thinking":
                        current_block = chunk_json["content_block"]
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
                    elif chunk_json["delta"]["type"] == "thinking_delta":
                        current_block["thinking"] += chunk_json["delta"]["thinking"]
                    elif chunk_json["delta"]["type"] == "signature_delta":
                        current_block["signature"] = chunk_json["delta"]["signature"]
                elif chunk_type == "content_block_stop":
                    current_content.append(current_block)

                    current_block = {
                        "type": None,
                    }
                elif chunk_type == "message_delta":
                    if chunk_json["delta"].get("stop_reason") == "tool_use":
                        self._cost.output_tokens += chunk_json["usage"]["output_tokens"]
                        break
                elif chunk_type == "error":
                    if chunk_json["error"]["type"] == "invalid_request_error":
                        raise ValueError(chunk_json["error"]["type"])
                    raise InferenceException(chunk_json["error"]["type"])
                elif chunk_type == "ping":
                    pass

            return current_content

        raise Exception("Max retries exceeded")


class AnthropicInferenceClient(InferenceClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        api_url: str = "https://api.anthropic.com/v1/messages",
        thinking: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.thinking = thinking

    async def _post(self, request: dict):
        async with httpx.AsyncClient() as client:
            return await client.post(
                self.api_url,
                timeout=300000,
                json=request,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "output-128k-2025-02-19",
                },
            )

    def _stream(self, client, request: dict):
        return client.stream(
            "POST",
            self.api_url,
            timeout=300000,
            json=request,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "output-128k-2025-02-19,interleaved-thinking-2025-05-14",
            },
        )

    @tracer.start_as_current_span(name="AnthropicInferenceClient.get_generation")
    @backoff.on_exception(backoff.expo, InferenceException, max_time=60)
    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=1024, top_p=0.9, temperature=0.5
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

        if self.thinking:
            request["thinking"] = {"type": "enabled", "budget_tokens": self.thinking}
            request["max_tokens"] += self.thinking
            request["temperature"] = 1
            del request["top_p"]

        response = await self._post(
            request,
        )

        if response.status_code == 400:
            # TODO: ContextLengthExceeded
            raise ValueError(await response.aread())
        elif response.status_code != 200:
            raise InferenceException(await response.aread())

        body: dict = response.json()

        # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
        self._cost.input_tokens += body["usage"]["input_tokens"]
        self._cost.cache_read_input_tokens += body["usage"].get(
            "cache_read_input_tokens", 0
        )
        self._cost.cache_write_input_tokens += body["usage"].get(
            "cache_creation_input_tokens", 0
        )
        self._cost.output_tokens += body["usage"]["output_tokens"]

        await self._trace(request["messages"], body["content"])

        return next(block["text"] for block in body["content"] if "text" in block)

    @tracer.start_as_current_span(name="AnthropicInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=1024, top_p=0.9, temperature=0.5
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

        if self.thinking:
            request["thinking"] = {"type": "enabled", "budget_tokens": self.thinking}
            request["max_tokens"] += self.thinking
            request["temperature"] = 1
            del request["top_p"]

        async with httpx.AsyncClient() as client:
            async with self._stream(
                client,
                request,
            ) as response:
                if response.status_code == 400:
                    raise ValueError(await response.aread())
                elif response.status_code != 200:
                    raise InferenceException(await response.aread())

                out = ""

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
                            out += text
                            yield text
                        elif chunk_type == "message_start":
                            self._cost.input_tokens += chunk_json["message"]["usage"][
                                "input_tokens"
                            ]
                            self._cost.cache_read_input_tokens += chunk_json["message"][
                                "usage"
                            ].get("cache_read_input_tokens", 0)
                            self._cost.cache_write_input_tokens += chunk_json[
                                "message"
                            ]["usage"].get("cache_creation_input_tokens", 0)
                        elif chunk_type == "message_delta":
                            self._cost.output_tokens += chunk_json["usage"][
                                "output_tokens"
                            ]

                await self._trace(request["messages"], [{"text": out}])

    @tracer.start_as_current_span(name="AnthropicInferenceClient._tool_chain_stream")
    async def _tool_chain_stream(
        self,
        serialized_messages,
        tools,
        system=None,
        max_tokens=1024,
        top_p=0.9,
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

        if self.thinking:
            request["thinking"] = {"type": "enabled", "budget_tokens": self.thinking}
            request["max_tokens"] += self.thinking
            request["temperature"] = 1
            del request["top_p"]
            request["tool_choice"] = {"type": "auto"}

        current_content = []
        current_block: dict[str, Any] = {
            "type": None,
        }
        current_json = ""

        async with httpx.AsyncClient() as client:
            async with self._stream(
                client,
                request,
            ) as response:
                if response.status_code == 400:
                    raise ValueError(await response.aread())
                elif response.status_code == 413:
                    raise ContextLengthExceeded(await response.aread())
                elif response.status_code != 200:
                    raise InferenceException(await response.aread())

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    chunk_json = json.loads(line[6:])
                    chunk_type = chunk_json["type"]

                    if chunk_type == "message_start":
                        self._cost.input_tokens += chunk_json["message"]["usage"][
                            "input_tokens"
                        ]
                        self._cost.cache_read_input_tokens += chunk_json["message"][
                            "usage"
                        ].get("cache_read_input_tokens", 0)
                        self._cost.cache_write_input_tokens += chunk_json["message"][
                            "usage"
                        ].get("cache_creation_input_tokens", 0)
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
                        elif block_type == "thinking":
                            current_block = {
                                "type": "thinking",
                                "thinking": chunk_json["content_block"]["thinking"],
                            }
                        elif block_type == "redacted_thinking":
                            current_block = chunk_json["content_block"]
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
                        elif chunk_json["delta"]["type"] == "thinking_delta":
                            current_block["thinking"] += chunk_json["delta"]["thinking"]
                        elif chunk_json["delta"]["type"] == "signature_delta":
                            current_block["signature"] = chunk_json["delta"][
                                "signature"
                            ]
                    elif chunk_type == "content_block_stop":
                        if current_block["type"]:
                            current_content.append(current_block)

                        current_block = {
                            "type": None,
                        }
                    elif chunk_type == "message_delta":
                        if chunk_json["delta"].get("stop_reason") == "tool_use":
                            self._cost.output_tokens += chunk_json["usage"][
                                "output_tokens"
                            ]
                            break
                    elif chunk_type == "error":
                        if chunk_json["error"]["type"] == "invalid_request_error":
                            raise ValueError(chunk_json["error"]["type"])
                        raise InferenceException(chunk_json["error"]["type"])
                    elif chunk_type == "ping":
                        pass
                    else:
                        logger.warning("Unknown message type from Anthropic stream.")

                print(current_content)
                return current_content


def _proto_to_dict(obj):
    type_name = str(type(obj).__name__)
    if hasattr(obj, "DESCRIPTOR"):  # Is protobuf message
        return {
            field.name: _proto_to_dict(getattr(obj, field.name))
            for field in obj.DESCRIPTOR.fields
        }
    elif type_name in ("RepeatedComposite", "RepeatedScalarContainer", "list", "tuple"):
        return [_proto_to_dict(x) for x in obj]
    elif type_name in ("dict", "MapComposite", "MessageMap"):
        return {k: _proto_to_dict(v) for k, v in obj.items()}
    return obj


def smart_unescape_code(s: str) -> str:
    # Common patterns in double-escaped code
    replacements = {
        "\\n": "\n",  # Newlines
        '\\"': '"',  # Quotes
        "\\'": "'",  # Single quotes
        "\\\\": "\\",  # Actual backslashes
        "\\t": "\t",  # Tabs
    }

    result = s
    for escaped, unescaped in replacements.items():
        result = result.replace(escaped, unescaped)

    return result


class GoogleAnthropicInferenceClient(AnthropicInferenceClient):
    def __init__(
        self,
        model: str,
        region: str = "us-east5",
        thinking: Optional[int] = None,
    ):
        InferenceClient.__init__(self)
        self.model = model
        self.region = region
        self.thinking = thinking
        self._get_token()

    def _get_token(self):
        if not hasattr(self, "creds"):
            import google.oauth2.id_token
            import google.auth.transport.requests

            self.creds, self.project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        if not self.creds.token or self.creds.expired:
            request = google.auth.transport.requests.Request()
            self.creds.refresh(request)

        return self.creds.token

    async def _post(self, request: dict):
        request.pop("model", None)
        request["anthropic_version"] = "vertex-2023-10-16"

        async with httpx.AsyncClient() as client:
            return await client.post(
                f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/anthropic/models/{self.model}:streamRawPredict",
                timeout=180,
                json=request,
                headers={
                    "Authorization": f"Bearer {self._get_token()}",
                },
            )

    def _stream(self, client, request: dict):
        request.pop("model", None)
        request["anthropic_version"] = "vertex-2023-10-16"

        return client.stream(
            "POST",
            f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/anthropic/models/{self.model}:streamRawPredict",
            timeout=180,
            json=request,
            headers={
                "Authorization": f"Bearer {self._get_token()}",
            },
        )


class GoogleAnthropicInferenceClient(AnthropicInferenceClient):
    def __init__(
        self,
        model: str,
        region: str = "us-east5",
        thinking: Optional[int] = None,
    ):
        InferenceClient.__init__(self)
        self.model = model
        self.region = region
        self.thinking = thinking
        self._get_token()

    def _get_token(self):
        if not hasattr(self, "creds"):
            import google.oauth2.id_token
            import google.auth.transport.requests

            self.creds, self.project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        if not self.creds.token or self.creds.expired:
            request = google.auth.transport.requests.Request()
            self.creds.refresh(request)

        return self.creds.token

    async def _post(self, request: dict):
        request.pop("model", None)
        request["anthropic_version"] = "vertex-2023-10-16"

        async with httpx.AsyncClient() as client:
            return await client.post(
                f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/anthropic/models/{self.model}:streamRawPredict",
                timeout=180,
                json=request,
                headers={
                    "Authorization": f"Bearer {self._get_token()}",
                },
            )

    def _stream(self, client, request: dict):
        request.pop("model", None)
        request["anthropic_version"] = "vertex-2023-10-16"

        return client.stream(
            "POST",
            f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/anthropic/models/{self.model}:streamRawPredict",
            timeout=180,
            json=request,
            headers={
                "Authorization": f"Bearer {self._get_token()}",
            },
        )


class GoogleGenAIInferenceClient(InferenceClient):
    def __init__(
        self,
        model: str,
        api_key: str,
    ):
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @tracer.start_as_current_span(name="GoogleGenAIInferenceClient._tool_chain_stream")
    async def _tool_chain_stream(
        self,
        serialized_messages,
        tools,
        max_tokens=1024,
        top_p=0.9,
        system=None,
        temperature=0.5,
        # Google is kind of meh with 'any' support
        tool_choice="any",
        middlewares=[],
    ):
        def convert_type_keys(schema):
            if not isinstance(schema, dict):
                return schema

            converted = {}
            for key, value in schema.items():
                if key == "type":
                    # Convert type value to uppercase and change key to type_
                    converted["type"] = value.upper()
                elif key == "properties":
                    # Recursively convert nested property schemas
                    converted[key] = {k: convert_type_keys(v) for k, v in value.items()}
                else:
                    # Recursively convert any other nested dictionaries
                    converted[key] = (
                        convert_type_keys(value) if isinstance(value, dict) else value
                    )

            if schema.get("type") == "OBJECT" and not schema.get("required"):
                schema["required"] = []

            return converted

        # Main tool processing
        filtered_tools = []
        for tool in tools:
            tool_schema = tool[1].copy()
            if "cache_control" in tool_schema:
                del tool_schema["cache_control"]

            # Convert the parameters schema
            parameters = tool_schema.get("parameters") or tool_schema.get(
                "input_schema"
            )

            if parameters:
                parameters = convert_type_keys(
                    tool_schema.get("parameters", tool_schema["input_schema"])
                )

            function_declaration = {
                "name": tool_schema["name"],
                "description": tool_schema["description"],
            }

            if (
                parameters
                and parameters["type"] == "OBJECT"
                and parameters["properties"] != {}
                or parameters
                and parameters["type"] != "OBJECT"
            ):
                function_declaration["parameters"] = parameters

            filtered_tools.append((tool[0], function_declaration))

        # Process messages to match Google's format
        processed_messages = []
        for message in serialized_messages:
            if (
                message["role"] == "user"
                and isinstance(message["content"], list)
                and message["content"][0]["type"] == "tool_result"
            ):
                # Handle tool result responses
                processed_message = types.Content(
                    role=message["role"],
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=message["content"][0]["tool_use_id"],
                                response={"result": message["content"][0]["content"]},
                            )
                        )
                    ],
                )
            else:
                if isinstance(message["content"], list):
                    tool_call = next(
                        (c for c in message["content"] if c["type"] == "tool_use"), None
                    )
                    if tool_call:
                        processed_message = types.Content(
                            role=message["role"],
                            parts=[
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=tool_call["name"],
                                        args=dict(tool_call["input"]),
                                    )
                                )
                            ],
                        )
                    else:
                        text_content = next(
                            c["text"] for c in message["content"] if c["type"] == "text"
                        )
                        processed_message = types.Content(
                            role=message["role"],
                            parts=[types.Part(text=text_content)],
                        )

            processed_messages.append(processed_message)

        tool_config = {"function_calling_config": {"mode": tool_choice.upper()}}

        if self.model == "gemini-2.0-flash-exp":
            await asyncio.sleep(6.5)

        request = self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=processed_messages,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system,
                top_p=top_p,
                max_output_tokens=max_tokens,
                tool_config=tool_config,
                tools=[
                    types.Tool(function_declarations=[t[1]]) for t in filtered_tools
                ],
            ),
        )

        try:
            text_block = None
            tool_call_blocks = []

            # Should be able to handle streaming directly without thread pool
            async for chunk in request:
                if chunk.candidates:
                    for part in chunk.candidates[0].content.parts:
                        if function_call := part.function_call:
                            args_dict = _proto_to_dict(function_call.args)
                            # Clean up over-escaped strings
                            if isinstance(args_dict, dict):
                                for k, v in args_dict.items():
                                    if isinstance(v, str):
                                        args_dict[k] = smart_unescape_code(v)

                            tool_block = {
                                "type": "tool_use",
                                "id": str(len(tool_call_blocks)),
                                "name": function_call.name,
                                "input": args_dict,
                            }
                            tool_call_blocks.append(tool_block)
                            for middleware in middlewares:
                                await middleware(tool_block)
                        elif part.text:
                            if not text_block:
                                text_block = {"type": "text", "text": part.text}
                            else:
                                text_block["text"] += part.text

            return ([text_block] if text_block else []) + tool_call_blocks

        except Exception as e:
            raise InferenceException(str(e))

    @tracer.start_as_current_span(name="GoogleGenAIInferenceClient.get_generation")
    @backoff.on_exception(backoff.expo, InferenceException, max_time=60)
    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.9, temperature=0.5
    ):
        system_instruction = None

        if messages[0].role.value == "system":
            system_instruction = messages[0].content
            messages = messages[1:]
        # Convert messages to Google's format
        processed_messages = [
            types.Content(role=msg.role.value, parts=[types.Part(text=msg.content)])
            for msg in messages
        ]

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=processed_messages,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                top_p=top_p,
                max_output_tokens=max_tokens,
            ),
        )

        if not response.candidates:
            return ""

        await self._trace(
            [
                {
                    "role": msg.role.value,
                    "content": [{"type": "text", "text": msg.content}],
                }
                for msg in messages
            ],
            [{"text": response.text}],
        )

        return response.text

    @tracer.start_as_current_span(name="GoogleGenAIInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.9, temperature=0.5
    ):
        system_instruction = None

        if messages[0].role.value == "system":
            system_instruction = messages[0].content
            messages = messages[1:]

        processed_messages = [
            types.Content(role=msg.role.value, parts=[types.Part(text=msg.content)])
            for msg in messages
        ]

        response = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=processed_messages,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                top_p=top_p,
                max_output_tokens=max_tokens,
            ),
        )

        out = ""

        async for chunk in response:
            if chunk.text:
                out += chunk.text
                yield chunk.text

        await self._trace(
            [
                {
                    "role": msg.role.value,
                    "content": [{"type": "text", "text": msg.content}],
                }
                for msg in messages
            ],
            [{"text": out}],
        )


class OAIRequest(AsimovBase):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int = 4096
    temperature: float = 0.5
    top_p: float = 0.9
    stream: bool = False
    stream_options: Dict[str, Any] = {}
    tools: list[dict] = [{}]
    tool_choice: str = "none"


class OAIInferenceClient(InferenceClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        api_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        super().__init__()
        self.model = model
        self.api_url = api_url
        self.api_key = api_key

    @tracer.start_as_current_span(name="OAIInferenceClient.get_generation")
    @backoff.on_exception(backoff.expo, InferenceException, max_time=60)
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
            ).model_dump(exclude={"stream_options", "tools", "tool_choice"})

        if "o3" in self.model or "o4" in self.model:
            del request["temperature"]
            del request["top_p"]
            del request["max_tokens"]
            request["reasoning_effort"] = "high"

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

            if response.status_code == 400:
                raise ValueError(await response.aread())
            elif response.status_code != 200:
                raise InferenceException(await response.aread())

            body: dict = response.json()
            if body.get("error"):
                raise InferenceException(
                    body["error"]["message"] + f" ({body['error']})"
                )

            try:
                self._cost.input_tokens += body["usage"]["prompt_tokens"]
                self._cost.cache_read_input_tokens += (
                    body["usage"]
                    .get("prompt_tokens_details", {})
                    .get("cached_tokens", 0)
                )
                self._cost.output_tokens += body["usage"]["completion_tokens"]
            except KeyError:
                logger.warning(f"Malformed usage? {repr(body)}")

            await self._trace(
                request["messages"],
                [{"text": body["choices"][0]["message"]["content"]}],
            )

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
        ).model_dump(exclude={"tools", "tool_choice"})

        if "o3" in self.model or "o4" in self.model:
            del request["temperature"]
            del request["top_p"]
            del request["max_tokens"]
            request["reasoning_effort"] = "high"

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
                if response.status_code == 400:
                    raise ValueError(await response.aread())
                elif response.status_code != 200:
                    raise InferenceException(await response.aread())

                out = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break
                        data = json.loads(line[6:])
                        if "error" in data:
                            raise InferenceException(
                                data["error"]["message"] + f" ({data['error']})"
                            )
                        if data["choices"] and data["choices"][0]["delta"].get(
                            "content"
                        ):
                            out += data["choices"][0]["delta"]["content"]
                            yield data["choices"][0]["delta"]["content"]
                        elif data.get("usage"):
                            self._cost.input_tokens += data["usage"]["prompt_tokens"]
                            # No reference to cached tokens in the docs for the streaming API response objects...
                            try:
                                self._cost.cache_read_input_tokens += (
                                    data["usage"]
                                    .get("prompt_tokens_details", {})
                                    .get("cached_tokens", 0)
                                )
                                self._cost.output_tokens += data["usage"][
                                    "completion_tokens"
                                ]
                            except AttributeError as e:
                                logger.warning(f"Malformed usage? {repr(data)}")

                await self._trace(
                    request["messages"],
                    [{"text": out}],
                )

    @property
    def include_cache_control(self):
        return "anthropic" in self.model

    @tracer.start_as_current_span(name="OAIInferenceClient._tool_chain_stream")
    async def _tool_chain_stream(
        self,
        serialized_messages,
        tools,
        system=None,
        max_tokens=1024,
        top_p=0.9,
        temperature=0.5,
        tool_choice="any",
        middlewares=[],
    ):
        if system:
            serialized_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ] + serialized_messages

        def wrap_tool_schema(tool):
            if "cache_control" in tool[1]:
                del tool[1]["cache_control"]

            return (tool[0], {"type": "function", "function": tool[1]})

        tools = list(map(wrap_tool_schema, tools))

        processed_messages = []

        for message in serialized_messages:
            if (not self.include_cache_control) and isinstance(
                message["content"], list
            ):
                for blk in message["content"]:
                    blk.pop("cache_control", None)

            if (
                message["role"] == "user"
                and isinstance(message["content"], list)
                and message["content"][0]["type"] == "tool_result"
            ):
                processed_messages.append(
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
                elif type(message["content"]) == list:
                    message = {
                        "role": message["role"],
                        "content": message["content"][0]["text"],
                    }
                processed_messages.append(message)

        request = OAIRequest(
            model=self.model,
            messages=processed_messages,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            tools=[x[1] for x in tools],
            tool_choice=tool_choice,
        )

        request = request.__dict__

        if "o3" in self.model or "o4" in self.model:
            del request["temperature"]
            del request["top_p"]
            del request["max_tokens"]
            request["reasoning_effort"] = "high"
            request["tool_choice"] = "required"

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.api_url,
                timeout=300000,
                json=request,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status_code == 400:
                    raise ValueError(await response.aread())
                elif response.status_code != 200:
                    raise InferenceException(await response.aread())

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
                            self._cost.input_tokens += data["usage"]["prompt_tokens"]
                            self._cost.output_tokens += data["usage"][
                                "completion_tokens"
                            ]


class OpenRouterInferenceClient(OAIInferenceClient):
    def __init__(self, model: str, api_key: str):
        super().__init__(
            model,
            api_key,
            api_url="https://openrouter.ai/api/v1/chat/completions",
        )

    async def _populate_cost(self, id: str):
        await asyncio.sleep(0.5)
        async with httpx.AsyncClient() as client:
            for _ in range(3):
                response = await client.get(
                    f"https://openrouter.ai/api/v1/generation?id={id}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )

                if response.status_code == 404:
                    await asyncio.sleep(0.5)
                    continue

                if response.status_code != 200:
                    await asyncio.sleep(0.5)
                    continue

                body: dict = response.json()

                self._cost.input_tokens += body["data"]["native_tokens_prompt"]
                self._cost.output_tokens += body["data"]["native_tokens_completion"]
                self._cost.dollar_adjust += -(body["data"]["cache_discount"] or 0)
                return

    @tracer.start_as_current_span(name="OpenRouterInferenceClient._tool_chain_stream")
    async def _tool_chain_stream(
        self,
        serialized_messages,
        tools,
        system=None,
        max_tokens=1024,
        top_p=0.9,
        temperature=0.5,
        tool_choice="any",
        middlewares=[],
    ):
        if system:
            serialized_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ] + serialized_messages

        openrouter_messages = []
        for message in serialized_messages:
            if (not self.include_cache_control) and isinstance(
                message["content"], list
            ):
                for blk in message["content"]:
                    blk.pop("cache_control", None)

            if (
                message["role"] == "user"
                and isinstance(message["content"], list)
                and message["content"][0]["type"] == "tool_result"
            ):
                openrouter_messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_use_id"],
                        # This has to be raw string not a {"type": "text", "text": ...} dict
                        "content": result["content"],
                    }
                    for result in message["content"]
                )
                openrouter_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Here is the result of the tool call",  # need text here to enable caching since cache control can only be set on text blocks.
                            }
                            | (
                                {"cache_control": {"type": "ephemeral"}}
                                if "cache_control" in message["content"][-1]
                                else {}
                            )
                        ],
                    }
                )
            else:
                tool_calls = [
                    content
                    for content in message["content"]
                    if content["type"] == "tool_use"
                ]
                if tool_calls:
                    text = next(
                        (
                            content
                            for content in message["content"]
                            if content["type"] == "text"
                        ),
                        None,
                    )
                    message = {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["input"]),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                openrouter_messages.append(message.copy())

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

        if self.model in ["openai/o3-mini"]:
            request["reasoning_effort"] = "high"

        if self.model in ["anthropic/claude-sonnet-4"]:
            request["reasoning"] = {
                "max_tokens": int(max_tokens * 0.8),
            }
            del request["tool_choice"]

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.api_url,
                json=request,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            ) as response:
                if response.status_code == 400:
                    print(request)
                    raise ValueError(await response.aread())
                elif response.status_code != 200:
                    raise InferenceException(await response.aread())

                id = None
                text_block = None
                tool_call_blocks = []

                async for line in response.aiter_lines():
                    if line == "data: [DONE]":
                        await self._populate_cost(id)
                        return ([text_block] if text_block else []) + tool_call_blocks
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        print(data)
                        if "id" in data:
                            id = data["id"]
                        if data.get("error"):
                            if "context" in str(data["error"]):
                                raise ContextLengthExceeded(str(data["error"]))
                            elif "invalid_request_error" in str(data["error"]):
                                raise NonRetryableException(str(data["error"]))
                            raise InferenceException(
                                data["error"]["message"] + f" ({data['error']})"
                            )
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
                                if tc.get("id"):
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
                                ].get("arguments", "")
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


class VertexInferenceClient(InferenceClient):
    def __init__(self, model: str):
        super().__init__()
        self.model = model

    @tracer.start_as_current_span(name="VertexInferenceClient.get_generation")
    @backoff.on_exception(backoff.expo, InferenceException, max_time=60)
    async def get_generation(
        self,
        messages: List[ChatMessage],
        max_tokens=4096,
        top_p=0.9,
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

        try:
            resp = await model.generate_content_async(
                [
                    vertexai.generative_models.Content(
                        role=msg.role.value,
                        parts=[vertexai.generative_models.Part.from_text(msg.content)],
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
        except google.api_core.exceptions.TooManyRequests:
            raise InferenceException("backoff")
        except google.api_core.exceptions.BadRequest:
            raise ValueError()
        except google.api_core.exceptions.ServerError:
            raise InferenceException("server_error")

        self._cost.input_tokens += resp.usage_metadata.prompt_token_count
        self._cost.output_tokens += resp.usage_metadata.candidates_token_count

        await self._trace(
            [
                {
                    "role": msg.role.value,
                    "content": [{"text": msg.content}],
                }
                for msg in messages
            ],
            [{"text": resp.text}],
        )
        return resp.text

    @tracer.start_as_current_span(name="VertexInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.9, temperature=0.5
    ):
        if messages[0].role == ChatRole.SYSTEM:
            model = vertexai.generative_models.GenerativeModel(
                self.model, system_instruction=messages[0].content
            )
            messages = messages[1:]
        else:
            model = vertexai.generative_models.GenerativeModel(self.model)

        out = ""
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
            out += chunk.text
            yield chunk.text
            if chunk.usage_metadata:
                self._cost.input_tokens += chunk.usage_metadata.prompt_token_count
                self._cost.output_tokens += chunk.usage_metadata.candidates_token_count

        await self._trace(
            [
                {
                    "role": msg.role.value,
                    "content": [{"text": msg.content}],
                }
                for msg in messages
            ],
            [{"text": out}],
        )
