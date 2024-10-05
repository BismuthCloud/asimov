import json
from typing import List, Dict, Any, Callable, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import aioboto3
import httpx


class ChatRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage:
    content: str
    role: ChatRole
    cache_marker: bool

    def __init__(self, content: str, role: ChatRole, cache_marker: bool = False):
        self.content = content
        self.role = role
        self.cache_marker = cache_marker


@dataclass
class AnthropicRequest:
    anthropic_version: str
    system: str
    messages: List[Dict[str, Any]]
    max_tokens: int = 512
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class AnthropicMessageContent:
    type: str
    text: str


@dataclass
class AnthropicMessage:
    role: str
    content: List[AnthropicMessageContent]


class ModelFamily(Enum):
    Anthropic = "Anthropic"


class InferenceClient(ABC):
    @abstractmethod
    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.5
    ):
        pass

    @abstractmethod
    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.5
    ):
        pass


class BedrockInferenceClient(InferenceClient):
    def __init__(self, model: str):
        self.model = model
        self.session = aioboto3.Session()
        self.model_family = ModelFamily.Anthropic
        self.anthropic_version = "bedrock-2023-05-31"

    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.5
    ):
        body = AnthropicRequest(
            anthropic_version=self.anthropic_version,
            system=messages[0]["content"],
            top_p=top_p,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                }
                for msg in messages[1:]
            ],
        )

        async with self.session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
        ) as client:
            response = await client.invoke_model(
                body=json.dumps(body.__dict__),
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
            )

            body = json.loads(await response["body"].read())
            return body["content"][0]["text"]

    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.5
    ):
        body = AnthropicRequest(
            anthropic_version=self.anthropic_version,
            system=messages[0]["content"],
            top_p=top_p,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                }
                for msg in messages[1:]
            ],
        )

        async with self.session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
        ) as client:
            response = await client.invoke_model_with_response_stream(
                body=json.dumps(body.__dict__),
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
            )

            async for chunk in response["body"]:
                chunk_json = json.loads(chunk["chunk"]["bytes"].decode())
                chunk_type = chunk_json["type"]

                if chunk_type in [
                    "message_start",
                    "content_block_start",
                    "message_delta",
                    "error",
                    "message_stop",
                ]:
                    text = ""
                elif chunk_type == "content_block_delta":
                    content_type = chunk_json["delta"]["type"]
                    text = (
                        chunk_json["delta"]["text"]
                        if content_type == "text_delta"
                        else ""
                    )
                else:
                    text = ""

                yield text


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

    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.5
    ):
        request = {
            "model": self.model,
            "system": [
                {"type": "text", "text": messages[0]["content"]}
                | (
                    {"cache_control": {"type": "ephemeral"}}
                    if messages[0].get("cache_marker")
                    else {}
                )
            ],
            "top_p": top_p,
            "max_tokens": max_tokens,
            "messages": [
                {"role": msg["role"], "content": msg["content"]}
                | (
                    {"cache_control": {"type": "ephemeral"}}
                    if msg.get("cache_marker")
                    else {}
                )
                for msg in messages[1:]
            ],
            "stream": False,
        }
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
                response.raise_for_status()

            return response.json()["content"][0]["text"]

    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=0.5
    ):
        request = {
            "model": self.model,
            "system": [
                {"type": "text", "text": messages[0]["content"]}
                | (
                    {"cache_control": {"type": "ephemeral"}}
                    if messages[0].get("cache_marker")
                    else {}
                )
            ],
            "top_p": top_p,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": msg["role"],
                    "content": [
                        {"type": "text", "text": msg["content"]}
                        | (
                            {"cache_control": {"type": "ephemeral"}}
                            if msg.get("cache_marker")
                            else {}
                        )
                    ],
                }
                for msg in messages[1:]
            ],
            "stream": True,
        }

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
                print("streaming status code:", response.status_code)
                from pprint import pprint

                async for line in response.aiter_lines():
                    if response.status_code != 200:
                        message_logs = [{"role": msg["role"]} for msg in messages[1:]]

                        print(line)
                        pprint(message_logs)

                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        chunk_type = data["type"]

                        if chunk_type in [
                            "content_block_start",
                            "message_delta",
                            "error",
                            "message_stop",
                        ]:
                            text = ""
                        elif chunk_type == "message_start":
                            text = ""
                            print(data["message"]["usage"])
                        elif chunk_type == "content_block_delta":
                            content_type = data["delta"]["type"]
                            text = (
                                data["delta"]["text"]
                                if content_type == "text_delta"
                                else ""
                            )
                        else:
                            text = ""
                        yield text


@dataclass
class OAIRequest:
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False


@dataclass
class OAIReasoningRequest:
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int = 4096


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

    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=1.0
    ):
        if self.model in ["o1-preview", "o1-mini"]:
            print("REASONING_REQUEST")
            request = {
                "model": self.model,
                "messages": messages,
            }
        else:
            request = OAIRequest(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False,
            ).__dict__

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

            if response.status_code != 200:
                response.raise_for_status()

            return response.json()["choices"][0]["message"]["content"]

    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=4096, top_p=1.0
    ):
        request = OAIRequest(
            model=self.model,
            messages=[
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ],
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
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
                if response.status_code != 200:
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break
                        data = json.loads(line[6:])
                        if data["choices"][0]["delta"].get("content"):
                            yield data["choices"][0]["delta"]["content"]
