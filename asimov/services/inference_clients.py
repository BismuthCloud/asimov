# BISMUTH FILE: inference_clients.py

import json
from typing import List, Dict, Any, Callable, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import boto3
from botocore.config import Config
import httpx


class ChatRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage:
    def __init__(self, content: str, role: ChatRole):
        self.content = content
        self.role = role


@dataclass
class ParticipantMessageProtocol:
    open: str
    close: str


class ModelMessageProtocol(ABC):
    @property
    @abstractmethod
    def system_message_protocol(self) -> ParticipantMessageProtocol:
        pass

    @property
    @abstractmethod
    def user_message_protocol(self) -> ParticipantMessageProtocol:
        pass

    @property
    @abstractmethod
    def ai_message_protocol(self) -> ParticipantMessageProtocol:
        pass

    def apply_protocol(self, message: ChatMessage) -> str:
        if message.role == ChatRole.SYSTEM:
            protocol = self.system_message_protocol
        elif message.role == ChatRole.USER:
            protocol = self.user_message_protocol
        elif message.role == ChatRole.ASSISTANT:
            protocol = self.ai_message_protocol
        else:
            raise ValueError("Invalid role")

        return protocol.open + message.content + protocol.close


class MixtralMessageProtocol(ModelMessageProtocol):
    @property
    def system_message_protocol(self) -> ParticipantMessageProtocol:
        return ParticipantMessageProtocol(open="<s>[INST]", close="[/INST]")

    @property
    def user_message_protocol(self) -> ParticipantMessageProtocol:
        return ParticipantMessageProtocol(open="<s>[INST]", close="[/INST]")

    @property
    def ai_message_protocol(self) -> ParticipantMessageProtocol:
        return ParticipantMessageProtocol(open="", close="</s>")


@dataclass
class MistralRequest:
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 50


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
    OpenAI = "OpenAI"
    Mistral = "Mistral"


class InferenceClient(ABC):
    @abstractmethod
    async def connect_and_listen(self, messages: List[ChatMessage]):
        pass


class BedrockInferenceClient(InferenceClient):
    def __init__(self, model: str):
        self.model = model
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            config=Config(retries={"max_attempts": 10, "mode": "standard"}),
        )
        self.model_family = ModelFamily.Anthropic
        self.anthropic_version = "bedrock-2023-05-31"
        self.mixtral_protocol = MixtralMessageProtocol()

    async def connect_and_listen(
        self,
        messages: List[ChatMessage],
    ):
        if self.model_family == ModelFamily.Mistral:
            protocol_applied = "".join(
                [self.mixtral_protocol.apply_protocol(msg) for msg in messages]
            )
            body = MistralRequest(
                prompt=protocol_applied, max_tokens=2048, temperature=0.1
            )
        else:
            body = AnthropicRequest(
                anthropic_version=self.anthropic_version,
                system=messages[0].content,
                top_p=0.5,
                max_tokens=8192,
                messages=[
                    {
                        "role": msg.role.value,
                        "content": [{"type": "text", "text": msg.content}],
                    }
                    for msg in messages[1:]
                ],
            )

        response = self.bedrock_runtime.invoke_model_with_response_stream(
            body=json.dumps(body.__dict__),
            modelId=self.model,
            contentType="application/json",
            accept="application/json",
        )

        for chunk in response["body"]:
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
                    chunk_json["delta"]["text"] if content_type == "text_delta" else ""
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

    async def connect_and_listen(self, messages: List[ChatMessage]):
        request = {
            "model": self.model,
            "system": messages[0]["content"],
            "top_p": 0.5,
            "max_tokens": 4096,
            "messages": [
                {"role": msg["role"], "content": msg["content"]} for msg in messages[1:]
            ],
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.api_url,
                timeout=300000,
                json=request,
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            ) as response:
                print(response.status_code)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        chunk_type = data["type"]

                        if chunk_type in [
                            "message_start",
                            "content_block_start",
                            "message_delta",
                            "error",
                            "message_stop",
                        ]:
                            text = ""
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
