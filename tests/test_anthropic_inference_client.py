import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from asimov.services.inference_clients import (
    AnthropicInferenceClient,
    ChatMessage,
    ChatRole,
)


@pytest.mark.asyncio
async def test_tool_chain_streaming():
    # Mock response chunks that simulate the streaming response
    response_chunks = [
        'data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":0},"content":[],"stop_reason":null}}\n',
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check"}}\n',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" the weather"}}\n',
        'data: {"type":"content_block_stop","index":0}\n',
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tool_123","name":"get_weather","input":{}}}\n',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"location\\": \\""}}\n',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"San Francisco"}}\n',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\\"}"}}\n',
        'data: {"type":"content_block_stop","index":1}\n',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":20}}\n',
        'data: {"type":"message_stop"}\n',
    ]

    # Mock the weather tool function
    get_weather = AsyncMock(return_value={"temperature": 72, "conditions": "sunny"})

    tools = [
        (
            get_weather,
            {"name": "get_weather", "description": "Get weather for location"},
        )
    ]

    # Create test messages
    messages = [
        ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant"),
        ChatMessage(
            role=ChatRole.USER, content="What's the weather like in San Francisco?"
        ),
    ]

    # Mock the HTTP client's stream method
    mock_aiter = MagicMock(name="aiter")
    mock_aiter.__aiter__.return_value = response_chunks

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = MagicMock(return_value=mock_aiter)
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = MagicMock()

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_response)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = MagicMock()

    # Create the client and patch httpx.AsyncClient
    client = AnthropicInferenceClient("claude-3-5-sonnet-20241022", "test-key")

    with patch("httpx.AsyncClient") as mock_http_client:
        mock_http_client.return_value = mock_client

        # Call tool_chain
        result = await client.tool_chain(
            messages=messages, tools=tools, max_iterations=1
        )

        # Verify the request was made correctly
        mock_client.stream.assert_called_once()
        call_args = mock_client.stream.call_args
        assert call_args[0][0] == "POST"  # First positional arg is the method
        assert call_args[1]["headers"]["anthropic-version"] == "2023-06-01"
        assert call_args[1]["json"]["stream"] == True
        assert call_args[1]["json"]["tools"] == [tools[0][1]]

        # Verify the result contains the expected messages
        print(result)
        assert len(result) == 3  # user, assistant, tool result
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][0]["text"] == "Let me check the weather"
        assert result[1]["content"][1]["type"] == "tool_use"
        assert result[1]["content"][1]["name"] == "get_weather"
        assert result[1]["content"][1]["input"] == {"location": "San Francisco"}
        assert result[2]["role"] == "user"
        assert result[2]["content"][0]["type"] == "tool_result"
        assert result[2]["content"][0]["content"] == "{'temperature': 72, 'conditions': 'sunny'}"

        get_weather.assert_awaited_once_with({"location": "San Francisco"})
