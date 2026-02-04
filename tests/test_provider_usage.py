"""Tests for usage overrides from response modules."""

import tempfile
from pathlib import Path

import pytest

from mockllm.config import ResponseConfig
from mockllm.models import (
    AnthropicChatRequest,
    AnthropicMessage,
    OpenAIChatRequest,
    OpenAIMessage,
)
from mockllm.providers.anthropic import AnthropicProvider
from mockllm.providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_usage_override_from_module():
    module_content = '''
def get_response(headers, body):
    return ("hello", {
        "prompt_tokens": 19,
        "completion_tokens": 10,
        "total_tokens": 29,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "audio_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0
        }
    })
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(module_content)
        module_path = f.name

    try:
        config = ResponseConfig(module_path=module_path)
        provider = OpenAIProvider(config)
        request = OpenAIChatRequest(
            model="gpt-4",
            messages=[OpenAIMessage(role="user", content="hi")],
            stream=False,
        )
        response = await provider.handle_chat_completion(request, headers={})
        assert response["choices"][0]["message"]["content"] == "hello"
        assert response["usage"] == {
            "prompt_tokens": 19,
            "completion_tokens": 10,
            "total_tokens": 29,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "audio_tokens": 0,
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        }
    finally:
        Path(module_path).unlink()


@pytest.mark.asyncio
async def test_anthropic_usage_override_from_module():
    module_content = '''
def get_response(headers, body):
    return ("hello", {
        "cache_creation": {
            "ephemeral_1h_input_tokens": 0,
            "ephemeral_5m_input_tokens": 0
        },
        "cache_creation_input_tokens": 2051,
        "cache_read_input_tokens": 2051,
        "input_tokens": 2095,
        "output_tokens": 503,
        "server_tool_use": {
            "web_search_requests": 0
        },
        "service_tier": "standard"
    })
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(module_content)
        module_path = f.name

    try:
        config = ResponseConfig(module_path=module_path)
        provider = AnthropicProvider(config)
        request = AnthropicChatRequest(
            model="claude-3-sonnet-20240229",
            messages=[AnthropicMessage(role="user", content="hi")],
            stream=False,
        )
        response = await provider.handle_chat_completion(request, headers={})
        assert response["content"][0]["text"] == "hello"
        assert response["usage"] == {
            "cache_creation": {
                "ephemeral_1h_input_tokens": 0,
                "ephemeral_5m_input_tokens": 0,
            },
            "cache_creation_input_tokens": 2051,
            "cache_read_input_tokens": 2051,
            "input_tokens": 2095,
            "output_tokens": 503,
            "server_tool_use": {
                "web_search_requests": 0,
            },
            "service_tier": "standard",
        }
    finally:
        Path(module_path).unlink()
