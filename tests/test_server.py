from unittest.mock import mock_open, patch

import pytest
from fastapi.testclient import TestClient

# Move MOCK_YAML_CONTENT definition to top
MOCK_YAML_CONTENT = """
responses:
  default: "Hello, this is a mock response."
"""

# Create a patch for ResponseConfig before importing server
with patch("builtins.open", mock_open(read_data=MOCK_YAML_CONTENT)), patch(
    "os.path.exists", return_value=True
), patch("mockllm.config.ResponseConfig.load_responses"):
    from mockllm.server import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_responses_file():
    # Update the fixture to also patch ResponseConfig.load_responses
    with patch("builtins.open", mock_open(read_data=MOCK_YAML_CONTENT)), patch(
        "os.path.exists", return_value=True
    ), patch("mockllm.config.ResponseConfig.load_responses"):
        yield


def test_openai_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "test message"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert "usage" in data


def test_anthropic_chat_completion():
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "test message"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert len(data["content"]) > 0
    assert "usage" in data


def test_openai_streaming():
    async def fake_stream(headers, body, chunk_size=None):
        for chunk in ["A", "B"]:
            yield chunk

    with patch(
        "mockllm.server.response_config.get_streaming_response_with_lag",
        new=fake_stream,
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-llm",
                "messages": [{"role": "user", "content": "test message"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert '"content":"A"' in response.text
        assert '"content":"B"' in response.text


def test_anthropic_streaming():
    async def fake_stream(headers, body, chunk_size=None):
        for chunk in ["A", "B"]:
            yield chunk

    with patch(
        "mockllm.server.response_config.get_streaming_response_with_lag",
        new=fake_stream,
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "test message"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert '"text":"A"' in response.text
        assert '"text":"B"' in response.text


def test_invalid_request():
    response = client.post(
        "/v1/chat/completions", json={"model": "mock-llm", "messages": []}
    )
    assert response.status_code == 500


def test_openai_with_authorization_header():
    """Test that authorization headers are passed through."""
    from mockllm.config import ResponsePayload

    async def fake_get_response_payload_with_lag(headers, body):
        assert headers.get("authorization") == "Bearer test-token"
        assert body.get("model") == "mock-llm"
        return ResponsePayload(content="header-ok")

    with patch(
        "mockllm.server.response_config.get_response_payload_with_lag",
        new=fake_get_response_payload_with_lag,
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-llm",
                "messages": [{"role": "user", "content": "test message"}],
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "header-ok"


def test_anthropic_with_api_key_header():
    """Test that x-api-key headers are passed through."""
    from mockllm.config import ResponsePayload

    async def fake_get_response_payload_with_lag(headers, body):
        assert headers.get("x-api-key") == "test-api-key"
        assert body.get("model") == "claude-3-sonnet-20240229"
        return ResponsePayload(content="header-ok")

    with patch(
        "mockllm.server.response_config.get_response_payload_with_lag",
        new=fake_get_response_payload_with_lag,
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "user", "content": "test message"}],
            },
            headers={"x-api-key": "test-api-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["content"][0]["text"] == "header-ok"
