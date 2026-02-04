"""Tests for custom response module functionality."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def reset_env():
    """Reset environment variables before each test."""
    # Save original values
    orig_module = os.environ.get("MOCKLLM_RESPONSE_MODULE")
    orig_config = os.environ.get("MOCKLLM_CONFIG_FILE")
    orig_responses = os.environ.get("MOCKLLM_RESPONSES_FILE")

    # Clear them
    keys = ["MOCKLLM_RESPONSE_MODULE", "MOCKLLM_CONFIG_FILE", "MOCKLLM_RESPONSES_FILE"]
    for key in keys:
        if key in os.environ:
            del os.environ[key]

    yield

    # Restore original values
    if orig_module is not None:
        os.environ["MOCKLLM_RESPONSE_MODULE"] = orig_module
    if orig_config is not None:
        os.environ["MOCKLLM_CONFIG_FILE"] = orig_config
    if orig_responses is not None:
        os.environ["MOCKLLM_RESPONSES_FILE"] = orig_responses


def test_module_get_response():
    """Test that a custom module's get_response is called correctly."""
    from mockllm.config import ResponseConfig

    # Create a temporary module file
    module_content = '''
def get_response(headers, body):
    """Return a response based on the request."""
    model = body.get("model", "unknown")
    return f"Response from custom module for model: {model}"
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(module_content)
        module_path = f.name

    try:
        config = ResponseConfig(module_path=module_path)
        response = config.get_response(
            headers={"authorization": "Bearer test"},
            body={"model": "gpt-4", "messages": []},
        )
        assert "Response from custom module" in response
        assert "gpt-4" in response
    finally:
        Path(module_path).unlink()


def test_module_with_headers_inspection():
    """Test that module receives headers correctly."""
    from mockllm.config import ResponseConfig

    module_content = '''
def get_response(headers, body):
    """Return auth header value if present."""
    auth = headers.get("authorization", "no-auth")
    return f"Auth: {auth}"
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(module_content)
        module_path = f.name

    try:
        config = ResponseConfig(module_path=module_path)
        response = config.get_response(
            headers={"authorization": "Bearer secret-token"},
            body={"messages": []},
        )
        assert "Auth: Bearer secret-token" in response
    finally:
        Path(module_path).unlink()


def test_module_with_body_inspection():
    """Test that module receives body correctly."""
    from mockllm.config import ResponseConfig

    module_content = '''
def get_response(headers, body):
    """Return user message if present."""
    messages = body.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return f"You said: {msg.get('content', '')}"
    return "No user message"
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(module_content)
        module_path = f.name

    try:
        config = ResponseConfig(module_path=module_path)
        response = config.get_response(
            headers={},
            body={"messages": [{"role": "user", "content": "Hello world"}]},
        )
        assert "You said: Hello world" in response
    finally:
        Path(module_path).unlink()


def test_yaml_fallback_when_no_module():
    """Test that YAML config is used when no module is provided."""
    from mockllm.config import ResponseConfig

    yaml_content = """responses:
  hello: world
defaults:
  unknown_response: I don't know
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        config = ResponseConfig(yaml_path=yaml_path)
        response = config.get_response(
            headers={},
            body={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert response == "world"

        # Test unknown prompt
        response = config.get_response(
            headers={},
            body={"messages": [{"role": "user", "content": "unknown"}]},
        )
        assert response == "I don't know"
    finally:
        Path(yaml_path).unlink()


def test_yaml_path_overrides_module_env():
    """Test that explicit yaml_path overrides MOCKLLM_RESPONSE_MODULE."""
    from mockllm.config import ResponseConfig

    module_content = '''
def get_response(headers, body):
    return "module-response"
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(module_content)
        module_path = f.name

    yaml_content = """responses:
  hello: world
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        os.environ["MOCKLLM_RESPONSE_MODULE"] = module_path
        config = ResponseConfig(yaml_path=yaml_path)
        response = config.get_response(
            headers={},
            body={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert response == "world"
    finally:
        Path(module_path).unlink()
        Path(yaml_path).unlink()


def test_module_response_with_usage():
    """Test that module can return usage along with content."""
    from mockllm.config import ResponseConfig

    module_content = '''
def get_response(headers, body):
    return ("module-response", {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(module_content)
        module_path = f.name

    try:
        config = ResponseConfig(module_path=module_path)
        payload = config.get_response_payload(headers={}, body={})
        assert payload.content == "module-response"
        assert payload.usage == {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }
    finally:
        Path(module_path).unlink()
