"""Tests for CLI functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mockllm.cli import cli


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


def test_validate_module_success():
    """Test validating a valid module."""
    module_content = '''
def get_response(headers, body):
    return "test"
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(module_content)
        module_path = f.name

    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", module_path])
        assert result.exit_code == 0
        assert "Valid response module" in result.output
    finally:
        Path(module_path).unlink()


def test_validate_module_missing_function():
    """Test validating a module without get_response."""
    module_content = '''
def wrong_function(headers, body):
    return "test"
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(module_content)
        module_path = f.name

    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", module_path])
        assert result.exit_code == 1
        assert "get_response" in result.output
    finally:
        Path(module_path).unlink()


def test_validate_config_success():
    """Test validating a valid config file."""
    yaml_content = """responses:
  hello: world
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", yaml_path])
        assert result.exit_code == 0
        assert "Valid config file" in result.output
    finally:
        Path(yaml_path).unlink()


def test_validate_config_missing_responses():
    """Test validating a config file without responses key."""
    yaml_content = """settings:
  lag_enabled: true
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", yaml_path])
        assert result.exit_code == 1
        assert "responses" in result.output.lower()
    finally:
        Path(yaml_path).unlink()


def test_cli_help():
    """Test that CLI help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "MockLLM" in result.output


def test_start_help():
    """Test that start --help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["start", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--response-module" in result.output


def test_start_config_overrides_module_env():
    """Test that --config overrides MOCKLLM_RESPONSE_MODULE."""
    yaml_content = """responses:
  hello: world
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        f.write(yaml_content)
        yaml_path = f.name

    def fake_run(*args, **kwargs):
        assert os.environ.get("MOCKLLM_CONFIG_FILE") == yaml_path
        assert os.environ.get("MOCKLLM_RESPONSES_FILE") == yaml_path
        assert "MOCKLLM_RESPONSE_MODULE" not in os.environ

    try:
        runner = CliRunner()
        with patch("mockllm.cli.uvicorn.run", new=fake_run):
            result = runner.invoke(
                cli,
                ["start", "--config", yaml_path],
                env={"MOCKLLM_RESPONSE_MODULE": "some_module.py"},
            )
        assert result.exit_code == 0
    finally:
        Path(yaml_path).unlink()
