import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn
import yaml

from . import __version__
from .config import ResponseConfig


def validate_config_file(
    ctx: click.Context, param: click.Parameter, value: Optional[str]
) -> Optional[str]:
    """Validate the config YAML file."""
    if not value:
        return None
    try:
        path = Path(value)
        if not path.exists():
            raise click.BadParameter(f"File {value} does not exist")
        with open(path) as f:
            data = yaml.safe_load(f)
            # Validate structure
            if not isinstance(data, dict):
                raise click.BadParameter("YAML file must contain a dictionary")
            if "responses" not in data:
                raise click.BadParameter("YAML file must contain 'responses' key")
            if not isinstance(data["responses"], dict):
                raise click.BadParameter("'responses' must be a dictionary")
        return value
    except yaml.YAMLError as e:
        raise click.BadParameter(f"Invalid YAML file: {e}")  # noqa: B904


def validate_module_file(
    ctx: click.Context, param: click.Parameter, value: Optional[str]
) -> Optional[str]:
    """Validate the response module Python file."""
    if not value:
        return None
    try:
        path = Path(value)
        if not path.exists():
            raise click.BadParameter(f"File {value} does not exist")
        if not path.suffix == ".py":
            raise click.BadParameter("Module must be a Python file (.py)")

        # Try to load the module to validate it has get_response function
        spec = importlib.util.spec_from_file_location("response_module", path)
        if spec is None or spec.loader is None:
            raise click.BadParameter(f"Could not load module from {value}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["response_module"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "get_response"):
            raise click.BadParameter(
                "Module must define a 'get_response(headers, body)' function"
            )
        if not callable(module.get_response):
            raise click.BadParameter("'get_response' must be callable")

        # Clean up
        del sys.modules["response_module"]

        return value
    except click.BadParameter:
        raise
    except Exception as e:
        raise click.BadParameter(f"Error loading module: {e}")  # noqa: B904


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """MockLLM - A mock server that mimics OpenAI and Anthropic API formats."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    "config_file",
    type=str,
    callback=validate_config_file,
    help="Path to config YAML file",
    default=None,
)
@click.option(
    "--responses",
    "-r",
    "responses_file",
    type=str,
    callback=validate_config_file,
    help="[DEPRECATED: use --config] Path to responses YAML file",
    default=None,
    hidden=True,
)
@click.option(
    "--response-module",
    "-m",
    type=str,
    callback=validate_module_file,
    help="Path to custom Python response module",
    default=None,
)
@click.option("--host", "-h", type=str, help="Host to bind to", default="0.0.0.0")
@click.option("--port", "-p", type=int, help="Port to bind to", default=8000)
@click.option(
    "--reload", is_flag=True, help="Enable auto-reload on file changes", default=True
)
def start(
    config_file: Optional[str],
    responses_file: Optional[str],
    response_module: Optional[str],
    host: str,
    port: int,
    reload: bool,
) -> None:
    """Start the MockLLM server."""
    # Handle --responses as legacy alias for --config
    effective_config = config_file or responses_file

    if responses_file and not config_file:
        click.echo(
            "Warning: --responses is deprecated, use --config instead", err=True
        )

    if response_module and effective_config:
        click.echo(
            "Warning: --response-module overrides --config/--responses", err=True
        )

    env_module = os.getenv("MOCKLLM_RESPONSE_MODULE")
    env_config = os.getenv("MOCKLLM_CONFIG_FILE") or os.getenv(
        "MOCKLLM_RESPONSES_FILE"
    )

    if response_module:
        click.echo(f"Using response module: {response_module}")
        os.environ["MOCKLLM_RESPONSE_MODULE"] = response_module
        os.environ.pop("MOCKLLM_CONFIG_FILE", None)
        os.environ.pop("MOCKLLM_RESPONSES_FILE", None)
    elif effective_config:
        click.echo(f"Using config file: {effective_config}")
        os.environ["MOCKLLM_CONFIG_FILE"] = effective_config
        # Also set legacy env var for backward compatibility
        os.environ["MOCKLLM_RESPONSES_FILE"] = effective_config
        os.environ.pop("MOCKLLM_RESPONSE_MODULE", None)
    elif env_module:
        click.echo(f"Using response module from environment: {env_module}")
    elif env_config:
        click.echo(f"Using config file from environment: {env_config}")
    else:
        # Default to responses.yml if it exists
        if Path("responses.yml").exists():
            click.echo("Using default config file: responses.yml")
            os.environ["MOCKLLM_CONFIG_FILE"] = "responses.yml"
            os.environ["MOCKLLM_RESPONSES_FILE"] = "responses.yml"

    click.echo(f"Starting server on {host}:{port}")
    uvicorn.run("mockllm.server:app", host=host, port=port, reload=reload)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--type",
    "-t",
    "file_type",
    type=click.Choice(["config", "module"]),
    default=None,
    help="Type of file to validate (auto-detected if not specified)",
)
def validate(file_path: str, file_type: Optional[str]) -> None:
    """Validate a config YAML file or response module."""
    path = Path(file_path)

    # Auto-detect file type if not specified
    if file_type is None:
        if path.suffix == ".py":
            file_type = "module"
        else:
            file_type = "config"

    if file_type == "module":
        _validate_module(path)
    else:
        _validate_config(path)


def _validate_config(path: Path) -> None:
    """Validate a config YAML file."""
    try:
        # First do structural validation
        with open(path) as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError("YAML file must contain a dictionary")
            if "responses" not in data:
                raise ValueError("YAML file must contain 'responses' key")
            if not isinstance(data["responses"], dict):
                raise ValueError("'responses' must be a dictionary")

        # Then try to load it via ResponseConfig
        config = ResponseConfig(yaml_path=str(path))
        config.load_responses()
        click.echo(click.style("Valid config file", fg="green"))
        click.echo(f"Found {len(config.responses)} responses")
    except Exception as e:
        click.echo(click.style("Invalid config file", fg="red"))
        click.echo(str(e))
        exit(1)


def _validate_module(path: Path) -> None:
    """Validate a response module."""
    try:
        spec = importlib.util.spec_from_file_location("response_module", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module from {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["response_module"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "get_response"):
            raise ValueError(
                "Module must define a 'get_response(headers, body)' function"
            )
        if not callable(module.get_response):
            raise ValueError("'get_response' must be callable")

        # Clean up
        del sys.modules["response_module"]

        click.echo(click.style("Valid response module", fg="green"))
        click.echo("Module defines get_response function")
    except Exception as e:
        click.echo(click.style("Invalid response module", fg="red"))
        click.echo(str(e))
        exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()
