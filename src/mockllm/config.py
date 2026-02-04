import asyncio
import importlib.util
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, AsyncGenerator, Callable, Dict, Generator, Optional, cast

import yaml
from fastapi import HTTPException
from pythonjsonlogger.json import JsonFormatter

log_handler = logging.StreamHandler()
log_handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[log_handler])
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResponsePayload:
    content: str
    usage: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None


class ResponseConfig:
    """Handles loading and managing response configurations from YAML or module."""

    def __init__(
        self,
        yaml_path: Optional[str] = None,
        module_path: Optional[str] = None,
    ):
        self.yaml_path: Optional[str] = None
        self.response_module: Optional[ModuleType] = None
        self.module_get_response: Optional[
            Callable[[Dict[str, Any], Dict[str, Any]], Any]
        ] = None

        # Explicit args override environment variables
        if module_path is not None:
            self.module_path = module_path
        elif yaml_path is None:
            self.module_path = os.getenv("MOCKLLM_RESPONSE_MODULE")
        else:
            self.module_path = None

        if self.module_path:
            self._load_module()
        else:
            # Fall back to YAML config
            self.yaml_path = cast(
                str,
                yaml_path
                or os.getenv("MOCKLLM_CONFIG_FILE")
                or os.getenv("MOCKLLM_RESPONSES_FILE", "responses.yml"),
            )

        self.last_modified = 0
        self.responses: Dict[str, str] = {}
        self.default_response = "I don't know the answer to that."
        self.lag_enabled = False
        self.lag_factor = 10

        if not self.module_path:
            self.load_responses()

    def _load_module(self) -> None:
        """Load the custom response module."""
        if not self.module_path:
            return

        try:
            path = Path(self.module_path)
            if not path.exists():
                raise ValueError(f"Module file not found: {self.module_path}")

            spec = importlib.util.spec_from_file_location("response_module", path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load module from {self.module_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["response_module"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "get_response"):
                raise ValueError(
                    "Module must define a 'get_response(headers, body)' function"
                )
            if not callable(module.get_response):
                raise ValueError("'get_response' must be callable")

            self.response_module = module
            self.module_get_response = module.get_response
            logger.info(f"Loaded response module from {self.module_path}")

        except Exception as e:
            logger.error(f"Error loading response module: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to load response module: {str(e)}"
            ) from e

    def load_responses(self) -> None:
        """Load or reload responses from YAML file if modified."""
        if self.module_path:
            # Module mode - no YAML to load
            return

        if not self.yaml_path:
            return

        try:
            path = Path(self.yaml_path)
            current_mtime = path.stat().st_mtime
            if current_mtime > self.last_modified:
                with path.open("r") as f:
                    data = yaml.safe_load(f)
                    self.responses = data.get("responses", {})
                    self.default_response = data.get("defaults", {}).get(
                        "unknown_response", self.default_response
                    )
                    settings = data.get("settings", {})
                    self.lag_enabled = settings.get("lag_enabled", False)
                    self.lag_factor = settings.get("lag_factor", 10)
                self.last_modified = int(current_mtime)
                logger.info(
                    f"Loaded {len(self.responses)} responses from {self.yaml_path}"
                )
        except Exception as e:
            logger.error(f"Error loading responses: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to load response configuration"
            ) from e

    def get_response(
        self, headers: Dict[str, Any], body: Dict[str, Any]
    ) -> str:
        """Get response for a given request.

        Args:
            headers: HTTP headers dict (e.g., {"authorization": "Bearer ..."})
            body: Request body dict (e.g., {"model": "gpt-4", "messages": [...]})

        Returns:
            Response string to return to client
        """
        payload = self.get_response_payload(headers, body)
        return payload.content

    def get_response_payload(
        self, headers: Dict[str, Any], body: Dict[str, Any]
    ) -> ResponsePayload:
        """Get response payload for a given request.

        Args:
            headers: HTTP headers dict (e.g., {"authorization": "Bearer ..."})
            body: Request body dict (e.g., {"model": "gpt-4", "messages": [...]})

        Returns:
            ResponsePayload with content and optional usage/reasoning.
        """
        if self.module_get_response is not None:
            # Use custom module
            return self._normalize_module_response(
                self.module_get_response(headers, body)
            )

        # Use YAML config - extract prompt from body
        self.load_responses()  # Check for updates
        prompt = self._extract_prompt(body)
        return ResponsePayload(content=self.responses.get(prompt, self.default_response))

    def _normalize_module_response(self, result: Any) -> ResponsePayload:
        """Normalize module responses to ResponsePayload."""
        if isinstance(result, ResponsePayload):
            return result

        if isinstance(result, str):
            return ResponsePayload(content=result)

        if isinstance(result, (tuple, list)):
            if len(result) == 2:
                content, usage = result
                reasoning = None
            elif len(result) == 3:
                content, reasoning, usage = result
            else:
                raise ValueError(
                    "get_response must return a string, "
                    "a (content, usage) tuple, or a (content, reasoning, usage) tuple"
                )

            if not isinstance(content, str):
                raise ValueError("response content must be a string")
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError("reasoning must be a string when provided")
            if usage is not None and not isinstance(usage, dict):
                raise ValueError("usage must be a dict when provided")

            return ResponsePayload(content=content, reasoning=reasoning, usage=usage)

        raise ValueError(
            "get_response must return a string, "
            "a (content, usage) tuple, or a (content, reasoning, usage) tuple"
        )

    def _extract_prompt(self, body: Dict[str, Any]) -> str:
        """Extract the user prompt from the request body."""
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle structured content (e.g., Anthropic format)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            return str(text) if text else ""
        return ""

    def get_streaming_response(
        self,
        headers: Dict[str, Any],
        body: Dict[str, Any],
        chunk_size: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Generator that yields response content
        character by character or in chunks."""
        response = self.get_response(headers, body)
        if chunk_size:
            # Yield response in chunks
            for i in range(0, len(response), chunk_size):
                yield response[i : i + chunk_size]
        else:
            # Yield response character by character
            for char in response:
                yield char

    async def get_response_with_lag(
        self, headers: Dict[str, Any], body: Dict[str, Any]
    ) -> str:
        """Get response with artificial lag for non-streaming responses."""
        payload = await self.get_response_payload_with_lag(headers, body)
        return payload.content

    async def get_response_payload_with_lag(
        self, headers: Dict[str, Any], body: Dict[str, Any]
    ) -> ResponsePayload:
        """Get response payload with artificial lag for non-streaming responses."""
        payload = self.get_response_payload(headers, body)
        if self.lag_enabled:
            # Base delay on response length and lag factor
            delay = len(payload.content) / (self.lag_factor * 10)
            await asyncio.sleep(delay)
        return payload

    async def get_streaming_response_with_lag(
        self,
        headers: Dict[str, Any],
        body: Dict[str, Any],
        chunk_size: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Generator that yields response content with artificial lag."""
        response = self.get_response(headers, body)

        if chunk_size:
            for i in range(0, len(response), chunk_size):
                chunk = response[i : i + chunk_size]
                if self.lag_enabled:
                    delay = len(chunk) / (self.lag_factor * 10)
                    await asyncio.sleep(delay)
                yield chunk
        else:
            for char in response:
                if self.lag_enabled:
                    # Add random variation to character delay
                    base_delay = 1 / (self.lag_factor * 10)
                    variation = random.uniform(-0.5, 0.5) * base_delay
                    delay = max(0, base_delay + variation)
                    await asyncio.sleep(delay)
                yield char
