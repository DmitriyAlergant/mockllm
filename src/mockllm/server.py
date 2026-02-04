import logging
from typing import Any, Dict, Union

import tiktoken
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pythonjsonlogger.json import JsonFormatter

from .config import ResponseConfig
from .models import AnthropicChatRequest, OpenAIChatRequest
from .providers.anthropic import AnthropicProvider
from .providers.openai import OpenAIProvider

log_handler = logging.StreamHandler()
log_handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[log_handler])
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock LLM Server")

response_config = ResponseConfig()
openai_provider = OpenAIProvider(response_config)
anthropic_provider = AnthropicProvider(response_config)


def count_tokens(text: str, model: str) -> int:
    """Get realistic token count for text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimation if model not supported
        return len(text.split())


def extract_headers(request: Request) -> Dict[str, Any]:
    """Extract headers from request as a dict."""
    return dict(request.headers)


@app.post("/v1/chat/completions", response_model=None)
async def openai_chat_completion(
    request: OpenAIChatRequest,
    raw_request: Request,
) -> Union[Dict[str, Any], StreamingResponse]:
    """Handle OpenAI chat completion requests"""
    try:
        logger.info(
            "Received chat completion request",
            extra={
                "model": request.model,
                "message_count": len(request.messages),
                "stream": request.stream,
            },
        )
        headers = extract_headers(raw_request)
        return await openai_provider.handle_chat_completion(request, headers)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@app.post("/v1/messages", response_model=None)
async def anthropic_chat_completion(
    request: AnthropicChatRequest,
    raw_request: Request,
) -> Union[Dict[str, Any], StreamingResponse]:
    """Handle Anthropic chat completion requests"""
    try:
        logger.info(
            "Received Anthropic chat completion request",
            extra={
                "model": request.model,
                "message_count": len(request.messages),
                "stream": request.stream,
            },
        )
        headers = extract_headers(raw_request)
        return await anthropic_provider.handle_chat_completion(request, headers)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e
