"""
Example custom response module for MockLLM.

This module demonstrates how to implement custom response logic that has full
access to request headers and body. Use this for complex matching scenarios
that go beyond simple prompt-to-response mapping.

Usage:
    mockllm start --response-module example_response_module.py
"""

from typing import Any, Dict


def get_response(headers: Dict[str, Any], body: Dict[str, Any]) -> str:
    """
    Generate a response based on the request headers and body.

    Args:
        headers: HTTP headers dict (e.g., {"authorization": "Bearer ..."})
        body: Request body dict (e.g., {"model": "gpt-4", "messages": [...]})

    Returns:
        Response string to return to the client, or a tuple of
        (response, usage) to override the usage object. Usage format depends
        on the endpoint (OpenAI vs Anthropic) and may include nested fields.

    Examples:
        # Access the model name
        model = body.get("model", "unknown")

        # Access the user's message
        messages = body.get("messages", [])
        user_message = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            ""
        )

        # Access authorization header
        auth = headers.get("authorization", "")
    """
    # Extract useful information from the request
    model = body.get("model", "unknown")
    messages = body.get("messages", [])

    # Find the last user message
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                # Handle structured content (e.g., Anthropic format)
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        user_message = item.get("text", "")
                        break
            break

    # Example: Route based on model name
    if "gpt-4" in model:
        return f"This is a GPT-4 style response to: {user_message}"
    elif "claude" in model.lower():
        return f"This is a Claude style response to: {user_message}"

    # Example: Check for specific keywords in the user message
    user_message_lower = user_message.lower()
    if "hello" in user_message_lower:
        return "Hello! How can I help you today?"
    if "weather" in user_message_lower:
        return "I'm a mock LLM, so I can't check actual weather. Let's say sunny!"
    if "code" in user_message_lower or "function" in user_message_lower:
        return """Here's an example function:

```python
def example():
    return "Hello, World!"
```"""

    # Example: Access headers for authentication-based routing
    auth_header = headers.get("authorization", "")
    if auth_header.startswith("Bearer premium-"):
        return f"Premium response for: {user_message}"

    # Default response
    return f"Mock response for model '{model}': {user_message}"


# Example: Override usage (uncomment to use)
# def get_response(headers: Dict[str, Any], body: Dict[str, Any]):
#     response = "Hello from the module!"
#     usage = {
#         "prompt_tokens": 5,
#         "completion_tokens": 7,
#         "total_tokens": 12,
#         "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
#     }
#     return (response, usage)
