"""Provider-specific model clients for MCP tool-call evaluation."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

SUPPORTED_PROVIDERS = {"openai", "anthropic", "gemini", "bedrock"}
PROVIDER_ENV_KEYS = {
    "openai": "API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
DEFAULT_OPENAI_BASE_URL = "https://livai-api.llnl.gov/v1"


@dataclass
class ModelToolCallResult:
    tool_name: str | None
    tool_arguments: dict[str, Any] | None
    tool_arguments_raw: str | None
    assistant_text: str | None
    raw_message: dict[str, Any] | None
    raw_tool_calls: list[dict[str, Any]]
    raw_response: dict[str, Any] | None
    usage: dict[str, int | None]
    latency_ms: int
    error_type: str | None = None
    error: str | None = None


def model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def get_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def get_tool_name(tool: Any) -> str:
    return str(get_value(tool, "name", ""))


def get_tool_description(tool: Any, server_name: str) -> str:
    return f"[{server_name}] {get_value(tool, 'description', '') or ''}"


def get_tool_schema(tool: Any) -> dict[str, Any]:
    schema = get_value(tool, "inputSchema", None)
    if schema is None:
        schema = get_value(tool, "input_schema", None)
    return schema if isinstance(schema, dict) else {"type": "object", "properties": {}}


def tool_to_openai_format(tool: Any, server_name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": get_tool_name(tool),
            "description": get_tool_description(tool, server_name),
            "parameters": get_tool_schema(tool),
        },
    }


def tool_to_anthropic_format(tool: Any, server_name: str) -> dict[str, Any]:
    return {
        "name": get_tool_name(tool),
        "description": get_tool_description(tool, server_name),
        "input_schema": get_tool_schema(tool),
    }


def tool_to_gemini_format(tool: Any, server_name: str) -> dict[str, Any]:
    return {
        "name": get_tool_name(tool),
        "description": get_tool_description(tool, server_name),
        "parameters": normalize_json_schema_for_gemini(get_tool_schema(tool)),
    }


def tool_to_bedrock_format(tool: Any, server_name: str) -> dict[str, Any]:
    return {
        "toolSpec": {
            "name": get_tool_name(tool),
            "description": get_tool_description(tool, server_name),
            "inputSchema": {"json": get_tool_schema(tool)},
        }
    }


def normalize_json_schema_for_gemini(value: Any) -> Any:
    """Drop JSON Schema keywords that Gemini function declarations commonly reject."""
    if isinstance(value, dict):
        return {
            key: normalize_json_schema_for_gemini(item)
            for key, item in value.items()
            if key not in {"$schema", "$id", "additionalProperties"}
        }
    if isinstance(value, list):
        return [normalize_json_schema_for_gemini(item) for item in value]
    return value


def format_tools(provider: str, tools: list[Any], server_name: str) -> list[dict[str, Any]]:
    if provider == "openai":
        return [tool_to_openai_format(tool, server_name) for tool in tools]
    if provider == "anthropic":
        return [tool_to_anthropic_format(tool, server_name) for tool in tools]
    if provider == "gemini":
        return [tool_to_gemini_format(tool, server_name) for tool in tools]
    if provider == "bedrock":
        return [tool_to_bedrock_format(tool, server_name) for tool in tools]
    raise ValueError(f"Unsupported provider: {provider}")


def normalize_usage(prompt_tokens: Any, completion_tokens: Any, total_tokens: Any = None) -> dict[str, int | None]:
    prompt = prompt_tokens if isinstance(prompt_tokens, int) else None
    completion = completion_tokens if isinstance(completion_tokens, int) else None
    total = total_tokens if isinstance(total_tokens, int) else None
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def openai_usage_dict(usage: Any) -> dict[str, int | None]:
    return normalize_usage(
        get_value(usage, "prompt_tokens"),
        get_value(usage, "completion_tokens"),
        get_value(usage, "total_tokens"),
    )


def anthropic_usage_dict(usage: Any) -> dict[str, int | None]:
    return normalize_usage(get_value(usage, "input_tokens"), get_value(usage, "output_tokens"))


def gemini_usage_dict(usage: Any) -> dict[str, int | None]:
    return normalize_usage(
        get_value(usage, "prompt_token_count"),
        get_value(usage, "candidates_token_count"),
        get_value(usage, "total_token_count"),
    )


def bedrock_usage_dict(usage: Any) -> dict[str, int | None]:
    return normalize_usage(
        get_value(usage, "inputTokens"),
        get_value(usage, "outputTokens"),
        get_value(usage, "totalTokens"),
    )


def empty_usage() -> dict[str, int | None]:
    return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def error_result(started: float, error_type: str, error: str) -> ModelToolCallResult:
    return ModelToolCallResult(
        tool_name=None,
        tool_arguments=None,
        tool_arguments_raw=None,
        assistant_text=None,
        raw_message=None,
        raw_tool_calls=[],
        raw_response=None,
        usage=empty_usage(),
        latency_ms=round((time.perf_counter() - started) * 1000),
        error_type=error_type,
        error=error,
    )


def parse_json_arguments(value: Any) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if isinstance(value, dict):
        return value, json.dumps(value, sort_keys=True), None
    raw = value if isinstance(value, str) else "{}"
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return None, raw, f"tool arguments are not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, raw, "tool arguments JSON must decode to an object"
    return parsed, raw, None


class BaseModelClient:
    provider = ""

    async def complete_tool_call(
        self,
        model: str,
        tools: list[Any],
        server_name: str,
        prompt: str,
        system_prompt: str,
        temperature: float | None,
        capture_raw_response: bool = False,
    ) -> ModelToolCallResult:
        raise NotImplementedError


class OpenAIModelClient(BaseModelClient):
    provider = "openai"

    def __init__(self, api_key: str, base_url: str, timeout: float) -> None:
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("openai package is required for OpenAI-compatible benchmark runs") from exc
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def complete_tool_call(
        self,
        model: str,
        tools: list[Any],
        server_name: str,
        prompt: str,
        system_prompt: str,
        temperature: float | None,
        capture_raw_response: bool = False,
    ) -> ModelToolCallResult:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "tools": format_tools(self.provider, tools, server_name),
            "tool_choice": "auto",
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            return error_result(started, "api_error", f"{type(exc).__name__}: {exc}")

        latency_ms = round((time.perf_counter() - started) * 1000)
        message = response.choices[0].message
        raw_message = model_dump(message)
        raw_response = model_dump(response) if capture_raw_response else None
        tool_calls = message.tool_calls or []
        raw_tool_calls = [model_dump(tool_call) for tool_call in tool_calls]
        if not tool_calls:
            return ModelToolCallResult(
                tool_name=None,
                tool_arguments=None,
                tool_arguments_raw=None,
                assistant_text=message.content,
                raw_message=raw_message,
                raw_tool_calls=raw_tool_calls,
                raw_response=raw_response,
                usage=openai_usage_dict(response.usage),
                latency_ms=latency_ms,
                error_type="no_tool_call",
                error="model returned no tool call",
            )

        call = tool_calls[0]
        arguments, raw_arguments, parse_error = parse_json_arguments(call.function.arguments or "{}")
        if parse_error:
            return ModelToolCallResult(
                tool_name=call.function.name,
                tool_arguments=None,
                tool_arguments_raw=raw_arguments,
                assistant_text=message.content,
                raw_message=raw_message,
                raw_tool_calls=raw_tool_calls,
                raw_response=raw_response,
                usage=openai_usage_dict(response.usage),
                latency_ms=latency_ms,
                error_type="bad_json",
                error=parse_error,
            )

        return ModelToolCallResult(
            tool_name=call.function.name,
            tool_arguments=arguments,
            tool_arguments_raw=raw_arguments,
            assistant_text=message.content,
            raw_message=raw_message,
            raw_tool_calls=raw_tool_calls,
            raw_response=raw_response,
            usage=openai_usage_dict(response.usage),
            latency_ms=latency_ms,
        )


class AnthropicModelClient(BaseModelClient):
    provider = "anthropic"

    def __init__(self, api_key: str, base_url: str | None, timeout: float) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("anthropic package is required for Anthropic benchmark runs") from exc
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncAnthropic(**kwargs)

    async def complete_tool_call(
        self,
        model: str,
        tools: list[Any],
        server_name: str,
        prompt: str,
        system_prompt: str,
        temperature: float | None,
        capture_raw_response: bool = False,
    ) -> ModelToolCallResult:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "tools": format_tools(self.provider, tools, server_name),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            response = await self.client.messages.create(**kwargs)
        except Exception as exc:
            return error_result(started, "api_error", f"{type(exc).__name__}: {exc}")

        latency_ms = round((time.perf_counter() - started) * 1000)
        content = list(get_value(response, "content", []) or [])
        raw_response = model_dump(response) if capture_raw_response else None
        raw_message = {"content": [model_dump(item) for item in content]}
        raw_tool_calls = [model_dump(item) for item in content if get_value(item, "type") == "tool_use"]
        text_parts = [get_value(item, "text") for item in content if get_value(item, "type") == "text"]
        assistant_text = "\n".join(part for part in text_parts if isinstance(part, str)) or None
        if not raw_tool_calls:
            return ModelToolCallResult(
                tool_name=None,
                tool_arguments=None,
                tool_arguments_raw=None,
                assistant_text=assistant_text,
                raw_message=raw_message,
                raw_tool_calls=[],
                raw_response=raw_response,
                usage=anthropic_usage_dict(get_value(response, "usage")),
                latency_ms=latency_ms,
                error_type="no_tool_call",
                error="model returned no tool call",
            )

        call = raw_tool_calls[0]
        arguments, raw_arguments, parse_error = parse_json_arguments(call.get("input", {}))
        if parse_error:
            return ModelToolCallResult(
                tool_name=call.get("name"),
                tool_arguments=None,
                tool_arguments_raw=raw_arguments,
                assistant_text=assistant_text,
                raw_message=raw_message,
                raw_tool_calls=raw_tool_calls,
                raw_response=raw_response,
                usage=anthropic_usage_dict(get_value(response, "usage")),
                latency_ms=latency_ms,
                error_type="bad_json",
                error=parse_error,
            )

        return ModelToolCallResult(
            tool_name=call.get("name"),
            tool_arguments=arguments,
            tool_arguments_raw=raw_arguments,
            assistant_text=assistant_text,
            raw_message=raw_message,
            raw_tool_calls=raw_tool_calls,
            raw_response=raw_response,
            usage=anthropic_usage_dict(get_value(response, "usage")),
            latency_ms=latency_ms,
        )


class GeminiModelClient(BaseModelClient):
    provider = "gemini"

    def __init__(self, api_key: str, timeout: float) -> None:
        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("google-genai package is required for Gemini benchmark runs") from exc
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.timeout = timeout

    async def complete_tool_call(
        self,
        model: str,
        tools: list[Any],
        server_name: str,
        prompt: str,
        system_prompt: str,
        temperature: float | None,
        capture_raw_response: bool = False,
    ) -> ModelToolCallResult:
        started = time.perf_counter()
        declarations = format_tools(self.provider, tools, server_name)
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "tools": [self.types.Tool(function_declarations=declarations)],
        }
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=self.types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            return error_result(started, "api_error", f"{type(exc).__name__}: {exc}")

        latency_ms = round((time.perf_counter() - started) * 1000)
        raw_response = model_dump(response) if capture_raw_response else None
        candidate = (get_value(response, "candidates", []) or [None])[0]
        content = get_value(candidate, "content")
        parts = list(get_value(content, "parts", []) or [])
        raw_message = {"parts": [model_dump(part) for part in parts]}
        text_parts = [get_value(part, "text") for part in parts if get_value(part, "text") is not None]
        assistant_text = "\n".join(str(part) for part in text_parts) or None
        function_calls = [get_value(part, "function_call") for part in parts if get_value(part, "function_call")]
        raw_tool_calls = [model_dump(call) for call in function_calls]
        usage = gemini_usage_dict(get_value(response, "usage_metadata"))
        if not function_calls:
            return ModelToolCallResult(
                tool_name=None,
                tool_arguments=None,
                tool_arguments_raw=None,
                assistant_text=assistant_text,
                raw_message=raw_message,
                raw_tool_calls=raw_tool_calls,
                raw_response=raw_response,
                usage=usage,
                latency_ms=latency_ms,
                error_type="no_tool_call",
                error="model returned no tool call",
            )

        call = function_calls[0]
        arguments, raw_arguments, parse_error = parse_json_arguments(get_value(call, "args", {}))
        if parse_error:
            return ModelToolCallResult(
                tool_name=get_value(call, "name"),
                tool_arguments=None,
                tool_arguments_raw=raw_arguments,
                assistant_text=assistant_text,
                raw_message=raw_message,
                raw_tool_calls=raw_tool_calls,
                raw_response=raw_response,
                usage=usage,
                latency_ms=latency_ms,
                error_type="bad_json",
                error=parse_error,
            )

        return ModelToolCallResult(
            tool_name=get_value(call, "name"),
            tool_arguments=arguments,
            tool_arguments_raw=raw_arguments,
            assistant_text=assistant_text,
            raw_message=raw_message,
            raw_tool_calls=raw_tool_calls,
            raw_response=raw_response,
            usage=usage,
            latency_ms=latency_ms,
        )


class BedrockModelClient(BaseModelClient):
    provider = "bedrock"

    def __init__(self, region_name: str | None, profile_name: str | None) -> None:
        try:
            import boto3
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("boto3 package is required for Bedrock benchmark runs") from exc
        session_kwargs: dict[str, Any] = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        session = boto3.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {}
        if region_name:
            client_kwargs["region_name"] = region_name
        self.client = session.client("bedrock-runtime", **client_kwargs)

    async def complete_tool_call(
        self,
        model: str,
        tools: list[Any],
        server_name: str,
        prompt: str,
        system_prompt: str,
        temperature: float | None,
        capture_raw_response: bool = False,
    ) -> ModelToolCallResult:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "modelId": model,
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "toolConfig": {"tools": format_tools(self.provider, tools, server_name)},
        }
        if temperature is not None:
            kwargs["inferenceConfig"] = {"temperature": temperature}

        try:
            response = await asyncio.to_thread(self.client.converse, **kwargs)
        except Exception as exc:
            return error_result(started, "api_error", f"{type(exc).__name__}: {exc}")

        latency_ms = round((time.perf_counter() - started) * 1000)
        message = get_value(get_value(response, "output", {}), "message", {})
        content = get_value(message, "content", []) or []
        raw_message = model_dump(message)
        raw_response = model_dump(response) if capture_raw_response else None
        raw_tool_calls = [
            model_dump(item["toolUse"]) for item in content if isinstance(item, dict) and "toolUse" in item
        ]
        text_parts = [
            item.get("text") for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        assistant_text = "\n".join(text_parts) or None
        usage = bedrock_usage_dict(get_value(response, "usage"))
        if not raw_tool_calls:
            return ModelToolCallResult(
                tool_name=None,
                tool_arguments=None,
                tool_arguments_raw=None,
                assistant_text=assistant_text,
                raw_message=raw_message,
                raw_tool_calls=raw_tool_calls,
                raw_response=raw_response,
                usage=usage,
                latency_ms=latency_ms,
                error_type="no_tool_call",
                error="model returned no tool call",
            )

        call = raw_tool_calls[0]
        arguments, raw_arguments, parse_error = parse_json_arguments(call.get("input", {}))
        if parse_error:
            return ModelToolCallResult(
                tool_name=call.get("name"),
                tool_arguments=None,
                tool_arguments_raw=raw_arguments,
                assistant_text=assistant_text,
                raw_message=raw_message,
                raw_tool_calls=raw_tool_calls,
                raw_response=raw_response,
                usage=usage,
                latency_ms=latency_ms,
                error_type="bad_json",
                error=parse_error,
            )

        return ModelToolCallResult(
            tool_name=call.get("name"),
            tool_arguments=arguments,
            tool_arguments_raw=raw_arguments,
            assistant_text=assistant_text,
            raw_message=raw_message,
            raw_tool_calls=raw_tool_calls,
            raw_response=raw_response,
            usage=usage,
            latency_ms=latency_ms,
        )


def provider_and_model(model: str, default_provider: str) -> tuple[str, str]:
    prefix, separator, remainder = model.partition(":")
    if separator and prefix in SUPPORTED_PROVIDERS and remainder:
        return prefix, remainder
    return default_provider, model


def provider_config(config: dict[str, Any], provider: str) -> dict[str, Any]:
    providers = config.get("providers", {})
    if isinstance(providers, dict) and isinstance(providers.get(provider), dict):
        return providers[provider]
    return {}


class ModelClientRegistry:
    def __init__(
        self,
        *,
        default_provider: str,
        config: dict[str, Any],
        timeout: float,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if default_provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported default provider {default_provider!r}")
        self.default_provider = default_provider
        self.config = config
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url
        self.clients: dict[str, BaseModelClient] = {}

    def split_model(self, model: str) -> tuple[str, str]:
        return provider_and_model(model, self.default_provider)

    def client_for_model(self, model: str) -> tuple[BaseModelClient, str]:
        provider, provider_model = self.split_model(model)
        if provider not in self.clients:
            self.clients[provider] = self.build_client(provider)
        return self.clients[provider], provider_model

    def build_client(self, provider: str) -> BaseModelClient:
        config = provider_config(self.config, provider)
        if provider == "openai":
            base_url = (
                self.base_url or string_config(config, "base_url") or os.getenv("API_BASE_URL", DEFAULT_OPENAI_BASE_URL)
            )
            api_key = self.api_key or string_config(config, "api_key") or os.getenv("API_KEY")
            if not api_key:
                api_key = "dummy" if base_url.startswith(("http://localhost", "http://127.0.0.1")) else None
            if not api_key:
                raise ValueError("OpenAI-compatible API key is required. Pass --api-key or set API_KEY.")
            return OpenAIModelClient(api_key=api_key, base_url=base_url, timeout=self.timeout)
        if provider == "anthropic":
            api_key = (
                self.provider_api_key(provider) or string_config(config, "api_key") or os.getenv("ANTHROPIC_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "Anthropic API key is required. Set ANTHROPIC_API_KEY or model_api.providers.anthropic.api_key."
                )
            return AnthropicModelClient(
                api_key=api_key,
                base_url=self.provider_base_url(provider) or string_config(config, "base_url"),
                timeout=self.timeout,
            )
        if provider == "gemini":
            api_key = (
                self.provider_api_key(provider)
                or string_config(config, "api_key")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "Gemini API key is required. Set GEMINI_API_KEY or model_api.providers.gemini.api_key."
                )
            return GeminiModelClient(api_key=api_key, timeout=self.timeout)
        if provider == "bedrock":
            return BedrockModelClient(
                region_name=(
                    string_config(config, "region_name") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
                ),
                profile_name=string_config(config, "profile_name") or os.getenv("AWS_PROFILE"),
            )
        raise ValueError(f"Unsupported provider: {provider}")

    def provider_api_key(self, provider: str) -> str | None:
        return self.api_key if provider == self.default_provider else None

    def provider_base_url(self, provider: str) -> str | None:
        return self.base_url if provider == self.default_provider else None


def string_config(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    return expand_env_vars(value) if isinstance(value, str) and value else None


def expand_env_vars(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        expression = match.group(1)
        if ":-" in expression:
            name, default = expression.split(":-", 1)
            return os.getenv(name.strip(), default.strip())
        resolved = os.getenv(expression.strip())
        if resolved is None:
            raise ValueError(f"Environment variable {expression} is not set")
        return resolved

    return re.sub(r"\$\{([^}]+)\}", replace, value)
