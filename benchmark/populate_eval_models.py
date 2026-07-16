#!/usr/bin/env python3
"""Discover available LLM model IDs and update eval model list files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from eval_io import load_model_levels, write_model_level_file  # noqa: E402

from mada_tools.shared.config import get_config_value, load_json_object_config  # noqa: E402

DEFAULT_BASE_URL = "https://livai-api.llnl.gov/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT = 30.0
DEFAULT_ALL_OUTPUT = SCRIPT_DIR / "eval_models_all.tsv"
DEFAULT_ENABLED_OUTPUT = SCRIPT_DIR / "eval_models.tsv"
PROVIDERS = {"openai", "anthropic", "gemini", "bedrock"}


def resolve_api_settings(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve API key and base URL with the same precedence as the eval script."""

    config = load_json_object_config(args.config)
    base_url = args.base_url or get_config_value(config, "base_url") or os.getenv("API_BASE_URL", DEFAULT_BASE_URL)
    api_key = args.api_key or get_config_value(config, "api_key") or os.getenv("API_KEY")

    if not api_key:
        api_key = "dummy" if base_url.startswith(("http://localhost", "http://127.0.0.1")) else None
    if not api_key:
        raise ValueError("API key is required. Provide --api-key, --config, or set API_KEY.")

    return api_key, base_url


def resolve_provider_api_settings(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Resolve API settings for providers that use HTTP API keys."""

    if args.provider == "openai":
        api_key, base_url = resolve_api_settings(args)
        return api_key, base_url

    config = load_json_object_config(args.config)
    providers = config.get("providers", {})
    provider_config = providers.get(args.provider, {}) if isinstance(providers, dict) else {}
    if not isinstance(provider_config, dict):
        provider_config = {}

    if args.provider == "anthropic":
        base_url = args.base_url or provider_config.get("base_url") or DEFAULT_ANTHROPIC_BASE_URL
        api_key = args.api_key or provider_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is required. Provide --api-key or set ANTHROPIC_API_KEY.")
        return str(api_key), str(base_url)

    if args.provider == "gemini":
        base_url = args.base_url or provider_config.get("base_url") or DEFAULT_GEMINI_BASE_URL
        api_key = (
            args.api_key or provider_config.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("Gemini API key is required. Provide --api-key or set GEMINI_API_KEY.")
        return str(api_key), str(base_url)

    if args.provider == "bedrock":
        return None, None

    raise ValueError(f"Unsupported provider: {args.provider}")


def extract_model_ids(payload: Any) -> list[str]:
    """Extract, dedupe, and sort model IDs from a /models response payload."""

    entries = payload
    if isinstance(payload, dict):
        entries = payload.get("data")

    if not isinstance(entries, list):
        raise ValueError("Models response must be a JSON object with a 'data' list or a list of model objects")

    model_ids = sorted(
        {
            str(entry["id"])
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"]
        }
    )
    if not model_ids:
        raise ValueError("Models response did not contain any model IDs")
    return model_ids


def fetch_models_payload(base_url: str, api_key: str, timeout: float) -> Any:
    """Fetch the raw payload from an OpenAI-compatible /models endpoint."""

    models_url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(
        models_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = f"HTTP {exc.code} from {models_url}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {models_url}: {exc.reason}") from exc

    return payload


def fetch_anthropic_models_payload(base_url: str, api_key: str, timeout: float) -> Any:
    models_url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(
        models_url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = f"HTTP {exc.code} from {models_url}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {models_url}: {exc.reason}") from exc


def fetch_gemini_models_payload(base_url: str, api_key: str, timeout: float) -> Any:
    models_url = f"{base_url.rstrip('/')}/models?{urllib.parse.urlencode({'key': api_key})}"
    request = urllib.request.Request(models_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = f"HTTP {exc.code} from {base_url.rstrip('/')}/models"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {base_url.rstrip('/')}/models: {exc.reason}") from exc


def fetch_bedrock_model_ids() -> list[str]:
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("boto3 package is required for Bedrock model discovery") from exc

    session = boto3.Session()
    client = session.client("bedrock")
    response = client.list_foundation_models()
    summaries = response.get("modelSummaries", [])
    return sorted(
        {
            str(summary["modelId"])
            for summary in summaries
            if isinstance(summary, dict)
            and isinstance(summary.get("modelId"), str)
            and "TEXT" in summary.get("outputModalities", [])
        }
    )


def extract_gemini_model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Gemini models response must be a JSON object with a 'models' list")
    model_ids = []
    for entry in payload["models"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        name = entry["name"]
        supported_methods = entry.get("supportedGenerationMethods", [])
        if supported_methods and "generateContent" not in supported_methods:
            continue
        model_ids.append(name.removeprefix("models/"))
    model_ids = sorted(set(model_ids))
    if not model_ids:
        raise ValueError("Gemini models response did not contain any usable model IDs")
    return model_ids


def discover_model_ids(args: argparse.Namespace) -> list[str]:
    api_key, base_url = resolve_provider_api_settings(args)
    if args.provider == "openai":
        assert api_key is not None and base_url is not None
        return extract_model_ids(fetch_models_payload(base_url, api_key, args.timeout))
    if args.provider == "anthropic":
        assert api_key is not None and base_url is not None
        return extract_model_ids(fetch_anthropic_models_payload(base_url, api_key, args.timeout))
    if args.provider == "gemini":
        assert api_key is not None and base_url is not None
        return extract_gemini_model_ids(fetch_gemini_models_payload(base_url, api_key, args.timeout))
    if args.provider == "bedrock":
        return fetch_bedrock_model_ids()
    raise ValueError(f"Unsupported provider: {args.provider}")


def known_model_levels(all_output: Path, enabled_output: Path) -> dict[str, int]:
    """Prefer curated levels, then fall back to the existing discovery snapshot."""
    if enabled_output.exists():
        return load_model_levels(enabled_output)
    return load_model_levels(all_output)


def refresh_model_files(model_ids: list[str], all_output: Path, enabled_output: Path) -> bool:
    """Refresh the discovery snapshot and initialize the curated list if needed."""

    levels = known_model_levels(all_output, enabled_output)
    write_model_level_file(all_output, model_ids, levels)
    if enabled_output.exists():
        return False
    write_model_level_file(enabled_output, model_ids, levels)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate shared eval model list files from a model provider API.")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="openai",
        help="Provider to query for model IDs (default: openai)",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON config with model.api_key and model.base_url")
    parser.add_argument("--api-key", help="Provider API key")
    parser.add_argument("--base-url", help="Provider API base URL")
    parser.add_argument(
        "--all-output",
        type=Path,
        default=DEFAULT_ALL_OUTPUT,
        help=f"Write discovered model snapshot to this file (default: {DEFAULT_ALL_OUTPUT})",
    )
    parser.add_argument(
        "--enabled-output",
        type=Path,
        default=DEFAULT_ENABLED_OUTPUT,
        help=f"Initialize curated enabled model list here if missing (default: {DEFAULT_ENABLED_OUTPUT})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds when querying /models (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--no-provider-prefix",
        action="store_true",
        help="Do not prefix non-OpenAI model IDs with provider:, useful for provider-specific run configs",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        model_ids = discover_model_ids(args)
        if args.provider != "openai" and not args.no_provider_prefix:
            model_ids = [f"{args.provider}:{model_id}" for model_id in model_ids]
        initialized_enabled = refresh_model_files(model_ids, args.all_output, args.enabled_output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    # Uncomment for Raw printout of model dump
    # print("Raw /models response:")
    # print(json.dumps(payload, indent=2, sort_keys=True))
    print("Extracted model IDs:")
    for model_id in model_ids:
        print(model_id)
    print(f"Wrote {len(model_ids)} discovered models to {args.all_output}")
    if initialized_enabled:
        print(f"Initialized curated enabled model list at {args.enabled_output}")
    else:
        print(f"Left curated enabled model list unchanged at {args.enabled_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
