"""Inference dispatcher: cloud API by default, local GPU optional."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

DEFAULT_MODEL_ID = "google/medgemma-4b-it"


def resolve_inference_mode(mode: str | None = None) -> str:
    value = (mode or os.getenv("INFERENCE_MODE") or "api").strip().lower()
    if value in {"api", "cloud"}:
        return "api"
    if value == "local":
        return "local"
    raise ValueError("INFERENCE_MODE must be 'api' or 'local'.")


def resolve_provider(provider: str | None = None) -> str:
    value = (provider or os.getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if value:
        return value
    api_url = (os.getenv("MEDGEMMA_API_URL") or "").lower()
    if api_url:
        return "modal" if "modal.run" in api_url else "openai_compatible"
    if os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        return "openai"
    return "modal"


def generate_report(
    image: Image.Image,
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    hf_token: str | None = None,
    inference_mode: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> str:
    """Draft a radiological finding report from a PIL image and text prompt."""
    mode = resolve_inference_mode(inference_mode)
    if mode == "local":
        from inference_local import generate_report_local

        return generate_report_local(
            image,
            prompt,
            model_id=model_id,
            hf_token=hf_token,
        )

    from inference_api import generate_report_api

    resolved_provider = resolve_provider(provider)
    api_model = None
    if resolved_provider in {"openai_compatible", "medgemma", "modal"}:
        api_model = model_id
    elif resolved_provider == "openai" and model_id != DEFAULT_MODEL_ID:
        api_model = model_id

    return generate_report_api(
        image,
        prompt,
        provider=resolved_provider,
        api_key=api_key,
        api_url=api_url,
        model_id=api_model,
    )
