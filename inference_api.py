"""Cloud inference: no model weights on the laptop."""

from __future__ import annotations

import os

import httpx
from PIL import Image

from image_prep import image_to_data_url, image_to_jpeg_bytes

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_MEDGEMMA_MODEL = "google/medgemma-4b-it"
MAX_NEW_TOKENS = 1024
TIMEOUT_SECONDS = 180.0
MODAL_TIMEOUT_SECONDS = 600.0

SYSTEM_PROMPT = (
    "You are an expert radiologist drafting a preliminary finding report. "
    "Describe only what is visible. Flag uncertainty. This is not a signed "
    "diagnostic report and must be reviewed by a qualified clinician."
)


def generate_report_api(
    image: Image.Image,
    prompt: str,
    provider: str,
    api_key: str | None = None,
    api_url: str | None = None,
    model_id: str | None = None,
) -> str:
    provider = (provider or "").strip().lower()
    if provider == "modal":
        return _generate_modal(image, prompt, api_key=api_key, api_url=api_url)
    if provider == "gemini":
        return _generate_gemini(image, prompt, api_key=api_key, model_id=model_id)
    if provider in {"openai", "openai_compatible", "medgemma"}:
        return _generate_openai_compatible(
            image,
            prompt,
            provider=provider,
            api_key=api_key,
            api_url=api_url,
            model_id=model_id,
        )
    raise ValueError(
        f"Unknown cloud provider '{provider}'. Use modal, gemini, openai, or openai_compatible."
    )


def _user_text(prompt: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{prompt.strip()}"


def _generate_modal(
    image: Image.Image,
    prompt: str,
    api_key: str | None = None,
    api_url: str | None = None,
) -> str:
    url = (api_url or os.getenv("MEDGEMMA_API_URL") or "").strip().rstrip("/")
    if not url:
        raise PermissionError(
            "Set MEDGEMMA_API_URL to your Modal endpoint "
            "(from `modal deploy modal_app.py`)."
        )
    if url.endswith("/analyze"):
        analyze_url = url
    else:
        analyze_url = f"{url}/analyze"

    token = (api_key or os.getenv("MEDGEMMA_API_KEY") or "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "prompt": _user_text(prompt),
        "image_base64": image_to_data_url(image),
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    with httpx.Client(timeout=MODAL_TIMEOUT_SECONDS) as client:
        response = client.post(analyze_url, headers=headers, json=payload)

    if response.status_code in {401, 403}:
        raise PermissionError(
            "Modal rejected the API key. Set MEDGEMMA_API_KEY to match the server."
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Modal MedGemma returned HTTP {response.status_code}: {response.text[:500]}"
        ) from exc

    data = response.json()
    text = data.get("report") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Unexpected Modal response: {data!r}")
    return text.strip()


def _generate_gemini(
    image: Image.Image,
    prompt: str,
    api_key: str | None = None,
    model_id: str | None = None,
) -> str:
    from google import genai
    from google.genai import types

    token = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not token:
        raise PermissionError(
            "Set GEMINI_API_KEY (Google AI Studio) to analyze images in cloud mode."
        )

    model = (model_id or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    client = genai.Client(api_key=token)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        data=image_to_jpeg_bytes(image),
                        mime_type="image/jpeg",
                    ),
                    types.Part.from_text(text=_user_text(prompt)),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
        ),
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("The Gemini API returned an empty response.")
    return text


def _generate_openai_compatible(
    image: Image.Image,
    prompt: str,
    provider: str,
    api_key: str | None = None,
    api_url: str | None = None,
    model_id: str | None = None,
) -> str:
    url = (
        api_url
        or os.getenv("MEDGEMMA_API_URL")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    ).strip()
    token = (
        api_key
        or os.getenv("MEDGEMMA_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()

    if provider == "openai":
        url = url or "https://api.openai.com/v1/chat/completions"
        model = (model_id or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip()
        if not token:
            raise PermissionError("Set OPENAI_API_KEY to use OpenAI vision.")
    else:
        if not url:
            raise PermissionError(
                "Set MEDGEMMA_API_URL to your hosted MedGemma chat-completions endpoint "
                "(Vertex AI, vLLM, or TGI)."
            )
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"
        model = (
            model_id
            or os.getenv("MEDGEMMA_MODEL")
            or DEFAULT_MEDGEMMA_MODEL
        ).strip()

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "model": model,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _user_text(prompt)},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image)},
                    },
                ],
            }
        ],
    }

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        response = client.post(url, headers=headers, json=payload)

    if response.status_code in {401, 403}:
        raise PermissionError(
            "The analysis API rejected the request (unauthorized). Check the API key "
            "and endpoint URL."
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text[:500]
        raise RuntimeError(
            f"Analysis API returned HTTP {response.status_code}: {detail}"
        ) from exc

    data = response.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected analysis API response: {data!r}") from exc
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("The analysis API returned an empty report.")
    return text.strip()
