"""Local GPU inference. Requires torch/transformers and enough VRAM."""

from __future__ import annotations

import os

import streamlit as st
from PIL import Image

from image_prep import prepare_rgb_image

DEFAULT_MODEL_ID = "google/medgemma-4b-it"


def _resolve_hf_token(hf_token: str | None = None) -> str | None:
    token = (hf_token or "").strip()
    if token:
        return token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None


@st.cache_resource
def load_model(model_id: str = DEFAULT_MODEL_ID, hf_token: str | None = None):
    """Load and cache MedGemma. Intended for machines with a GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    token = _resolve_hf_token(hf_token)
    auth = {"token": token} if token else {}

    processor = AutoProcessor.from_pretrained(model_id, **auth)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        **auth,
    )
    model.eval()
    return processor, model


def generate_report_local(
    image: Image.Image,
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    hf_token: str | None = None,
) -> str:
    import torch

    processor, model = load_model(model_id, hf_token=hf_token)
    prepared = prepare_rgb_image(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": prepared},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = inputs.to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
        )

    generated_tokens = output_ids[0][input_len:]
    return processor.decode(generated_tokens, skip_special_tokens=True).strip()
