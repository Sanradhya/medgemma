"""MedGemma on Modal: GPU runs only during Analyze, then scales to zero.

After `pip install modal` and `modal setup`:

    modal secret create huggingface HF_TOKEN=hf_your_token
    modal deploy modal_app.py

Copy the printed `*.modal.run` URL into `.env` as MEDGEMMA_API_URL
(no trailing path needed). Optional shared password:

    Set MEDGEMMA_API_KEY in the huggingface secret or a second secret,
    and use the same value in the laptop `.env`.

Live reload while developing:

    modal serve modal_app.py
"""

from __future__ import annotations

import base64
import io
import os

import modal

MODEL_ID = "google/medgemma-4b-it"
HF_CACHE_PATH = "/root/.cache/huggingface"
SCALEDOWN_SECONDS = 120
TIMEOUT_SECONDS = 15 * 60
STARTUP_TIMEOUT_SECONDS = 15 * 60
MAX_NEW_TOKENS = 1024
MAX_IMAGE_EDGE = 1280

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "pillow",
        "fastapi[standard]",
        "huggingface_hub",
        "pydantic",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

hf_cache_vol = modal.Volume.from_name("medgemma-hf-cache", create_if_missing=True)

app = modal.App("medgemma-analyzer", image=image)


def _decode_image(image_base64: str):
    from PIL import Image

    payload = (image_base64 or "").strip()
    if not payload:
        raise ValueError("image_base64 is required")
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / float(longest)
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def _check_api_key(authorization: str | None) -> None:
    expected = (os.getenv("MEDGEMMA_API_KEY") or "").strip()
    if not expected:
        return
    incoming = (authorization or "").strip()
    if incoming.lower().startswith("bearer "):
        incoming = incoming[7:].strip()
    if incoming != expected:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.cls(
    gpu=["L4", "A10G", "T4"],
    timeout=TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_SECONDS,
    min_containers=0,
    memory=16384,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
class MedGemma:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoProcessor

        os.environ.setdefault("HF_HOME", HF_CACHE_PATH)
        token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
        if not token:
            raise RuntimeError("HF_TOKEN is missing. Create Modal secret `huggingface`.")

        auth = {"token": token}
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, **auth)
        try:
            from transformers import AutoModelForImageTextToText as ModelCls
        except ImportError:
            from transformers import AutoModelForCausalLM as ModelCls

        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = ModelCls.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            device_map="auto",
            **auth,
        )
        self.model.eval()
        hf_cache_vol.commit()

    def generate_report(self, prompt: str, image_base64: str, max_new_tokens: int) -> str:
        import torch

        image = _decode_image(image_base64)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        inputs = inputs.to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        input_len = inputs["input_ids"].shape[-1]
        token_budget = max(32, min(int(max_new_tokens), 2048))
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=token_budget,
                do_sample=False,
            )
        generated = output_ids[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    @modal.asgi_app()
    def api(self):
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel, Field

        web = FastAPI(title="MedGemma Analyzer", docs_url="/docs")

        class AnalyzeRequest(BaseModel):
            prompt: str
            image_base64: str
            max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=32, le=2048)

        @web.post("/analyze")
        def analyze(
            payload: AnalyzeRequest,
            authorization: str | None = Header(default=None),
        ):
            _check_api_key(authorization)
            if not payload.prompt.strip():
                raise HTTPException(status_code=400, detail="prompt is required")
            report = self.generate_report(
                payload.prompt.strip(),
                payload.image_base64,
                payload.max_new_tokens,
            )
            if not report:
                raise HTTPException(status_code=502, detail="Model returned an empty report")
            return {"report": report, "model": MODEL_ID}

        return web
