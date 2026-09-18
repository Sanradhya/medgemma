"""Prepare images for local or cloud inference."""

from __future__ import annotations

import base64
import io

from PIL import Image

MAX_EDGE = 1280
JPEG_QUALITY = 90


def prepare_rgb_image(image: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    """Return an RGB copy, shrinking the long edge so API payloads stay small."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    longest = max(width, height)
    if longest <= max_edge:
        return rgb
    scale = max_edge / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return rgb.resize(new_size, Image.Resampling.LANCZOS)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    prepared = prepare_rgb_image(image)
    buffer = io.BytesIO()
    prepared.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def image_to_data_url(image: Image.Image) -> str:
    encoded = base64.b64encode(image_to_jpeg_bytes(image)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
