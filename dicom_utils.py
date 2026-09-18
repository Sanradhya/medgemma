"""DICOM pixel extraction with PHI stripped before model input."""

from __future__ import annotations

from typing import BinaryIO

import numpy as np
import pydicom
from PIL import Image

BytesLike = BinaryIO | str


def dicom_to_pil_image(source: BytesLike) -> Image.Image:
    """Convert a DICOM file to an RGB PIL image using pixel data only.

    All DICOM metadata (patient identifiers, study tags, windowing, etc.) is
    ignored so Protected Health Information never reaches the LLM.
    """
    if hasattr(source, "seek"):
        source.seek(0)

    dataset = pydicom.dcmread(source, force=True)
    pixel_array = np.asarray(dataset.pixel_array)
    del dataset

    return _normalize_pixels_to_rgb(pixel_array)


def _normalize_pixels_to_rgb(pixel_array: np.ndarray) -> Image.Image:
    """Min-max normalize pixel values to 8-bit and return an RGB PIL image."""
    arr = np.asarray(pixel_array)

    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[-1] not in (3, 4) and arr.shape[0] not in (3, 4):
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    arr = arr.astype(np.float32, copy=False)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin) * 255.0
    else:
        arr = np.zeros_like(arr)

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")

    return Image.fromarray(arr[..., :3], mode="RGB")
