"""Decode, infer and upload entirely in memory."""

from io import BytesIO
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from utils.storage_utils import upload_image
from .gradcam import render_gradcam

MAX_IMAGE_PIXELS = 20_000_000


class InvalidImage(ValueError):
    pass


def decode_image(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImage("Use a JPEG, PNG or WebP image.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise InvalidImage("Image exceeds the 20 megapixel limit.")
                return np.array(ImageOps.exif_transpose(image).convert("RGB"))
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise InvalidImage("Invalid or oversized image.") from exc


def encode_png(image):
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("Image encoding failed.")
    return encoded.tobytes()


def predict_xray(image_bytes, model_config):
    rgb = decode_image(image_bytes)
    height, width = model_config.get("input_size", (224, 224))
    resized = cv2.resize(rgb, (width, height))
    batch = model_config["preprocess"](resized[None].astype(np.float32))
    with model_config["inference_lock"]:
        prediction, heatmap = model_config["gradcam"](batch)
    label = "Pneumonia" if prediction >= 0.5 else "Normal"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    original = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    has_activation = bool(np.any(heatmap > 0))
    available = prediction >= 0.5 and has_activation
    overlay = render_gradcam(original, heatmap) if available else original
    status = (
        "available"
        if available
        else "not_applicable" if prediction < 0.5 else "no_positive_activation"
    )
    images = upload_image(encode_png(original), encode_png(overlay))
    return {
        "prediction": {
            "label": label,
            "confidence": confidence * 100,
            "probability": {
                "Normal": round(1 - prediction, 4),
                "Pneumonia": round(prediction, 4),
            },
        },
        "images": images,
        "gradcam": {
            "target": "Pneumonia",
            "available": available,
            "status": status,
            "lung_localization_validated": False,
        },
    }
