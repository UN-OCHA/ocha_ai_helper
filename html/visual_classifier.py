"""
CPU-only ONNX visual classifier (Map / Infographic / Other).

Preprocessing matches the training/export pipeline in rw-visual-classifier
(letterbox 224, ImageNet normalize) without depending on that package or torch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

CLASS_NAMES = ("Map", "Infographic", "Other")
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)


def letterbox(
    image: Image.Image,
    size: int = IMAGE_SIZE,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize preserving aspect ratio and pad to a square canvas."""
    image = image.convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def preprocess(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Return float32 NCHW batch matching torchvision ToTensor + Normalize.

    Steps: letterbox → HWC uint8 → /255 → CHW → ImageNet normalize → batch.
    """
    boxed = letterbox(image, size=size)
    array = np.asarray(boxed, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))  # HWC -> CHW
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(array, axis=0).astype(np.float32)


def load_session(
    path: Union[str, Path],
    intra_op_threads: int = 2,
) -> ort.InferenceSession:
    """Load an ONNX Runtime CPU session from path."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    # Product analytics only; not required for inference.
    ort.disable_telemetry_events()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_threads
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def predict(session: ort.InferenceSession, image: Image.Image) -> Dict[str, object]:
    """Run classification and return label plus class probabilities."""
    batch = preprocess(image)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch})[0]
    logits = np.asarray(logits)[0]
    # Stable softmax
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    idx = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[idx],
        "probabilities": {
            name: float(prob) for name, prob in zip(CLASS_NAMES, probs)
        },
    }
