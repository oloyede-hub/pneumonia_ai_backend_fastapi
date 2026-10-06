from threading import Lock

import numpy as np
from tensorflow import keras
from huggingface_hub import hf_hub_download

from services.gradcam import GradCAM

MODELS = {
    "vgg16": {
        "repo_id": "Buore/VGG16-CHEST-CLASSIFIER",
        "filename": "near_final__vgg16_pneumonia.keras",
        "preprocess": keras.applications.vgg16.preprocess_input,
        "last_conv_layer": "block5_conv3",
        "backbone": "vgg16",
    },
    "resnet50": {
        "repo_id": "Buore/RESNET50-CHEST-CLASSIFIER",
        "filename": "resnet50_shortcut_aware_final.keras",
        # This saved model embeds RGB-to-BGR conversion and mean subtraction.
        "preprocess": np.asarray,
        "last_conv_layer": "conv5_block3_out",
        "backbone": "resnet50",
    },
}

def load_model_config(model_name: str = "resnet50"):
    if model_name not in MODELS:
        raise ValueError(
            f"Unsupported MODEL_NAME {model_name!r}. Choose from: {', '.join(MODELS)}."
        )
    config = MODELS[model_name].copy()
    model_path = hf_hub_download(
        repo_id=config["repo_id"], filename=config["filename"]
    )
    model = keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape
    if (
        isinstance(input_shape, list)
        or len(input_shape) != 4
        or input_shape[-1] != 3
        or any(size is None for size in input_shape[1:3])
    ):
        raise ValueError("Expected one fixed-size RGB image input (batch, height, width, 3).")
    config["input_size"] = tuple(input_shape[1:3])
    gradcam = GradCAM(model, config["backbone"], config["last_conv_layer"])
    # Fail at startup if the explanation graph changes inference behavior.
    for pixels in (
        np.zeros((1, *config["input_size"], 3), dtype=np.float32),
        np.random.default_rng(0).uniform(0, 255, (1, *config["input_size"], 3)).astype(np.float32),
    ):
        probe = config["preprocess"](pixels)
        probability, _ = gradcam(probe)
        np.testing.assert_allclose(
            probability,
            np.asarray(model(probe, training=False))[0, 0],
            rtol=1e-5,
            atol=1e-6,
        )
    config.update(model=model, gradcam=gradcam, inference_lock=Lock())
    return config
