from tensorflow import keras
from huggingface_hub import hf_hub_download

MODELS = {
    "vgg16": {
        "filename": "penumonia1.keras",
        "preprocess": keras.applications.vgg16.preprocess_input,
        "last_conv_layer": "block5_conv3",
        "backbone": "vgg16",
        "gradcam_mode": "reconstuct",
    },
    "resnet50": {
        "filename": "resnnet_penumonia_v3.keras",
        "preprocess": keras.applications.resnet.preprocess_input,
        "last_conv_layer": "conv5_block3_out",
        "backbone": "resnet50",
        "gradcam_mode": "graph",
    }
}

def load_model_config(model_name: str):
    config = MODELS[model_name]

    model_path = hf_hub_download(
        repo_id="Buore/pneumonia_detectors",
        filename=config["filename"]
    )

    model = keras.models.load_model(model_path)

    return {
        "model": model,
        "preprocess": config["preprocess"],
        "last_conv_layer": config["last_conv_layer"],
        "backbone": config["backbone"],
        "gradcam_mode": config["gradcam_mode"]
    }