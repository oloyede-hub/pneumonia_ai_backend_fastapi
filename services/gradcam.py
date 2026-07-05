from zipfile import Path

import tensorflow as tf
import numpy as np
import cv2
import uuid
import base64
from pathlib import Path



def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="block5_conv3"
):
    vgg16 = model.get_layer("vgg16")

    last_conv_layer = vgg16.get_layer(
        last_conv_layer_name
    )

    # Image -> block5_conv3
    # This create a new model
    conv_model = tf.keras.Model(
        inputs=vgg16.input,
        outputs=last_conv_layer.output
    )

    # Build classifier head manually
    classifier_input = tf.keras.Input(
        shape=last_conv_layer.output.shape[1:]
    )

    x = classifier_input

    passed_target = False

    for layer in vgg16.layers:
        if passed_target:
            x = layer(x)

        if layer.name == last_conv_layer_name:
            passed_target = True

    for layer in model.layers[1:]:
        x = layer(x)


    classifier_model = tf.keras.Model(
        classifier_input,
        x
    )


    with tf.GradientTape() as tape:

        conv_output = conv_model(img_array)

        tape.watch(conv_output)

        predictions = classifier_model(
            conv_output
        )

        class_channel = predictions[:, 0]

    grads = tape.gradient(
        class_channel,
        conv_output
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


def save_gradcam(
    original_image,
    heatmap
):
    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.resize(
        heatmap,
        (
            original_image.shape[1],
            original_image.shape[0]
        )
    )

    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    output_dir = Path("static") / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{uuid.uuid4()}.png"

    cv2.imwrite(
        filepath,
        overlay
    )

    return filepath



