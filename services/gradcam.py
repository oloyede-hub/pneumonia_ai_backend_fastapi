from zipfile import Path

import tensorflow as tf
import numpy as np
import cv2
import uuid
import base64
from pathlib import Path



# def make_gradcam_heatmap(
#     img_array,
#     model,
#     backbone_name,
#     last_conv_name
# ):
#     backbone = model.get_layer(backbone_name)

#     last_conv = backbone.get_layer(
#         last_conv_name
#     )

#     # Image -> block5_conv3
#     # This create a new model
#     conv_model = tf.keras.Model(
#     inputs=backbone.input,
#     outputs=last_conv.output
#     )

    

#     # Build classifier head manually
#     classifier_input = tf.keras.Input(
#         shape=last_conv.output.shape[1:]
#     )

#     x = classifier_input

#     passed_target = False

#     for layer in backbone.layers:
#         if passed_target:
#             x = layer(x)

#         if layer.name == last_conv_name:
#             passed_target = True

#     for layer in model.layers[1:]:
#         x = layer(x)


#     classifier_model = tf.keras.Model(
#         classifier_input,
#         x
#     )


#     with tf.GradientTape() as tape:

#         conv_output = conv_model(img_array)

#         tape.watch(conv_output)

#         predictions = classifier_model(
#             conv_output
#         )

#         class_channel = predictions[:, 0]

#     grads = tape.gradient(
#         class_channel,
#         conv_output
#     )

#     pooled_grads = tf.reduce_mean(
#         grads,
#         axis=(0, 1, 2)
#     )

#     conv_output = conv_output[0]

#     heatmap = tf.reduce_sum(
#         conv_output * pooled_grads,
#         axis=-1
#     )

#     heatmap = tf.maximum(
#         heatmap,
#         0
#     )

#     heatmap /= (
#         tf.reduce_max(heatmap) + 1e-8
#     )

#     return heatmap.numpy()




# def make_gradcam_heatmap(
#     img_array,
#     model,
#     backbone_name,
#     last_conv_name,
# ):
#     backbone = model.get_layer(backbone_name)


#     last_conv = backbone.get_layer(last_conv_name)

#     conv_model = tf.keras.Model(
#         backbone.input,
#         last_conv.output
#     )

#     classifier_input = tf.keras.Input(
#         shape=last_conv.output.shape[1:]
#     )

#     x = classifier_input

#     passed = False

#     for layer in backbone.layers:

#         if passed:
#             x = layer(x)

#         if layer.name == last_conv_name:
#             passed = True

#     for layer in model.layers[1:]:
#         x = layer(x)

#     classifier_model = tf.keras.Model(
#         classifier_input,
#         x
#     )

#     conv_model = tf.keras.Model(
#         inputs=backbone.input,
#         outputs=last_conv.output
#     )

#     classifier_input = tf.keras.Input(
#         shape=last_conv.output.shape[1:]
#     )

#     x = classifier_input

#     passed_target = False

#     for layer in backbone.layers:

#         if passed_target:
#             x = layer(x)

#         if layer.name == last_conv_name:
#             passed_target = True

#     for layer in model.layers[1:]:
#         x = layer(x)

#     classifier_model = tf.keras.Model(
#         classifier_input,
#         x
#     )

#     with tf.GradientTape() as tape:

#         conv_output = conv_model(img_array)

#         tape.watch(conv_output)

#         predictions = classifier_model(
#             conv_output
#         )

#         class_channel = predictions[:, 0]

#     grads = tape.gradient(
#         class_channel,
#         conv_output
#     )

#     pooled_grads = tf.reduce_mean(
#         grads,
#         axis=(0, 1, 2)
#     )

#     conv_output = conv_output[0]

#     heatmap = tf.reduce_sum(
#         conv_output * pooled_grads,
#         axis=-1
#     )

#     heatmap = tf.maximum(
#         heatmap,
#         0
#     )

#     heatmap /= (
#         tf.reduce_max(heatmap) + 1e-8
#     )

#     return heatmap.numpy()


def make_gradcam_heatmap(
    img_array,
    model,
    backbone_name,
    last_conv_name
):
    # 1. Extract the inner backbone (e.g., ResNet50 or VGG16) from the wrapper
    backbone = model.get_layer(backbone_name)
    last_conv_layer = backbone.get_layer(last_conv_name)

    # 2. Build a sub-model *only* for the backbone up to the last conv layer
    # This is safe and built directly from the backbone's fixed inputs
    conv_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=last_conv_layer.output
    )

    # 3. Track the operations using GradientTape
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        
        # Get the feature maps from the last conv layer
        conv_outputs = conv_model(inputs)
        tape.watch(conv_outputs)
        
        # 4. Extract the classifier head layers dynamically from your main model
        # Find where the backbone sits in your global sequential model
        backbone_idx = None
        for idx, layer in enumerate(model.layers):
            if layer.name == backbone_name:
                backbone_idx = idx
                break
                
        # Forward pass through the rest of the backbone after last_conv (if any)
        # This handles downstream layers within the backbone wrapper itself
        x = conv_outputs
        passed_conv = False
        for layer in backbone.layers:
            if passed_conv:
                x = layer(x)
            if layer.name == last_conv_name:
                passed_conv = True

        # Forward pass through all layers following the backbone in the top-level model
        if backbone_idx is not None:
            for layer in model.layers[backbone_idx + 1:]:
                x = layer(x)
        else:
            # Fallback if backbone structure differs
            for layer in model.layers[1:]:
                x = layer(x)
                
        predictions = x
        
        # 5. Automatically target the top predicted class channel
        best_class_idx = tf.argmax(predictions[0])
        class_channel = predictions[:, best_class_idx]

    # 6. Extract gradients of the top class score w.r.t the feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # 7. Global average pool the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 8. Weight the feature maps by their gradient importances
    # Extract index 0 to get the pure 2D feature dimensions
    conv_outputs_0 = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs_0 * pooled_grads, axis=-1)

    # 9. Apply ReLU and normalize safely
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    print("Raw feature map shape:", conv_outputs[0].shape) 
    # Max coordinates in 7x7 matrix
    print("Max activation coordinate:", np.unravel_index(np.argmax(heatmap), heatmap.shape))

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
        ),
         interpolation=cv2.INTER_CUBIC
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



