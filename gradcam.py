import numpy as np
import tensorflow as tf

def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    # Generic Grad-CAM implementation that tries to find a conv layer automatically.
    if last_conv_layer_name is None:
        # find last conv layer
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        # fallback: return zeros
        return np.zeros((224,224))

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.resize(heatmap, (224,224))
    return heatmap
