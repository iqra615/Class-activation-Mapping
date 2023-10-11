import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom
from tensorflow.keras import Model

def grad_cam_plus(batch_holder,images,model,layer_id,batch_size,so_ba,img_size):
    batch_holder_image = batch_holder[so_ba:so_ba + batch_size]
    original_images = images[so_ba:so_ba + batch_size]
    last_conv_layer = model.layers[layer_id[0]]
    gradcamplus_model = Model([model.inputs], [last_conv_layer.output, model.output])
    heatmap_batch = np.zeros((len(batch_holder_image), model.layers[layer_id[0]].output.shape[1],
                              model.layers[layer_id[0]].output.shape[1]))

    class_channel = tf.TensorArray(dtype=tf.float32, size=len(batch_holder_image))

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                feature_map, predictions = gradcamplus_model(batch_holder_image)
                for i in range(len(batch_holder_image)):
                    pred_index = tf.argmax(predictions[i])
                    class_channel = class_channel.write(i, predictions[i, pred_index])
                class_channel = class_channel.stack()
                feature_map_first_grad = gtape3.gradient(class_channel, feature_map)
            feature_map_second_grad = gtape2.gradient(feature_map_first_grad, feature_map)
        feature_map_third_grad = gtape1.gradient(feature_map_second_grad, feature_map)

    total_sum = np.sum(feature_map, axis=(1, 2))

    for i in range(len(batch_holder_image)):
        alpha_num = feature_map_second_grad[i]
        alpha_denom = feature_map_second_grad[i] * 2.0 + feature_map_third_grad[i] * total_sum[i]
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

        alphas_value = alpha_num / alpha_denom
        alpha_normalized_constant = np.sum(alphas_value, axis=(0, 1))
        alphas_value /= alpha_normalized_constant

        weights = np.maximum(feature_map_first_grad[i], 0.0)

        weights_for_gradcamplus = np.sum(weights * alphas_value, axis=(0, 1))
        grad_CAM_map = np.sum(weights_for_gradcamplus * feature_map[i], axis=2)

        heatmap = np.maximum(grad_CAM_map, 0)

        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat

        heatmap_batch[i, :] = heatmap

    heatmap_resized = np.zeros((heatmap_batch.shape[0], img_size, img_size))
    for img_idx in range(heatmap_batch.shape[0]):
        heatmap_resized[img_idx, :] = zoom(heatmap_batch[img_idx], (int(img_size/feature_map.shape[1]),int(img_size/feature_map.shape[1]),), order=2)

    heatmap_final = np.expand_dims(heatmap_resized, -1)
    return heatmap_final,original_images,batch_holder_image
