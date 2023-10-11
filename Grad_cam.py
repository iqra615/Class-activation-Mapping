import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom

def compute_gradcam(batch_holder,images,model,layer_id,batch_size,so_ba,img_size):
    batch_holder_image = batch_holder[so_ba:so_ba + batch_size]
    original_images = images[so_ba:so_ba + batch_size]
    gradcam_model = tf.keras.models.Model(
        [model.inputs], [model.layers[layer_id[0]].output, model.output])

    heatmap_batch = np.zeros((len(batch_holder_image), model.layers[layer_id[0]].output.shape[1],
                              model.layers[layer_id[0]].output.shape[1]))
    class_channel = tf.TensorArray(dtype=tf.float32, size=len(batch_holder_image))
    with tf.GradientTape() as tape:
        feature_map, class_predictions = gradcam_model(batch_holder_image)
        for i in range(len(batch_holder_image)):
            pred_index = tf.argmax(class_predictions[i])
            class_channel = class_channel.write(i, class_predictions[i, pred_index])
        class_channel = class_channel.stack()
        grads = tape.gradient(class_channel, feature_map)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        for i in range(len(batch_holder_image)):
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads[i], feature_map[i]), axis=-1)
            heatmap_batch[i, :] = heatmap

        heatmap_resized = np.zeros( (heatmap_batch.shape[0], img_size, img_size))
        for img_idx in range(heatmap_batch.shape[0]):
            heatmap_resized[img_idx,:] = zoom(heatmap_batch[img_idx],(int(img_size/feature_map.shape[1]),int(img_size/feature_map.shape[1])), order=2)


        heatmap_resized = np.maximum(heatmap_resized, 0)
        heatmap_resized = heatmap_resized / np.reshape(np.max(heatmap_resized, axis=(1,2)),(-1,1,1))

        heatmap = np.expand_dims( heatmap_resized, -1)
        return heatmap,original_images,batch_holder_image
