import cv2
import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom
from tensorflow.keras import Model



def softmax(x):
    value = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return value

def ScoreCam(batch_holder,images,model, layer_id, batch_size,so_ba,img_size,max_N=-1):
    batch_holder_image = batch_holder[so_ba:so_ba + batch_size]
    original_images = images[so_ba:so_ba + batch_size]

    class_id = np.argmax(model.predict(batch_holder_image), axis=-1)
    activation_map_array = Model(inputs=model.input, outputs=model.layers[layer_id[0]].output).predict(batch_holder_image)

    if max_N != -1:
        activation_map_standard_list = [np.std(activation_map_array[0, :, :, k]) for k in range(activation_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(activation_map_standard_list), max_N)[:max_N]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(activation_map_standard_list)[unsorted_max_indices])]
        activation_map_array = activation_map_array[:, :, :, max_N_indices]
    heatmap_batch = np.zeros(
        (len(batch_holder_image), model.layers[layer_id[0]].output.shape[1], model.layers[layer_id[0]].output.shape[1]))

    for i in range(len(batch_holder_image)):
        input_shape = model.layers[0].output_shape[0][1:]

        activation_map_resized_list = [cv2.resize(activation_map_array[i, :, :, k], input_shape[:2], interpolation=cv2.INTER_LINEAR)
                                for k in range(activation_map_array.shape[3])]

        activation_map_normalized_list = []
        for activation_map_resized in activation_map_resized_list:
            if np.max(activation_map_resized) - np.min(activation_map_resized) != 0:
                activation_map_normalized = activation_map_resized / (np.max(activation_map_resized) - np.min(activation_map_resized))
            else:
                activation_map_normalized = activation_map_resized
            activation_map_normalized_list.append(activation_map_normalized)

        masked_input_list = []
        for activation_map_normalized in activation_map_normalized_list:
            masked_input = np.copy(batch_holder_image[i:i + 1])
            for k in range(3):
                masked_input[0, :, :, k] *= activation_map_normalized
            masked_input_list.append(masked_input)
        masked_input_array = np.concatenate(masked_input_list, axis=0)

        pred_from_masked_input_array = softmax(model.predict(masked_input_array))

        weights = pred_from_masked_input_array[:, class_id[i]]

        cam = np.dot(activation_map_array[i, :, :, :], weights)
        cam = np.maximum(0, cam)

        cam /= np.max(cam)
        heatmap_batch[i, :] = cam

    heatmap_resized = np.zeros((heatmap_batch.shape[0], img_size, img_size))
    for img_idx in range(heatmap_batch.shape[0]):
        heatmap_resized[img_idx, :] = zoom(heatmap_batch[img_idx], (
        int(img_size / model.layers[layer_id[0]].output.shape[1]), int(img_size / model.layers[layer_id[0]].output.shape[1]),), order=2)

    heatmap_final = np.expand_dims(heatmap_resized, -1)
    return heatmap_final, original_images, batch_holder_image


