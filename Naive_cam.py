import numpy as np
from tensorflow.keras import Model
from scipy.ndimage import zoom


# naive cam model
def cam_vis_model(model,layer_id):
    weights, _ = model.layers[layer_id[0]].get_weights()
    cam_model = Model(inputs=model.input, outputs=(model.layers[layer_id[1]].output, model.layers[layer_id[0]].output))
    return cam_model,weights

# naive cam computation
def compute_naive_cam(batch_holder,images,vis_model,weights,batch_size,so_ba,img_size):
        batch_holder_image = batch_holder[so_ba:so_ba + batch_size]
        original_images = images[so_ba:so_ba + batch_size]
        feature_maps, class_predictions = vis_model.predict(batch_holder_image)
        class_id = np.argmax(class_predictions, axis=1)
        cam_batch = np.zeros((batch_size, feature_maps.shape[1], feature_maps.shape[1]))
        for j in range(len(class_id)):
            feature_maps_batch = feature_maps[j]
            cam = np.zeros(dtype=np.float32, shape=feature_maps_batch.shape[0:2])
            for i, w in enumerate(weights[:, class_id[j]]):
                cam += w * feature_maps_batch[:, :, i]
            cam_batch[j, :] = cam

        cam_batch_resized = np.zeros((cam_batch.shape[0], img_size, img_size))

        for img_idx in range(cam_batch.shape[0]):
            cam_batch_resized[img_idx, :] = zoom(cam_batch[img_idx], (int(img_size/feature_maps.shape[1]), int(img_size/feature_maps.shape[1])), order=2)

        cam_batch_resized = np.maximum(cam_batch_resized, 0)
        cam_batch_resized = cam_batch_resized / np.reshape(np.max(cam_batch_resized, axis=(1, 2)), (-1, 1, 1))

        cam_batch = np.expand_dims(cam_batch_resized, -1)

        return cam_batch,original_images,batch_holder_image
