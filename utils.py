import tensorflow as tf
import numpy as np
import os
import random
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import datasets

#pre-process_input in batches
def pre_process_input(be,img_dir,img_size,model_name):
    images = np.zeros((be, img_size, img_size, 3))
    for i, img in enumerate(os.listdir(img_dir)):
        load_image = image.load_img(os.path.join(img_dir, img))
        array_image = image.img_to_array(load_image.resize((img_size, img_size)))
        images[i, :] = array_image
        if i == be-1:
            break
    if model_name == "resnet50":
        batch_holder = tf.keras.applications.resnet50.preprocess_input(images.copy())
    elif model_name == 'vgg16':
        batch_holder = tf.keras.applications.vgg16.preprocess_input(images.copy())
    else:
        print("Incorrect model_name")
    return batch_holder,images

  #class probabilites prediciton
def class_prob(img,model,model_name,class_id = None):
    if class_id is None:
        class_predictions = model.predict(img)
        class_id = np.argmax(class_predictions,axis=1)
        class_probal = np.max(class_predictions,axis=1)
        return class_id,np.round(class_probal*100,2)
    else:
        if model_name == 'resnet50':
            preprocessed = tf.keras.applications.resnet50.preprocess_input(img.copy())
        elif model_name == 'vgg16':
            preprocessed = tf.keras.applications.vgg16.preprocess_input(img.copy())
        else:
            print("Incorrect model_name for Explanation map")
        class_predictions = model.predict(preprocessed)
        class_probal = np.zeros(len(class_id))
        for i in range(len(class_id)):
            class_probal[i] = class_predictions[i,class_id[i]]
        return np.round(class_probal*100,2)
