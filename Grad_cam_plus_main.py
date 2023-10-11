#importing Libraries
import tensorflow as tf
from gradcam_plus import grad_cam_plus
from utils import pre_process_input,evaluate_metrics,final_metrics

#hyper_parameters
No_of_images = 2500
batch_size = 100
img_size = 224
layer_id = [-6] #layer id to consider for gradcam++ computation
img_dir = '/Input_data'
model_name = 'vgg16'

#variables for loop and metrics evalaution
so_ba = 0
sum_i = count_Ioc = sum_un = count_Doc = 0

#number of batches taken to evaluate
steps = int(No_of_images / batch_size)

#images_in_batch
batch_holder,images = pre_process_input(No_of_images,img_dir,img_size,model_name)

#pre-trained vgg16 model
model = tf.keras.applications.vgg16.VGG16(
    include_top=True, weights='imagenet',classes=1000)

#gradcam++ computation and its metrics.
for i in range(steps):
    gradcam_plus_batch,original_images,batch_holder_image = grad_cam_plus(batch_holder,images,model,layer_id,batch_size,so_ba,img_size)
