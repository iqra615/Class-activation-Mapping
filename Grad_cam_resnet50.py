#importting Libraries
from utils import pre_process_input
import tensorflow as tf
from grad_cam import compute_gradcam

#hyper_parameters
No_of_images =2500
batch_size = 250
img_size = 224
layer_id = [-3]  #layer id to consider for gradcam computation
img_dir = '/Input_data'
model_name = 'resnet50'

#variables for loop and metrics evalaution
so_ba = 0
sum_i = count_Ioc = sum_un = count_Doc = 0

#number of batches taken to evaluate
steps = int(No_of_images / batch_size)

#images_in_batch
batch_holder,images = pre_process_input(No_of_images,img_dir,img_size,model_name)

#pre-trained resnet50 model
model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet',classes=1000)

#gradcam computation and its metrics.
for i in range(steps):
    gradcam_batch,original_images,batch_holder_image = compute_gradcam(batch_holder,images,model,layer_id,batch_size,so_ba,img_size)
