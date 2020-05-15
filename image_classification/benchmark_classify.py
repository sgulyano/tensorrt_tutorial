import time
import numpy as np
from console_classify import read_img, inference
import argparse
import cv2
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
import tensorflow.compat.v1.keras.backend as K

config = ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = Session(config=config)
K.set_session(session)  # set this TensorFlow session as the default session for Keras

# %%bash
# mkdir ./data
# wget  -O ./data/img0.JPG "https://d17fnq9dkz9hgj.cloudfront.net/breed-uploads/2018/08/siberian-husky-detail.jpg?bust=1535566590&width=630"
# wget  -O ./data/img1.JPG "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
# wget  -O ./data/img2.JPG "https://www.artis.nl/media/filer_public_thumbnails/filer_public/00/f1/00f1b6db-fbed-4fef-9ab0-84e944ff11f8/chimpansee_amber_r_1920x1080.jpg__1920x1080_q85_subject_location-923%2C365_subsampling-2.jpg"
# wget  -O ./data/img3.JPG "https://www.familyhandyman.com/wp-content/uploads/2018/09/How-to-Avoid-Snakes-Slithering-Up-Your-Toilet-shutterstock_780480850.jpg"

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help="SavedModel Directory")
args = parser.parse_args()

model_path = args.model_dir


##
batch_size = 8
batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)

print('Load Images')
for i in range(batch_size):
    img = read_img('./data/img%d.JPG' % (i % 4))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    batched_input[i, :] = x
batched_input = tf.constant(batched_input)
print('batched_input shape: ', batched_input.shape)

##
print('Load Model')
saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
print(signature_keys)

infer = saved_model_loaded.signatures['serving_default']
print(infer.structured_outputs)

print('Pillow')
for i in range(4):
    img_path = 'data/img%d.JPG'%i
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)
    
    labeling = infer(x)
    preds = labeling['predictions'].numpy()
    
    print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))

print('OpenCV')
for i in range(4):
    img_path = 'data/img%d.JPG'%i
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)
    
    labeling = infer(x)
    preds = labeling['predictions'].numpy()
    
    print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))

##
N_warmup_run = 50
N_run = 1000
elapsed_time = []

for i in range(N_warmup_run):
    labeling = infer(batched_input)

for i in range(N_run):
    start_time = time.time()
    labeling = infer(batched_input)
    end_time = time.time()
    elapsed_time = np.append(elapsed_time, end_time - start_time)
    if i % 50 == 0:
        print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean())*1000))

print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))