import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

print("Tensorflow version: ", tf.version.VERSION)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
import tensorflow.compat.v1.keras.backend as K

config = ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = Session(config=config)
K.set_session(session)  # set this TensorFlow session as the default session for Keras


def read_img(img_path, size=(224,224)):
    # read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, size)

def inference(img, infer):
    # inference
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)
    labeling = infer(x)
    return labeling['Logits'].numpy()


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help="Image Path")
    parser.add_argument('model_dir', help="SavedModel Directory")
    args = parser.parse_args()

    img_path = args.img_path
    model_path = args.model_dir
    
    ##
    print('Read image')
    img = read_img(img_path)
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (224, 224))

    cv2.imshow('image',img.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ##
    print('Load Model')
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    print(infer.structured_outputs)

    ##
    print('Inference')
    preds = inference(img, infer)
    print('Predicted: ', decode_predictions(preds, top=3)[0])

    ##
    from tensorflow.keras.preprocessing import image
    img_keras = image.load_img(img_path, target_size=(224, 224))
    img_keras = image.img_to_array(img_keras)
    ##
    print('Inference')
    preds_keras = inference(img_keras, infer)
    print('Predicted Keras: ', decode_predictions(preds_keras, top=3)[0])