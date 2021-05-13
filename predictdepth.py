import sys
sys.path.insert(0, './FCRN_DepthPrediction_master')

import os
import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2

from FCRN_DepthPrediction_master.tensorflow import models

tf.disable_v2_behavior()

# Take an image in OpenCV to estimate depth
def take_image():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
    cv2.imshow("test", frame)
    img_name = "img.jpg"
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    cam.release()
    cv2.destroyAllWindows()


def predict():
    # Take Image to Calibrate Scene
    take_image()

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open("img.jpg")
    img = img.resize([width, height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, "NYU_FCRN.ckpt")

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()

        # Save Output Array
        print(f"Prediction: {pred}")
        np.save('pred.npy', pred)
        return pred


def main():
    # Predict the image
    predict()
    os._exit(0)

if __name__ == '__main__':
    main()

        



