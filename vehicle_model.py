import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# works for keras 2.0 and tensorflow 1.15
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

from utils import load_weights, Box, yolo_net_out_to_car_boxes, draw_box

keras.backend.set_image_dim_ordering('th')


class vehicle_model:
    def __init__(self):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model = self.create_model_with_weights()
        
    def create_model_with_weights(self):
        model = Sequential()
        model.add(Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        model.add(Convolution2D(64,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        model.add(Convolution2D(128,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        model.add(Convolution2D(256,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        model.add(Convolution2D(512,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        model.add(Convolution2D(1024,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Convolution2D(1024,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Convolution2D(1024,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(1470))
        load_weights(model,'./yolo-tiny.weights')
        return model

    def predict(self,imagePath, threshold=0.17):
        image = plt.imread(imagePath)
        image_crop = image
        resized = cv2.resize(image_crop,(448,448))
        batch = np.transpose(resized,(2,0,1))
        batch = 2*(batch/255.) - 1
        temp  = 2*(resized/255.) - 1
        batch = np.expand_dims(batch, axis=0)
        out   = self.model.predict(batch)
        boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)
        return boxes

    def draw_on_image(self, boxes, imagePath):
        f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
        ax1.imshow(plt.imread(imagePath))
        ax2.imshow(draw_box(boxes,plt.imread(imagePath),[[0, plt.imread(imagePath).shape[1]], [0, plt.imread(imagePath).shape[0]]]))
    
    def frame_func(self, image):
        crop = image
        resized = cv2.resize(crop,(448,448))
        batch = np.array([resized[:,:,0],resized[:,:,1],resized[:,:,2]])
        batch = 2*(batch/255.) - 1
        batch = np.expand_dims(batch, axis=0)
        out = self.model.predict(batch)
        boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)
        return draw_box(boxes,image,[[0, image.shape[1]], [0, image.shape[0]]])
    
    def clip_predict(self, filename, project_video_output):
        clip1 = VideoFileClip("./project_video.mp4")
        lane_clip = clip1.fl_image(self.frame_func) #NOTE: this function expects color images!!
        lane_clip.write_videofile(project_video_output, audio=False)
