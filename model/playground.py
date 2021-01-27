"""
PLAYGROUND
Make predictions for one image
"""
import cv2
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class AlliumConfig(Config):
    """Configuration for training.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "allium"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + not_dividing + dividing

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < threshold confidence
    DETECTION_MIN_CONFIDENCE = 0.3
    DETECTION_NMS_THRESHOLD = 0.5
    
    MAX_GT_INSTANCES = 200
    DETECTION_MAX_INSTANCES = 200
    
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1152

############################################################
#  Dataset
############################################################


def detect(model, image_path=None): 
    '''
    Detect cells for specified piece and display the predictions 
    '''
    class_names = ['BG', 'not_dividing', 'dividing']
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Define colors
    colors = []
    for i in range(len(r['class_ids'])):
        if r['class_ids'][i] == 1:
            color = [0, 0, 0] #Black color for not dividing cells
        else:
            color = [0, 0, 1] #Blue color for dividing cells
        colors.append(color)
    # Display image with masks
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'],
        class_names, r['scores'], colors=colors)

    

class InferenceConfig(AlliumConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=DEFAULT_LOGS_DIR)

weights_path = model.find_last()


# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


'''
Perform the manipulations here
'''
dir_path = 'data/test/'







