"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 allium.py train --dataset=/path/to/allium/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 allium.py train --dataset=/path/to/allium/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 allium.py train --dataset=/path/to/allium/dataset --weights=imagenet

    # Apply detections to an image
    python3 allium.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>

"""

import cv2
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

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

class AlliumDataset(utils.Dataset):

    def load_allium(self, dataset_dir, subset):
        """Load a subset of the Allium dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        print("==" * 10)
        print("Loading dataset")
        print("==" * 10)
        
        # Add classes. Not dividing and dividing
        self.add_class("allium", 1, "not_dividing")
        self.add_class("allium", 2, "dividing")
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations        
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stored in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [[r['region_attributes'], r['shape_attributes']] for r in a['regions'].values()]
            else:
                polygons = [[r['region_attributes'], r['shape_attributes']] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "allium",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)
            
        print("==" * 10)
        print("Finished loading dataset")
        print("==" * 10)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not an allium dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "allium":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        # Initialize class IDs list
        classIDs = []
        # Create the mask
        info = self.image_info[image_id]
        
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get class ID
            classID = None
            class_name = p[0]["cell_type"]
            if class_name == "not_dividing":
                classID = 1
            elif class_name == "dividing":
                classID = 2
            else:
                raise Exception('Encountered unsupported class name. Terminate.')
                break
            classIDs.append(classID)
            
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p[1]['all_points_y'], p[1]['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance.
        classIDs = np.array(classIDs, dtype=np.int32)
        return mask.astype(np.bool), classIDs

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "allium":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = AlliumDataset()
    dataset_train.load_allium(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = AlliumDataset()
    dataset_val.load_allium(args.dataset, "val")
    dataset_val.prepare()

    # Training - Stage 1
    # Pretraing heads
    print("\nTraining network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15,
                layers='heads')
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Training Resnet layer 4+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=20,
                layers='4+')
    # Training - Stage 3
    # Finetune layers from ResNet stage 3 and up
    print("Training Resnet layer 3+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=50,
                layers='all')


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

    
def detect_and_annotate(model, dir_path=None):
    '''
    Perform the detection for the images in the specified folder, approximate 
    predicted masks with polygons and compose annotations in VIA annotation
    format. 
    This is used to generate the predictions to make it easier to 
    label the data. 
    '''
    
    # List files in the directory
    image_list = os.listdir(dir_path)
    image_list.sort()
    
    annotations = {}
    for image_name in image_list[0:2]:
        # Load the image
        image = skimage.io.imread(os.path.join(dir_path, image_name))
        # Print the status
        print("Annotating image {}".format(image_name))
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        class_ids = r['class_ids']
        masks = r['masks']
        
        # Create regions from masks
        regions = []
        for mask_no in range(masks.shape[2]):
            class_id = class_ids[mask_no]
            mask = masks[:, :, mask_no]
            mask = np.array(mask * 255, dtype=np.uint8)
            mask = np.expand_dims(mask, 2)
            # Approximate masks with polygons
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours != []: 
                c = contours[0]
                peri = cv2.arcLength(c, closed=True)
                approx = cv2.approxPolyDP(c, epsilon=0.002 * peri, closed=True)
                approx = approx.reshape(approx.shape[0], approx.shape[2])
                # Create region part
                region = {}
                region['region_attributes'] = {}
                region['shape_attributes'] = {}
                region['shape_attributes']['name'] = 'polygon'
                all_points_x = list(approx[:, 0])
                all_points_y = list(approx[:, 1])
                all_points_x = [int(x) for x in all_points_x]
                all_points_y = [int(y) for y in all_points_y]
                region['shape_attributes']['all_points_x'] = all_points_x
                region['shape_attributes']['all_points_y'] = all_points_y
                if class_id == 1:
                    region['region_attributes']['cell_type'] = 'not_dividing'
                else:
                    region['region_attributes']['cell_type'] = 'dividing'
                regions.append(region)
            else: 
                continue
            
        # Create annotation
        ann = {}
        ann['file_attributes'] = {}
        ann['filename'] = image_name
        ann['regions'] = regions
        size = int(os.stat(os.path.join(dir_path, image_name)).st_size)
        ann['size'] = size
        # Add annotation to the annotation list
        annotations[image_name + str(size)] = ann

    # Save annotations to .json file 
    annotations = json.dumps(annotations)
    with open("generated_annotations.json", "w") as outfile:  
        outfile.write(annotations)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect onion cells.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/allium/dataset/",
                        help='Directory of the Allium dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Directory to annotate images in')
    parser.add_argument('--directory', required=False,
                    metavar="path or URL to image",
                    help='Image to apply detections on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = AlliumConfig()
    else:
        class InferenceConfig(AlliumConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, image_path=args.image)
    elif args.command == "detect_and_annotate":
        detect_and_annotate(model, dir_path=args.directory)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))