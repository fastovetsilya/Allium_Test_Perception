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

import os
import sys
import json
import glob
import shutil
import imgaug
import datetime
import numpy as np
import skimage.draw
from sklearn.model_selection import KFold

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.getcwd()

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
    
    USE_MINI_MASK = False
    # MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

############################################################
#  Dataset class
############################################################

class AlliumDataset(utils.Dataset):

    def load_allium(self, dataset_dir, subset):
        """Load a subset of the Allium dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        # Add classes. Not dividing and dividing
        self.add_class("allium", 1, "not_dividing")
        self.add_class("allium", 2, "dividing")
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Print status
        print("==" * 30)
        print("Loading dataset: {}".format(subset))
        print("==" * 30)

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
            
        # Print status
        print("==" * 30)
        print("Finished loading dataset")
        print("==" * 30)

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


def compute_batch_ap(model, model_config, dataset, limit, verbose=1):
    """
    Validates the model on the dataset in the provided directory, and 
    computes validation metric (mAP).
    """
    # Validation dataset
    dataset_val = AlliumDataset()
    dataset_val.load_allium(dataset, "val")
    dataset_val.prepare()
    if limit:
        image_ids = dataset_val.image_ids[:limit]
    else: 
        image_ids = dataset_val.image_ids
    print("Images: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))
    
    # Compute mAP
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
        # Compute AP over range 0.5 to 0.95
        r = results[0]
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        APs.append(ap)
        if verbose:
            # info = dataset.image_info[image_id]
            meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
            print("{:3} {}   AP: {:.2f}".format(
                meta["image_id"][0], meta["original_image_shape"][0], ap))
    
    # Print the results
    mAP = np.mean(APs)
    print("Average precisions are: {}".format(APs))
    print("Mean average precision is: {}".format(mAP))
          
    return mAP


def prepare_crossval_splits(images_path="data/images/", annotations_path="data/annotations", 
                            random_state=123, n_splits=10, split_no=0):
    """
    TODO: write description here and with comments in the arguments
    Parameters
    ----------
    images_path : TYPE
        DESCRIPTION.
    annotations_path : TYPE
        DESCRIPTION.
    random_state : TYPE, optional
        DESCRIPTION. The default is None.
    n_splits : TYPE, optional
        DESCRIPTION. The default is 10.
    split_no : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    
    # Define paths to images and annotations
    images_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames]
    annotations_list = os.listdir(annotations_path)
    
    # Load annotations
    annotations = {}
    for annotation_file in annotations_list:
        annotation = json.load(open(os.path.join(annotations_path, annotation_file)))
        annotations.update(annotation)

    # Create train and val directories
    try:
        train_path = os.path.join(images_path.split("/")[0], "train")
        val_path = os.path.join(images_path.split("/")[0], "val")
        shutil.rmtree(train_path)
        os.mkdir(train_path)
        shutil.rmtree(val_path)
        os.mkdir(val_path)
    except:
        os.mkdir(train_path)
        os.mkdir(val_path)
    
    # Create train and validation splits
    if not n_splits:
        images_list_train = images_list
        images_list_val = images_list
    else:
        kf = KFold(n_splits=n_splits, random_state=random_state)
        kf.get_n_splits(images_list)
        kf_count = 0
        for train_index, val_index in kf.split(images_list):
            if kf_count == split_no:
                print("TRAIN:", train_index, "VAL:", val_index)
                images_list_train, images_list_val = np.array(images_list)[train_index].tolist(), np.array(images_list)[val_index].tolist()
                break
            kf_count += 1
        
    # Create annotations for train and validation splits
    # Save annotations and copy images to the directories
    annotations_train = {}
    annotations_val = {}
    # Train
    for image_path in images_list_train:
        image_name = image_path.split("/")[-1]
        for key in annotations.keys():
            if image_name == key.split(".")[0] + "." + key.split(".")[1][:3]:
                annotations_train[image_name + key.split(".")[1][3:]] = annotations[key]
                # Copy image to the train directory
                shutil.copyfile(image_path, os.path.join(train_path, image_name))       
    annotations_train = json.dumps(annotations_train)
    with open(os.path.join(train_path, "via_region_data.json"), "w") as ann_file:
        ann_file.write(annotations_train)
                
    # Validation
    for image_path in images_list_val:
        image_name = image_path.split("/")[-1]
        for key in annotations.keys():
            if image_name == key.split(".")[0] + "." + key.split(".")[1][:3]:
                annotations_val[image_name + key.split(".")[1][3:]] = annotations[key]
                # Copy image to the train directory
                shutil.copyfile(image_path, os.path.join(val_path, image_name))        
    annotations_val = json.dumps(annotations_val)
    with open(os.path.join(val_path, "via_region_data.json"), "w") as ann_file:
        ann_file.write(annotations_val)

    print("Dataset prepared.")
       

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
    
    # Data augmentations
    augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.Flipud(0.5), 
                    imgaug.augmenters.Multiply((0.5, 1.5))
                    #imgaug.augmenters.Rotate((-45, 45)),
                    #imgaug.augmenters.Rot90((1, 3))
                    #imgaug.augmenters.MultiplyBrightness((0.5, 1.5))
                    #imgaug.augmenters.ChangeColorTemperature((1100, 10000)), 
                    #imgaug.augmenters.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5)), 
                    #imgaug.augmenters.GammaContrast((0.5, 2.0), per_channel=True),
                ])

    # # Training - Stage 1
    # # Pretraing heads
    print("\nTraining network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15,
                augmentation = augmentation,
                layers='heads')
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Training Resnet layer 4+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=20,
                augmentation = augmentation,
                layers='4+')
    # Training - Stage 3
    # Finetune layers from ResNet stage 3 and up
    print("Training Resnet layer 3+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                augmentation = augmentation,
                epochs=100,
                layers='all')

def train_cv(model_dir, model_weights, model_config, k_fold):
    """
    //Write docstrings here//
    """
    import keras.backend as backend
    
    # Initialize CV directory
    cv_dir = os.path.join(model_dir, "CV")
    # Clean CV directory
    try:
        shutil.rmtree(cv_dir)
        os.mkdir(cv_dir)
    except:
        os.mkdir(cv_dir)
        
    # Train model for each K
    for k in range(k_fold):
        k_logs_dir = os.path.join(cv_dir, str(k))
        os.mkdir(k_logs_dir)
    
        # Load the model
        model = modellib.MaskRCNN(mode="training", config=model_config,
                              model_dir=k_logs_dir)
        
        # Load model weights
        if model_weights.lower() == "coco":
            weights_path = COCO_WEIGHTS_PATH
            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        elif model_weights.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()
        elif model_weights.lower() == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = model_weights
            
        print("Loading weights ", weights_path)
        if model_weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
        
        # Create K's split
        print("Preparing new cross-validation split, k={}".format(k))
        prepare_crossval_splits(images_path="data/images/", annotations_path="data/annotations", 
                            random_state=123, n_splits=k_fold, split_no=k)
        
        # Train the model
        print("Training model for k={}".format(k))
        train(model)
        
        # Clean Keras sesson to free the memory
        backend.clear_session()
        

def compute_cv_results(cv_dir, model_config, n_splits=5, random_state=123):
    """
    //Add docstrings here//
    """
    import keras.backend as backend
    cv_results = {}
    
    # Compute K for k-fold CV
    K = os.listdir(cv_dir)
    K = [int(el) for el in K]
    K.sort()

    # Check the consistency of CV folders
    num_models = []
    for k in K:
        k_dir = os.path.join(cv_dir, str(k))
        k_model_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(k_dir) for f in filenames]
        k_model_list = [k_model for k_model in k_model_list if k_model.split(".")[-1] == "h5"]
        num_models.append(len(k_model_list))
    if not all(n==num_models[0] for n in num_models):
        print("=" * 50)
        print("Warning: inconsistent number of models in K-fold CV directory")
        print("=" * 50)
        
    # Go through the models and compute CV metrics
    for k in K:
        k_dir = os.path.join(cv_dir, str(k))
        k_model_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(k_dir) for f in filenames]
        k_model_list = [k_model for k_model in k_model_list if k_model.split(".")[-1] == "h5"]
        
        # Prepare validation split
        prepare_crossval_splits(images_path="data/images/", annotations_path="data/annotations", 
                            random_state=random_state, n_splits=n_splits, split_no=k)
        
        for model_path in k_model_list:
           n_model = int(model_path.split("/")[-1].split(".")[0].split("_")[-1])
           
           # Load the model
           model_logdir = "/".join(model_path.split("/")[:-1]) + "/"
           model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_logdir)
           model.load_weights(model_path, by_name=True)
           
           # Run the prediction
           mAP = int(compute_batch_ap(model, model_config=model_config, 
                                      dataset="data/", limit=None, verbose=1))
    
           # Append the result to the dictionary
           cv_result = {"k": k, 
                        "n": n_model, 
                        "mAP": mAP}
           cv_results.update(cv_result)
           
           # Clear the session
           backend.clear_session()
           
           # Json serialize and save the results at each step
           cv_results = json.dumps(cv_results)
           with open(os.path.join(cv_dir, "CV_results"), "w") as f: 
               f.write(cv_results)
    
          
def detect(model, image_path=None): 
    """
    Detect cells for specified piece and display the predictions 
    """
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
    """
    Perform the detection for the images in the specified folder, approximate 
    predicted masks with polygons and compose annotations in VIA annotation
    format. 
    This is used to generate the predictions to make it easier to 
    label the data. 
    """
    import cv2
    
    # List files in the directory
    image_list = os.listdir(dir_path)
    image_list.sort()
    
    annotations = {}
    for image_name in image_list:
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


def report(model, config, dir_path=None):
    """
    Detect full size images using sliding window and generate report for the 
    selected batch (directory)
    """
    import cv2
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 933120000 * 10**2 #Increase max image size
    from math import ceil
    
    # Extract the configurations
    overlap_factor = 0.3
    window_size = [4000, 4000]
    confidence = config.DETECTION_MIN_CONFIDENCE 
    threshold = config.DETECTION_NMS_THRESHOLD 
    overlap_factor = [overlap_factor, overlap_factor]
    
    # List files in the directory
    image_list = os.listdir(dir_path)
    image_list.sort()
    
    for image_name in image_list:
        print("=" * 50)
        print("Analysing image {}".format(image_name))
        print("=" * 50)
        
        image = Image.open(os.path.join(dir_path, image_name))
        width, height = image.size
        image_size = [width, height]
        
        # Re-compute the optimal overlap factor (so that the window slides smoothly)
        steps = [ceil((image_size[0] - window_size[0]) // (window_size[0] * (1 - overlap_factor[0]))), 
                 ceil((image_size[1] - window_size[1]) // (window_size[1] * (1 - overlap_factor[1])))]  
        
        # Adjust steps
        if steps[0] < 0:
            steps[0] = 0
            
        if steps[1] < 0:
            steps[1] = 0
        
        if steps[0] <= 1 and window_size[0] < image_size[0]:
            steps[0] = 2
        
        if steps[1] <= 1 and window_size[1] < image_size[1]:
            steps[1] = 2
        
        print("\n--- The prediction will be performed in ", (steps[0] + 1) * (steps[1] + 1), 
              " steps ---")    
            
        if steps[0] == 0:
            overlap_factor_refined_0 = 0
        else:
            overlap_factor_refined_0 = 1 - ((image_size[0] - window_size[0]) / 
                                            steps[0] / window_size[0])
            
        if steps[1] == 0:
            overlap_factor_refined_1 = 0
        else:
            overlap_factor_refined_1 = 1 - ((image_size[1] - window_size[1]) / 
                                            steps[1] / window_size[1]) 
    
        overlap_factor_refined = [overlap_factor_refined_0, 
                                  overlap_factor_refined_1]
        
        print("\nRefined overlap factor(x,y) is {},{}".format(overlap_factor_refined_0, overlap_factor_refined_1))
        
        # Perform sliding window prediction
        # Initialize collectors for all predictions 
        all_steps = []
        all_boxes = []
        all_polygons = []
        all_confidences = []
        all_classIDs = []
        
        # Crop the image and perform sliding window predictions
        print("\nPerforming sliding window prediction")
        cropped_part_no = 0
        for height_step in range(steps[1] + 1):
            for width_step in range(steps[0] + 1):
                print("\nAnalyzing part {}".format(cropped_part_no))
                # Crop the image region
                cropped_part_pos_x = int(width_step * (1-overlap_factor_refined[0]) * window_size[0])
                cropped_part_pos_y = int(height_step * (1-overlap_factor_refined[1]) * window_size[1])
                crop_box = (cropped_part_pos_x, cropped_part_pos_y, 
                            cropped_part_pos_x + window_size[0], cropped_part_pos_y + window_size[1])
                cropped_part = image.crop(crop_box)
                # Perform predictions
                cropped_part = np.array(cropped_part)
                r = model.detect([cropped_part], verbose=0)[0]
                boxes = r['rois']
                confidences = r['scores']
                confidences = [float(conf) for conf in confidences]
                classIDs = r['class_ids']
                classIDs = [int(cid) for cid in classIDs]
                masks = r['masks']
                # Transform the output of the model
                boxes = [[int(box[1] + cropped_part_pos_x), int(box[0] + cropped_part_pos_y), 
                          int(box[3] - box[1]), 
                          int(box[2] - box[0])] for box in boxes]
                # Create polygons from masks (to save memory)
                polygons = []
                for mask_no in range(masks.shape[2]):
                    mask = masks[:, :, mask_no]
                    mask = np.array(mask * 255, dtype=np.uint8)
                    mask = np.expand_dims(mask, 2)
                    # Approximate masks with polygons
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if contours != []: 
                        c = contours[0]
                        peri = cv2.arcLength(c, closed=True)
                        approx = cv2.approxPolyDP(c, epsilon=0.002 * peri, closed=True)
                        approx = np.array([[a[0][0] + cropped_part_pos_x, a[0][1] + cropped_part_pos_y] for a in approx])
                        approx = approx.reshape(approx.shape[0], 1, approx.shape[1])
                        polygons.append(approx)
                    else: 
                        polygons.append([])
                
                # Add to the list of all outputs
                steps_lst = [[width_step, height_step] for i in range(len(classIDs))]
                all_steps.extend(steps_lst)
                all_boxes.extend(boxes)
                all_polygons.extend(polygons)
                all_confidences.extend(confidences)
                all_classIDs.extend(classIDs)
                
                cropped_part_no += 1
                
        # Apply non-max suppression to the detected boxes
        print("\nApplying Non-Max suppression")
        all_idxs = cv2.dnn.NMSBoxes(all_boxes, all_confidences, confidence, threshold)
        # Save the predictions
        # predictions = {"boxes": all_boxes,
        #                "polygons": all_polygons,
        #                "confidences": all_confidences, 
        #                "classIDs": all_classIDs,
        #                "idxs": all_idxs}
        
        # Draw the predictions      
        print("\nDrawing predictions")
        image = np.array(image)
        image = image[:,:,::-1].copy()
        image_polygon_canvas = image.copy()
        colors = [[0, 0, 0], [0, 0, 0], [255, 0, 0]]
        if len(all_idxs) > 0:
            for i in all_idxs.flatten():
                # extract bounding box coordinates
                x, y = all_boxes[i][0], all_boxes[i][1]
                w, h = all_boxes[i][2], all_boxes[i][3]
                # draw the bounding box and label on the image
                color = [int(c) for c in colors[all_classIDs[i]]]
                try:
                    # Draw bounding boxes
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 10)
                    # Draw polygons 
                    cv2.polylines(image,[all_polygons[i]],True,color, 10)
                    # Filling the polygons
                    cv2.fillPoly(image_polygon_canvas, [all_polygons[i]], color)
                except:
                    continue
        image = cv2.addWeighted(image_polygon_canvas, 0.3, image, 0.7, 0)
        cv2.imwrite('predictions.jpg', image)
        
        
############################################################
#  Main block
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect onion cells.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'detect' or 'report'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/allium/dataset/",
                        help='Directory of the Allium dataset')
    parser.add_argument('--weights', required=False,
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
                        metavar="",
                        help='Image to apply detections on')
    parser.add_argument('--k_fold', required=False,
                        default=5,
                        metavar="",
                        help='K number for K-fold cross-validation')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train" or args.command == "train_cv":
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
    elif args.command == "train_cv":
        pass
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.command not in ["train_cv", "compute_cv_results"]:
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
    else: 
        pass

    # Load weights
    if args.command not in ["train_cv", "compute_cv_results"]:
        print("Loading weights ", weights_path)
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
    else:
        pass

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "train_cv":
        train_cv(model_dir=args.logs, model_weights=args.weights, 
                 model_config=config, k_fold=args.k_fold)
    elif args.command == "compute_cv_results":
        compute_cv_results(cv_dir=args.directory, model_config=config, n_splits=args.k_fold)
    elif args.command == "detect":
        detect(model, image_path=args.image)
    elif args.command == "validate":
        compute_batch_ap(model, model_config=config, dataset=args.dataset, limit=None)
    elif args.command == "detect_and_annotate":
        detect_and_annotate(model, dir_path=args.directory)
    elif args.command == "report":
        report(model, config=config, dir_path=args.directory)
    else:
        print("'{}' is not recognized. "
              "Use 'train', 'detect' or 'report'".format(args.command))
