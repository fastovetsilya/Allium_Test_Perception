"""
Change sizes in annotations for .png to .jpg convertation
"""

# Import libraries
import os
import json

# Path to the images and annotations
path_to_images = "/media/saltair/Library/Allium_cepa_experiment/repo/Allium_Test_Perception/" \
                 "crop_images/output/jpg/Nd_Control_1_2"
path_to_annotations = "/media/saltair/Library/Allium_cepa_experiment/via_annotations/" \
                      "reviewed/VLAD_Nd_Control_1_2/Nd_Control_1_2_annotations.json"

# Load json annotations
with open(path_to_annotations, 'r') as f:
    annotations = f.read()
annotations = json.loads(annotations)

# Create new annotations file
annotations_modified = {}

# Iterate through annotation keys
for key in annotations.keys():
    image_filename = key[:-8]
    image_size = int(os.stat(os.path.join(path_to_images, image_filename)).st_size)
    new_key = image_filename + str(image_size)
    new_value = annotations[key]
    new_value["size"] = image_size
    annotations_modified[new_key] = new_value
    
# Save modified annotations
annotations_modified = json.dumps(annotations_modified)
with open("modified_annotations.json", "w") as outfile:  
    outfile.write(annotations_modified)
    