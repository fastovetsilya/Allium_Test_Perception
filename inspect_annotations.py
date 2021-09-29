"""
Inspect VIA .json annotations 
"""
# Import libraries
import json

# Load json annotations
path_to_annotations = '/media/saltair/Library/Allium_cepa_experiment/repo/Allium_Test_Perception/model/data/annotations/Nd_Control_1_2_annotations.json'
path_to_annotations2 = '/media/saltair/Library/Allium_cepa_experiment/repo/Allium_Test_Perception/model/data/annotations/Nd_Control_2_1_annotations.json'

with open(path_to_annotations, 'r') as f:
    annotations = f.read()
annotations = json.loads(annotations)

with open(path_to_annotations2, 'r') as f:
    annotations2 = f.read()
annotations2 = json.loads(annotations2)

# Perform additional operations below to inspect annotations

# # don't need the dict keys
# annotations = list(annotations.values())
# # The VIA tool saves images in the JSON even if they don't have any
# # annotations. Skip unannotated images.
# annotations = [a for a in annotations if a['regions']]
#  # Add images
# for a in annotations[0:3]:
#     # Get the x, y coordinaets of points of the polygons that make up
#     # the outline of each object instance. These are stores in the
#     # shape_attributes (see json format above)
#     # The if condition is needed to support VIA versions 1.x and 2.x.
#     if type(a['regions']) is dict:
#         polygons = [[r['region_attributes'], r['shape_attributes']] for r in a['regions'].values()]
#     else:
#         polygons = [[r['region_attributes'], r['shape_attributes']] for r in a['regions']]
        
# for i, p in enumerate(polygons):
#     print(i)
#     print(p)