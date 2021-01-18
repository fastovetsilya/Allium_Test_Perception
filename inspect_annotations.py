"""
Inspect VIA annotations 
"""
# Import libraries
import json

# Load json annotations
path_to_annotations = '/media/saltair/Library/Allium_cepa_experiment/via_annotations/reviewed/ILYA_Nd_Control_1_1/Nd_Control_1_1_annotations.json'
with open(path_to_annotations, 'r') as f:
    annotations = f.read()
annotations = json.loads(annotations)

# Perform additional operations below to inspect annotations