import json
from typing import List

# Classes the model is trained to detect.
# The index here is the class ID written into YOLO label files.
CLASSES = ['duckie', 'truck', 'sign']

# Images are resized to this square size before training.
IMAGE_SIZE = 416


def convert_labelme_json(json_path: str, img_w: int, img_h: int) -> List[str]:
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    yolo_annotations = []
    
    for shape in data.get("shapes", []):
        label = shape.get("label")
        
        if label not in CLASSES:
            continue
            
        cls_id = CLASSES.index(label)
        
        p1, p2 = shape.get("points")
        xmin = min(p1[0], p2[0])
        xmax = max(p1[0], p2[0])
        ymin = min(p1[1], p2[1])
        ymax = max(p1[1], p2[1])
        
        xmin = xmin * IMAGE_SIZE / img_w
        xmax = xmax * IMAGE_SIZE / img_w
        ymin = ymin * IMAGE_SIZE / img_h
        ymax = ymax * IMAGE_SIZE / img_h
        
        cx = (xmin + xmax) / 2 / IMAGE_SIZE
        cy = (ymin + ymax) / 2 / IMAGE_SIZE
        w  = (xmax - xmin) / IMAGE_SIZE
        h  = (ymax - ymin) / IMAGE_SIZE
        
        yolo_annotations.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
    return yolo_annotations
