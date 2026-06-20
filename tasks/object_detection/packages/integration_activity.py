from typing import Tuple

MODEL_PATH = "tasks/object_detection/models/best.onnx"

def NUMBER_FRAMES_SKIPPED() -> int:
    return 1

def filter_by_classes(pred_class: int) -> bool:
    return pred_class == 0

# def filter_by_scores(score: float) -> bool:
#     return score >= 0.6

def filter_by_scores(score: float) -> bool:
    keep = score >= 0.6
    print(f"[filter_by_scores] score={score:.3f} keep={keep}")
    return keep

def filter_by_bboxes(bbox):
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    area = w * h
    keep = w >= 10 and h >= 10 and area > 800
    print(f"[filter_by_bboxes] bbox={bbox} area={area} keep={keep}")
    return keep

# def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
#     xmin, ymin, xmax, ymax = bbox
#     width = xmax - xmin
#     height = ymax - ymin

#     if width < 10 or height < 10:
#         return False

#     area = width * height
#     return area > 800