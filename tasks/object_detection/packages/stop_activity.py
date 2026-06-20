from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]
class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}

DUCKIE_CLASS         = 0
LATERAL_MARGIN_RATIO = 0.15

STOP_PERSISTENCE_FRAMES = 4
_consecutive_in_path = 0

LANE_OFFSET_RATIO     = 0.10   
LANE_HALF_WIDTH_BOTTOM = 0.32  
LANE_HALF_WIDTH_TOP    = 0.06  
PROXIMITY_Y_RATIO      = 0.55  
PROXIMITY_AREA_PX      = 4000  

def _in_lane(x_ground, y_ground, W, H):
    lane_center_x = W * (0.5 + LANE_OFFSET_RATIO)

    prox_y = H * PROXIMITY_Y_RATIO
    if y_ground <= prox_y:
        return False  
    t = (y_ground - prox_y) / max(1.0, (H - prox_y))   
    half_w = W * (LANE_HALF_WIDTH_TOP + t * (LANE_HALF_WIDTH_BOTTOM - LANE_HALF_WIDTH_TOP))

    return abs(x_ground - lane_center_x) <= half_w

def _bbox_overlaps_lane(xmin, xmax, y_ground, W, H):
    lane_center_x = W * (0.5 + LANE_OFFSET_RATIO)
    prox_y = H * PROXIMITY_Y_RATIO
    if y_ground <= prox_y:
        return False
    t = (y_ground - prox_y) / max(1.0, (H - prox_y))
    half_w = W * (LANE_HALF_WIDTH_TOP + t * (LANE_HALF_WIDTH_BOTTOM - LANE_HALF_WIDTH_TOP))
    lane_left  = lane_center_x - half_w
    lane_right = lane_center_x + half_w
    return xmax >= lane_left and xmin <= lane_right

def _frame_has_blocking_duckie(detections, img_w, img_h):
    if not detections:
        return False, ""

    best = None
    for bbox, score, class_id in detections:
        if class_id != DUCKIE_CLASS:
            continue
        xmin, ymin, xmax, ymax = bbox
        area = (xmax - xmin) * (ymax - ymin)
        x_ground = (xmin + xmax) / 2
        y_ground = ymax

        if y_ground < img_h * PROXIMITY_Y_RATIO and area < PROXIMITY_AREA_PX:
            continue
        if not _bbox_overlaps_lane(xmin, xmax, y_ground, img_w, img_h):
            continue

        reason = (f"duckie ahead: score={score:.2f}, foot=({x_ground:.0f},{y_ground}), "
                  f"area={area}px^2")
        if best is None or y_ground > best[0]:
            best = (y_ground, reason)

    if best is not None:
        return True, best[1]
    return False, ""

STOP_PERSISTENCE_FRAMES   = 4   
RESUME_PERSISTENCE_FRAMES = 6   

_consecutive_in_path = 0
_consecutive_clear   = 0
_is_stopped          = False

def should_stop(detections, img_size):
    global _consecutive_in_path, _consecutive_clear, _is_stopped

    frame_blocking, reason = _frame_has_blocking_duckie(detections, img_size, img_size)

    if frame_blocking:
        _consecutive_in_path += 1
        _consecutive_clear = 0
    else:
        _consecutive_clear += 1
        _consecutive_in_path = 0

    if _is_stopped:
        if _consecutive_clear >= RESUME_PERSISTENCE_FRAMES:
            _is_stopped = False
    else:
        if _consecutive_in_path >= STOP_PERSISTENCE_FRAMES:
            _is_stopped = True

    return (_is_stopped, reason if _is_stopped else "")