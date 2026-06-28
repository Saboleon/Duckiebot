import os
import yaml
import cv2
import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent

_CFG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config'))
_CFG_REAL = os.path.join(_CFG_DIR, 'lane_servoing_config.yaml')
_CFG_SIM  = os.path.join(_CFG_DIR, 'lane_servoing_config_sim.yaml')


class LaneFollower:

    def __init__(self, cfg, sim=True):
        config_path = _CFG_SIM if sim else _CFG_REAL
        self._agent   = LaneServoingAgent(config_path=config_path)
        self._nominal = float(self._agent.base_speed) or 0.15

        try:
            with open(config_path) as f:
                sc = yaml.safe_load(f) or {}
            self._trim          = float(sc.get('trim', 0.0))
            self._creep_on_lost = bool(sc.get('creep_on_lost', not sim))
        except Exception:
            self._trim          = 0.0
            self._creep_on_lost = not sim  # sim stops, real bot creeps

    def compute(self, frame_bgr, base_speed):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        left, right = self._agent.compute_commands(rgb)

        info  = self._agent.last_debug_info
        ratio = base_speed / self._nominal if self._nominal > 1e-6 else 1.0

        if not info.get('lane_detected', False):
            if self._creep_on_lost:
                creep = base_speed * 0.35
                left, right = creep, creep
            else:
                left, right = 0.0, 0.0
        else:
            left  = float(np.clip((left  + self._trim) * ratio, 0.0, 1.0))
            right = float(np.clip((right - self._trim) * ratio, 0.0, 1.0))

        debug = {
            "error":    round(float(info.get("lateral_error", 0.0)), 3),
            "lane_px":  int(info.get("total_lane_pixels", 0)),
            "detected": bool(info.get("lane_detected", False)),
        }
        return left, right, debug

    def draw(self, frame_bgr, debug):
        info = self._agent.last_debug_info
        h, w = frame_bgr.shape[:2]
        cv2.line(frame_bgr, (w // 2, int(h * 0.5)), (w // 2, h), (255, 255, 255), 1)

        slice_ys = info.get("slice_ys", [])
        for i, y in enumerate(slice_ys):
            ys = info.get("yellow_xs", [])
            ws = info.get("white_xs", [])
            if i < len(ys):
                cv2.circle(frame_bgr, (int(ys[i]), int(y)), 6, (0, 255, 255), -1)
            if i < len(ws):
                cv2.circle(frame_bgr, (int(ws[i]), int(y)), 6, (255, 255, 255), -1)

        err = info.get("lateral_error", 0.0)
        cv2.putText(frame_bgr, f"lane err: {err:+.2f}", (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame_bgr
