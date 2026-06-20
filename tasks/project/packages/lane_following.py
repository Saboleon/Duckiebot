"""Lane following.

We reuse the lane follower from the `visual_lane_servoing` task - it is already
tuned and proven (edge + colour masks to find the yellow/white lane boundaries,
an adaptive lane-width estimate so it aims at the lane *centre* even when only
one boundary is visible, and a PD controller). This wrapper just lets the agent
ask for a slower "approach" speed while keeping that proven steering.

`compute()` returns `(left_speed, right_speed, debug)`. At cruise speed the
output is identical to the visual_lane_servoing agent; lower speeds scale both
wheels equally so the turn geometry is preserved.
"""

import cv2
import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent


class LaneFollower:

    def __init__(self, cfg):
        self._agent = LaneServoingAgent()
        # the servoing agent's own tuned speed is our "cruise" reference
        self._nominal = float(self._agent.base_speed) or 0.15

    def compute(self, frame_bgr, base_speed):
        """base_speed is the forward speed to aim for (cruise or approach)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        left, right = self._agent.compute_commands(rgb)

        ratio = base_speed / self._nominal if self._nominal > 1e-6 else 1.0
        left  = float(np.clip(left  * ratio, 0.0, 1.0))
        right = float(np.clip(right * ratio, 0.0, 1.0))

        info = self._agent.last_debug_info
        debug = {
            "error":    round(float(info.get("lateral_error", 0.0)), 3),
            "lane_px":  int(info.get("total_lane_pixels", 0)),
            "detected": bool(info.get("lane_detected", False)),
        }
        return left, right, debug

    def draw(self, frame_bgr, debug):
        """Overlay the lane-line sample points the controller is tracking."""
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
