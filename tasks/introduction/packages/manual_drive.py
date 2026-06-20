from typing import Dict, Tuple
import logging
logger = logging.getLogger(__name__)

SPEED = 1
TURN = 0.5


def get_motor_speeds(keys_pressed: Dict[str, bool]) -> Tuple[float, float]:
    up = keys_pressed.get('up', False)
    down = keys_pressed.get('down', False)
    left = keys_pressed.get('left', False)
    right = keys_pressed.get('right', False)

    y = up - down     
    x = right - left 

    if y != 0:
        left_motor = y * 0.5 if x >= 0 else y * 0.2
        right_motor = y * 0.5 if x <= 0 else y * 0.2
        return left_motor, right_motor
    else:
        return x * 0.5, x * -0.5

