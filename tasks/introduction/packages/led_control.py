import colorsys
from typing import List


def set_turning_leds(direction: str) -> dict:
    """Set LEDs to indicate turning direction."""
    off = [0.0, 0.0, 0.0]
    yellow = [1.0, 1.0, 0.0]
    white = [1.0, 1.0, 1.0]
    red = [1.0, 0.0, 0.0]
    if direction == "left":
        return {
            0: yellow,
            2: off,
            3: off,
            4: yellow 
        }
    elif direction == "right":
        return {
            0: off,
            2: yellow,
            3: yellow,
            4: off 
        }
    elif direction == "forward":
        return {
            0: white,
            2: white,
            3: off,
            4: off
        }
    elif direction == "stop":
        return {
            0: off,
            2: off,
            3: red,
            4: red
        }
    else:
        return {
            0: off,
            2: off,
            3: off,
            4: off
        }