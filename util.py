# A-Ware Code Library
# Xander Riddle, Austin Graves, Josh Mao, Max Pethel, Aditya Garg
# Spring 2025

def map_distance_to_pwm(distance):
    """
    An inverse function of camera-detected distance to pwm haptic duty cycle.

    distance: Distance to a point in mm ranging from 500 to 10000.

    Returns duty cycle percentage.
    """
    if distance is None or distance > 10000:
        return 0  # No signal if out of range
    elif distance < 500:
        return 100  # Max signal if too close
    return int((100 - ((distance - 500) / (10000 - 500) * 100)))