def map_distance_to_pwm(distance):
    if distance is None or distance > 10000:
        return 0  # No signal if out of range
    elif distance < 500:
        return 100  # Max signal if too close
    return int((100 - ((distance - 500) / (10000 - 500) * 100)))