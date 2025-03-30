#!/usr/bin/env python3
import gpiod
from gpiod.line import Direction, Value

# Configuration
CHIP_PATH = '/dev/gpiochip0'
GPIO_OFFSET = 0  # CHANGE THIS to your actual GPIO offset
CONSUMER = "pwm-test"

chip = gpiod.Chip("/dev/gpiochip0")
line = chip.request_lines(0)
line.set_value(Value.ACTIVE)

try:
    while True:
        pass
finally:
    line.set_value(Value.INACTIVE)
    line.release()
    chip.close()