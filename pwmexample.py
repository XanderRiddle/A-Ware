#!/usr/bin/env python3
import gpiod
from gpiod.line import Direction, Value

# Configuration
CHIP_PATH = '/dev/gpiochip0'
GPIO_OFFSET = 0  # CHANGE THIS to your actual GPIO offset
CONSUMER = "pwm-test"

# 1. Get the GPIO chip
chip = gpiod.Chip(CHIP_PATH)

# 2. Create line request configuration
config = gpiod.LineRequest()
config.consumer = CONSUMER
config.request_type = Direction.OUTPUT

# 3. Get the line and make the request
line = config.get_value()
chip.request_lines(line)  # This creates the LineRequest object

# 4. Now we can control the line
print(f"Setting GPIO {GPIO_OFFSET} to HIGH")
line.set_value(Value.ACTIVE)  # 3.3V output

# Keep it running
try:
    while True:
        pass
finally:
    line.set_value(Value.INACTIVE)
    line.release()
    chip.close()