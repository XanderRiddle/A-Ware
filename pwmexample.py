#!/usr/bin/env python3
import gpiod
import time

# RAW GPIO OUTPUT - NO ERROR HANDLING
chip = gpiod.Chip('/dev/gpiochip0')  # Will crash if doesn't exist
line = chip.get_line(0)             # Will crash if invalid offset
line.request(consumer="raw-output", type=gpiod.LINE_REQ_DIR_OUT)

print("Setting GPIO HIGH (3.3V) - Ctrl+C to stop")
line.set_value(1)  # 3.3V output

while True:
    time.sleep(1)