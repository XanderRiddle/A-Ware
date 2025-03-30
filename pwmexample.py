#!/usr/bin/env python3
import gpiod
import time

# Using gpiochip0, offset 0 as example (change to your working GPIO)
CHIP = '/dev/gpiochip0'
OFFSET = 0  # Change this to your working GPIO offset

try:
    chip = gpiod.Chip(CHIP)
    line = chip.get_line(OFFSET)
    line.request(consumer="5v-test", type=gpiod.LINE_REQ_DIR_OUT)
    
    print(f"Setting GPIO {OFFSET} to HIGH (3.3V)")
    line.set_value(1)
    print("Voltage should now be active. Press Ctrl+C to stop.")
    
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nShutting down")
finally:
    line.set_value(0)
    line.release()
    chip.close()