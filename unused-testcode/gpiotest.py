#!/usr/bin/env python3
import gpiod
import time

# Configuration - Using GPIO0_C4 (Pin 31)
GPIO_CHIP = '/dev/gpiochip0'  
GPIO_LINE_OFFSET = 100  # GPIO0_C4 is typically offset 100 (Rockchip convention)
CONSUMER = 'pwm-test'


    # Open GPIO chip
chip = gpiod.Chip(GPIO_CHIP)

# Get the GPIO line
line = chip.get_line(GPIO_LINE_OFFSET)

# Request as output, starting LOW
line.request(consumer=CONSUMER, type=gpiod.LINE_REQ_DIR_OUT, default_val=0)

print("Testing GPIO output on Pin 31 (GPIO0_C4):")

# Blink test (1 second on, 1 second off)
for i in range(5):
    line.set_value(1)
    print("HIGH - You should measure ~3.3V")
    time.sleep(1)
    
    line.set_value(0)
    print("LOW - You should measure 0V")
    time.sleep(1)
# Cleanup
if 'line' in locals():
    line.set_value(0)
    line.release()
if 'chip' in locals():
    chip.close()
print("GPIO cleanup complete")