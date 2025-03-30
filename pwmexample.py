#!/usr/bin/env python3
import gpiod
import time

# Use GPIO0_C4 (Pin 31)
GPIO_CHIP = '/dev/gpiochip0'  
GPIO_PIN = 31  # GPIO0_C4
CONSUMER = 'pwm-demo'

try:
    chip = gpiod.Chip(GPIO_CHIP)
    line = chip.get_line(GPIO_PIN)
    line.request(consumer=CONSUMER, type=gpiod.LINE_REQ_DIR_OUT)
    
    print(f"Outputting 100% PWM on GPIO0_C4 (Pin 31)")
    line.set_value(1)  # Constant high = 100% PWM
    
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nStopping PWM")
finally:
    line.set_value(0)
    line.release()
    chip.close()