#!/usr/bin/env python3
import gpiod
from gpiod.line import Direction, Value

# 1. Create the line settings
line_settings = gpiod.line_settings()
line_settings.direction = Direction.OUTPUT  # Set as output
line_settings.output_value = Value.ACTIVE   # Default HIGH (3.3V)

# 2. Create the config dictionary
config = {
    # Key: GPIO offset (or name if available)
    # Value: LineSettings object (or None for defaults)
    31: line_settings  # Using offset 31 for physical pin 31
}

# 3. Request the lines
with gpiod.Chip('/dev/gpiochip0') as chip:
    request = chip.request_lines(
        config=config,
        consumer="my-pwm-control",
        output_values={31: Value.ACTIVE}  # Immediately set HIGH
    )
    
    # Keep the line active until Ctrl+C
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass  # Context manager handles cleanup