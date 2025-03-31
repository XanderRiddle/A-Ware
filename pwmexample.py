import gpiod

# Request pin 22 (GPIO5_C3, offset 179)
with gpiod.Chip("/dev/gpiochip0") as chip:
    config = {
        7: gpiod.LineSettings(
            direction=gpiod.line.Direction.OUTPUT,
            output_value=gpiod.line.Value(0)
        )
    }
    request = chip.request_lines(config=config, consumer="test-pin22")
    request.set_value(7, gpiod.line.Value(1))  # Set high
    input("Press Enter to exit...")  # Keeps the line held high
# Line released automatically after 'with' block