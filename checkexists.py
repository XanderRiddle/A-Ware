import gpiod

gpiod.is_gpiochip_device("/dev/gpiochip0")

with gpiod.Chip("/dev/gpiochip0") as chip:
    info = chip.get_info()
    print(f"{info.name} [{info.label}] ({info.num_lines} lines)")