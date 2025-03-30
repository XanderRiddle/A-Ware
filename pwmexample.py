#!/usr/bin/env python3
import gpiod
import time

# Configuration
GPIO_CHIP = 'gpiochip0'  # Default GPIO chip
GPIO_PIN = 0             # GPIO pin number
PWM_FREQ = 1000          # Frequency in Hz (1kHz)
DUTY_CYCLE = 100         # 100% duty cycle

def main():
    print(f"Setting GPIO {GPIO_PIN} to 100% PWM at {PWM_FREQ}Hz")
    
    try:
        # Get the GPIO chip
        chip = gpiod.Chip(GPIO_CHIP)
        
        # Get the GPIO line
        line = chip.get_line(GPIO_PIN)
        
        # Request the line as output
        line.request(consumer="pwm_100", type=gpiod.LINE_REQ_DIR_OUT)
        
        print("Outputting 100% PWM (continuous high signal)")
        
        try:
            # For 100% duty cycle, we just set the line high continuously
            line.set_value(1)
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping PWM")
            
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Cleanup
        try:
            if 'line' in locals():
                line.set_value(0)
                line.release()
            if 'chip' in locals():
                chip.close()
        except:
            pass
        
        print("GPIO cleaned up")

if __name__ == "__main__":
    main()