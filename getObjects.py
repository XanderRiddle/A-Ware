import depthai as dai
import gpiod
import time
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Define sources
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)

# Define outputs
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")

# Set Mono Camera Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Configure Stereo Depth
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# Connect Mono Cameras to Stereo Depth
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# RGB Camera Setup
camRgb.setPreviewSize(300, 300)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setFps(30)

# Set up Neural Network
detectionNetwork.setBlobPath("mobilenet-ssd_openvino_2021.4_6shave.blob")  # Pretrained Model
detectionNetwork.setConfidenceThreshold(0.5)
camRgb.preview.link(detectionNetwork.input)

# Linking Outputs
detectionNetwork.out.link(xoutNN.input)
stereo.depth.link(xoutDepth.input)

# GPIO Setup using gpiod
try:
    chip = gpiod.chip('gpiochip0')  # Changed from gpiod.Chip to gpiod.chip
except Exception as e:
    print(f"Failed to open GPIO chip: {e}")
    sys.exit(1)

# Define the GPIO pins (adjust these to your actual GPIO numbers)
pins = {
    'Left': 17,    # Example GPIO number
    'Middle': 27,  # Example GPIO number
    'Right': 22    # Example GPIO number
}

# Dictionary to hold our GPIO lines
gpio_lines = {}

try:
    for name, pin in pins.items():
        try:
            line = chip.get_line(pin)
            line.request(consumer='depthai_pwm', type=gpiod.LINE_REQ_DIR_OUT)
            gpio_lines[name] = line
            print(f"Successfully configured GPIO {pin} for {name}")
        except Exception as e:
            print(f"Failed to configure GPIO {pin} for {name}: {e}")
            gpio_lines[name] = None
except Exception as e:
    print(f"GPIO configuration failed: {e}")
    sys.exit(1)

# PWM Settings
def set_pwm_duty_cycle(line, duty_cycle):
    """
    Set the PWM duty cycle for the specified GPIO line.
    line: gpiod.Line object
    duty_cycle: The duty cycle in percentage (0 - 100).
    """
    if line is None:
        return
        
    period = 0.01  # 10 ms period for PWM
    pulse_width = (duty_cycle / 100) * period
    
    try:
        line.set_value(1)  # Activate pin
        time.sleep(pulse_width)
        line.set_value(0)  # Deactivate pin
        time.sleep(period - pulse_width)
    except Exception as e:
        print(f"PWM control error: {e}")

# Connect to device and start pipeline
try:
    with dai.Device(pipeline) as device:
        print("DepthAI pipeline started")
        detectionsQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        while True:
            try:
                detections = detectionsQueue.get().detections
                depthFrame = depthQueue.get().getFrame()
                depthFrame = np.array(depthFrame, dtype=np.uint16)  # Convert depth frame to NumPy array

                depthHeight = depthFrame.shape[0]
                depthWidth = depthFrame.shape[1]

                object_list = []
                if detections:
                    for detection in detections:
                        x1 = int(detection.xmin * depthWidth)
                        y1 = int(detection.ymin * depthHeight)
                        x2 = int(detection.xmax * depthWidth)
                        y2 = int(detection.ymax * depthHeight)
                        label = detection.label

                        # Compute center of bounding box
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Ensure coordinates are within bounds
                        if 0 <= cx < depthFrame.shape[1] and 0 <= cy < depthFrame.shape[0]:
                            distance = int(depthFrame[cy, cx])  # Extract distance in mm
                            if distance == 0:
                                distance = None  # Ignore invalid depth values
                        else:
                            distance = None

                        if cx < depthWidth / 3:
                            quadrant = "Left"
                        elif cx < 2 * depthWidth / 3:
                            quadrant = "Middle"
                        else:
                            quadrant = "Right"

                        object_list.append({
                            "label": label, 
                            "distance_mm": distance, 
                            "quadrant": quadrant
                        })
                        
                        # Set PWM duty cycle based on the detected distance
                        if distance is not None and quadrant in gpio_lines:
                            # Map distance to duty cycle (adjust these values as needed)
                            duty_cycle = max(0, min(100, ((10000 - distance) / 9500) * 100))
                            set_pwm_duty_cycle(gpio_lines[quadrant], duty_cycle)
                        elif quadrant in gpio_lines:
                            set_pwm_duty_cycle(gpio_lines[quadrant], 0)

                print("Detected Objects:", object_list)
                
                time.sleep(0.1)  # Reduced sleep time for more responsive PWM

            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(1)

except Exception as e:
    print(f"DepthAI error: {e}")

finally:
    # Cleanup GPIO
    print("Cleaning up GPIO...")
    for name, line in gpio_lines.items():
        if line is not None:
            try:
                line.set_value(0)
                line.release()
            except Exception as e:
                print(f"Error cleaning up GPIO {name}: {e}")
    
    if 'chip' in locals():
        try:
            chip.close()
        except Exception as e:
            print(f"Error closing GPIO chip: {e}")
    
    print("Script ended")