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
chip = gpiod.Chip('/dev/gpiochip0')

# Define the GPIO pins
pins = {
    'Left': 0,    # Replace with actual pin numbers
    'Middle': 4,  # Replace with actual pin numbers
    'Right': 3    # Replace with actual pin numbers
}

# Set pins as outputs
outputs = {key: chip.get_line(pin) for key, pin in pins.items()}
for line in outputs.values():
    line.request(consumer='pwm', type=gpiod.LINE_REQ_DIR_OUT)

# PWM Settings
def set_pwm_duty_cycle(pin, duty_cycle):
    """
    Set the PWM duty cycle for the specified pin.
    duty_cycle: The duty cycle in percentage (0 - 100).
    """
    period = 10000  # 10 ms period for PWM
    pulse_width = int((duty_cycle / 100) * period)
    chip.set_line_value(pin, 1)  # Activate pin
    time.sleep(pulse_width / 1000000)  # Sleep for pulse width in seconds
    chip.set_line_value(pin, 0)  # Deactivate pin
    time.sleep((period - pulse_width) / 1000000)  # Sleep for the rest of the period

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    detectionsQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        detections = detectionsQueue.get().detections
        depthFrame = depthQueue.get().getFrame()
        depthFrame = np.array(depthFrame, dtype=np.uint16)  # Convert depth frame to NumPy array

        depthHeight = depthFrame.shape[0]
        depthWidth = depthFrame.shape[1]

        object_list = []
        if detections:
            for detection in detections:
                x1, y1, x2, y2 = int(detection.xmin * depthWidth), int(detection.ymin * depthHeight), \
                                int(detection.xmax * depthWidth), int(detection.ymax * depthHeight)
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

                object_list.append({"label": label, "distance_mm": distance, "quadrant": quadrant})
                
                # Set PWM duty cycle based on the detected distance
                if distance is not None:
                    duty_cycle = max(0, min(100, ((10000 - distance) / 9500) * 100))
                    set_pwm_duty_cycle(pins[quadrant], duty_cycle)
                else:
                    set_pwm_duty_cycle(pins[quadrant], 0)

        print("Detected Objects:", object_list)
        
        time.sleep(0.5)  # Print every 0.5 seconds