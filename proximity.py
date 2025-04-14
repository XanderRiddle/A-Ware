import depthai as dai
import time
import numpy as np
# import RPi.GPIO as GPIO
from util import *

# Define GPIO pins for PWM output
LEFT_PWM_PIN = 18
MIDDLE_PWM_PIN = 19
RIGHT_PWM_PIN = 20

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(LEFT_PWM_PIN, GPIO.OUT)
GPIO.setup(MIDDLE_PWM_PIN, GPIO.OUT)
GPIO.setup(RIGHT_PWM_PIN, GPIO.OUT)

# Initialize PWM with a frequency of 1000Hz
left_pwm = GPIO.PWM(LEFT_PWM_PIN, 1000)
middle_pwm = GPIO.PWM(MIDDLE_PWM_PIN, 1000)
right_pwm = GPIO.PWM(RIGHT_PWM_PIN, 1000)

left_pwm.start(0)
middle_pwm.start(0)
right_pwm.start(0)

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

        # Reset PWM signals
        left_intensity = 0
        middle_intensity = 0
        right_intensity = 0

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

                intensity = map_distance_to_pwm(distance)

                if cx < depthWidth / 3:
                    left_intensity = max(left_intensity, intensity)
                elif cx < 2 * depthWidth / 3:
                    middle_intensity = max(middle_intensity, intensity)
                else:
                    right_intensity = max(right_intensity, intensity)

        left_pwm.ChangeDutyCycle(left_intensity)
        middle_pwm.ChangeDutyCycle(middle_intensity)
        right_pwm.ChangeDutyCycle(right_intensity)

        print("Left PWM:", left_intensity, "Middle PWM:", middle_intensity, "Right PWM:", right_intensity)
        
        time.sleep(0.5)  # Print every 0.5 seconds

# Cleanup GPIO on exit
GPIO.cleanup()
