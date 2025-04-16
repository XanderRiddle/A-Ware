# Car Detection
# A-Ware
# 4-14-25

send_to_GPIO = True

import numpy as np
import time
from depthai import NNData, CameraBoardSocket, ImgDetection
from depthai_sdk import OakCamera
from depthai_sdk.classes import Detections, DetectionPacket
if send_to_GPIO:
    import RPi.GPIO as GPIO

# --- Config ---
CONFIDENCE_THRESHOLD = 0.5 # Required confidence for object detection
MAX_MATCH_DIST = 0.3 # Maximum distance for a detection to be matched to an existing track
MIN_NEW_OBJ_DIST = 0.2 # Minimum distance for a new detection to be separate from existing tracks
POS_SMOOTHING = 0.2 # Smooths position readings, 1 = no smoothing
VEL_SMOOTHING = 0.2 # Smooths velocity readings, 1 = no smoothing
TIMEOUT_LENGTH = 0.5 # Time without detections before GPIO output resets

# --- GPIO pins for PWM output ---
if send_to_GPIO:
    timeout = 0

    LEFT_PWM_PIN = 18
    MIDDLE_PWM_PIN = 19
    RIGHT_PWM_PIN = 20

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LEFT_PWM_PIN, GPIO.OUT)
    GPIO.setup(MIDDLE_PWM_PIN, GPIO.OUT)
    GPIO.setup(RIGHT_PWM_PIN, GPIO.OUT)

    left_pwm = GPIO.PWM(LEFT_PWM_PIN, 1000)
    middle_pwm = GPIO.PWM(MIDDLE_PWM_PIN, 1000)
    right_pwm = GPIO.PWM(RIGHT_PWM_PIN, 1000)

    left_intensity = 0
    middle_intensity = 0
    right_intensity = 0

    left_pwm.start(0)
    middle_pwm.start(0)
    right_pwm.start(0)

# --- Camera Intrinsic Analysis ---
oak = OakCamera()

# width, height = [4000, 3040] # Resolution best for Oak-D Pro Wide FOV?
width, height = [4180, 3120] # Resolution best for Oak-D Lite FOV

calibration = oak.device.readCalibration()
intrinsics = calibration.getCameraIntrinsics(CameraBoardSocket.CAM_A, width, height)

fl_x = intrinsics[0][0] # Focal Length X
fl_y = intrinsics[1][1] # Focal Length Y
oc_x = intrinsics[0][2] # Optical Center X
oc_y = intrinsics[1][2] # Optcial Center Y
fov_x = 2 * np.degrees(np.arctan2(width / 2, fl_x))
fov_y = 2 * np.degrees(np.arctan2(height / 2, fl_y))

# --- State ---
tracked = {}
next_id = 1
last_time = time.time()
detections = None
depth_frame = None

# --- Utility Functions ---
def map_distance_to_pwm(distance):
    """
    An inverse function of camera-detected distance to PWM haptic duty cycle.

    distance: Distance to a point in mm ranging from 500 to 10000.

    Returns duty cycle percentage.
    """
    if distance is None or distance > 10000:
        return 0  # No signal if out of range
    elif distance < 500:
        return 100  # Max signal if too close
    return int((100 - ((distance - 500) / (10000 - 500) * 100)))

def decode(nn_data: NNData) -> Detections:
    """
    Function to interpret the output data of our neural network.
     - nn_data: Neural network data

    Reshapes data in lists of 7 values: image ID, class ID, confidence, xmin, ymin, xmax, ymax
    Returns a 'Detections' list of detections above the specified confidence threshold.
    """
    # dets = Detections(nn_data)
    # dets.detections = [d for d in dets.detections if d.confidence > CONFIDENCE_THRESHOLD]
    # for d in dets.detections:
    #     print(f"Detected: {d.label} with confidence {d.confidence:.2f}")
    # return dets
    results = np.array(nn_data.getFirstLayerFp16()).reshape((1, 1, -1, 7))
    dets = Detections(nn_data)
    for result in results[0][0]:
        if result[2] > CONFIDENCE_THRESHOLD:
            label = int(result[1])
            conf = result[2]
            bbox = result[3:]
            det = ImgDetection()
            det.confidence = conf
            det.label = label
            det.xmin = bbox[0]
            det.ymin = bbox[1]
            det.xmax = bbox[2]
            det.ymax = bbox[3]
            dets.detections.append(det)

    return dets

def center(b):
    """
    Finds center of a box.
     - b: bounding box with format [x1, y1, x2, y2]

    Returns the center (float, float).
    """
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

def get_3d_pos(bbox, depth, oc_x, oc_y, fl_x, fl_y):
    """
    Function to map an object to 3D.
     - bbox: bounding box [x1, y1, x2, y2]
     - depth: detected depth grid where each coordinate corresponds to a depth in mm
     - oc_x/y: optical center X/Y
     - fl_x/y: focal length X/Y

    Returns 3D float coordinates in meters relative to camera.
    """
    h, w = depth.shape
    cx, cy = center(bbox) # 2D decimal position
    cx = int(cx * w) # Pixel center X
    cy = int(cy * h) # Pixel center Y
    z = depth[cy, cx] / 1000.0 # Depth in meters

    if z == 0 or np.isnan(z) or np.isinf(z): # Invalid depth
        return (0, 0, 0)

    x = (cx - oc_x) * z / fl_x
    y = (cy - oc_y) * z / fl_y
    return (x, y, z)

def velocity(old, new, dt):
    """
    Function to find velocity given two 3D coordinates and change in time.
     - old: old position
     - new: new position
     - dt: change in time

    Returns (dx, dy, dz) per unit time.
    """
    if dt <= 0:
        return (0, 0, 0)
    return ((new[0] - old[0]) / dt, (new[1] - old[1]) / dt, (new[2] - old[2]) / dt)

def match_tracks(dets, tracks):
    """
    Function to match detections to tracks. This creates a known trajectory for objects through space.
     - dets: detected objects
     - tracks: existing tracks
    
    Returns the list of matches.
    """
    global next_id
    matches = {}
    used_dets = []

    track_centers = np.array([track['center'] for track in tracks.values()])
    det_centers = np.array([center([d.xmin, d.ymin, d.xmax, d.ymax]) for d in dets])

    if len(track_centers) > 0 and len(det_centers) > 0:
        dists = np.linalg.norm(track_centers[:, None, :] - det_centers[None, :, :], axis = 2) # ndarray (tracks, dets)

        track_ids = list(tracks.keys())
        for i in range(dists.shape[0]):
            j = np.argmin(dists[i])
            if dists[i, j] < MAX_MATCH_DIST:
                matches[track_ids[i]] = j
                used_dets.append(j)
                dists[i, :] = np.inf
                dists[:, j] = np.inf

    for i, c in enumerate(det_centers):
        if i in used_dets:
            continue
        if all(np.linalg.norm(np.subtract(c, t['center'])) >= MIN_NEW_OBJ_DIST for t in tracks.values()):
            matches[next_id] = i
            next_id += 1
    return matches

# --- Callbacks ---
def detection_cb(pkt: DetectionPacket):
    global detections
    detections = pkt.img_detections.detections
    update_tracks()

def depth_cb(pkt: DetectionPacket):
    global depth_frame
    depth_frame = pkt.frame
    update_tracks()

def update_tracks():
    global detections, depth_frame, tracked, last_time, left_intensity, middle_intensity, right_intensity, timeout

    if not detections or depth_frame is None:
        print("No detections or depth frame available.")
        tracked.clear() # Clear the tracked objects if detections are empty
        return

    dt = time.time() - last_time
    last_time = time.time()

    matches = match_tracks(detections, tracked)
    updated = {} # Dictionary to store updated tracked objects

    # Process each matched track
    for tid, didx in matches.items():
        det = detections[didx]
        oldCxy, oldPos, oldVel = tracked.get(tid, {'center': (0, 0, 0), 'position': (0, 0, 0), 'velocity': (0, 0, 0)}).values()
        bbox = [det.xmin, det.ymin, det.xmax, det.ymax]
        cxy = center(bbox)

        pos = get_3d_pos(bbox, depth_frame, oc_x, oc_y, fl_x, fl_y)
        print(pos)
        vel = (0, 0, 0)

        if pos == (0, 0, 0):
            if oldPos == (0, 0, 0) or oldPos == None:
                print(f"Skipping track ID {tid} due to invalid position and no old data.")
                continue
            elif type(oldPos) == tuple:
                pos = oldPos # Use the old position if no new position found
        else:
            vel = velocity(oldPos, pos, dt)
            # Apply smoothing for position and velocity
            pos = tuple(oldPos[i] + POS_SMOOTHING * (pos[i] - oldPos[i]) for i in range(3))
            vel = tuple(oldVel[i] + VEL_SMOOTHING * (vel[i] - oldVel[i]) for i in range(3))

        updated[tid] = {'center': cxy, 'position': pos, 'velocity': vel}

        print(f"ID {tid}: Pos=({pos[0]},{pos[1]},{pos[2]})m Vel=({vel[0]},{vel[1]},{vel[2]})m/s")

        if send_to_GPIO and 0 <= cxy[0] <= width and 0 <= cxy[1] <= height:
            intensity = map_distance_to_pwm(pos[2])  # Map position to PWM intensity
            if cxy[0] < width / 3:
                left_intensity = max(left_intensity, intensity)
            elif cxy[0] < 2 * width / 3:
                middle_intensity = max(middle_intensity, intensity)
            else:
                right_intensity = max(right_intensity, intensity)
            timeout = time.time()  # Reset the timeout counter

    tracked = updated

    # If no new detections were found (timeout), reset PWM intensities
    if time.time() - timeout >= TIMEOUT_LENGTH and send_to_GPIO:
        left_intensity = 0
        middle_intensity = 0
        right_intensity = 0

# --- Main ---
# Run Object Detection
color = oak.create_camera("color", str(width) + "p")
stereo = oak.create_stereo("800p")
nn = oak.create_nn("vehicle-detection-0202", color, decode_fn=decode)
stereo.config_stereo(align=color)

oak.callback(nn.out.main, detection_cb)
oak.callback(stereo.out.depth, depth_cb)

# oak.visualize(nn, fps=True).detections(thickness=2).text(auto_scale=True)
oak.start(blocking=False)

while oak.running():
    if send_to_GPIO: # Reset GPIO Duty Cycles
        if time.time() - timeout >= TIMEOUT_LENGTH:
            left_intensity = 0
            middle_intensity = 0
            right_intensity = 0

        oak.poll()

        left_pwm.ChangeDutyCycle(left_intensity)
        middle_pwm.ChangeDutyCycle(middle_intensity)
        right_pwm.ChangeDutyCycle(right_intensity)
        # print("Left PWM:", left_intensity, "Middle PWM:", middle_intensity, "Right PWM:", right_intensity)
    else:
        oak.poll()

oak.close()
if send_to_GPIO:
    GPIO.cleanup()