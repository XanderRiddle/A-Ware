import numpy as np
from depthai import NNData
from depthai_sdk import OakCamera
from depthai_sdk.classes import Detections, DetectionPacket
import time
from collections import defaultdict

# Tracking configuration
MAX_MATCH_DISTANCE = 0.1  # Normalized distance threshold for matching
MIN_NEW_OBJECT_DISTANCE = 0.05  # Minimum distance to consider a new object
DEPTH_SMOOTHING_FACTOR = 0.2  # Smoothing factor for depth measurements

# Tracking state
tracked_objects = {}
next_object_id = 1
last_update_time = time.time()
current_detections = None
current_depth_frame = None

def decode(nn_data: NNData) -> Detections:
    layer = nn_data.getFirstLayerFp16()
    results = np.array(layer).reshape((1, 1, -1, 7))
    dets = Detections(nn_data)
    for result in results[0][0]:
        conf = result[2]
        if conf > 0.5:
            label = int(result[1])
            bbox = result[3:]  # [x_min, y_min, x_max, y_max]
            dets.add(label, conf, bbox)
    return dets

def calculate_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def calculate_3d_position(bbox, depth_frame):
    """Calculate 3D position (x, y, z) from bounding box and depth frame"""
    x_center = int((bbox[0] + bbox[2]) / 2 * depth_frame.shape[1])
    y_center = int((bbox[1] + bbox[3]) / 2 * depth_frame.shape[0])
    
    # Get depth at center point
    z = depth_frame[y_center, x_center] / 1000.0  # Convert mm to meters
    
    # Calculate x and y in 3D space (simplified approximation)
    fov = 70  # Approximate FOV in degrees (adjust based on your camera)
    f_pixels = depth_frame.shape[1] / (2 * np.tan(np.radians(fov / 2)))
    
    x = (x_center - depth_frame.shape[1]/2) * z / f_pixels
    y = (y_center - depth_frame.shape[0]/2) * z / f_pixels
    
    return (x, y, z)

def calculate_velocity(old_pos, new_pos, time_elapsed):
    if old_pos is None or time_elapsed == 0:
        return (0, 0, 0)
    return ((new_pos[0] - old_pos[0]) / time_elapsed,
            (new_pos[1] - old_pos[1]) / time_elapsed,
            (new_pos[2] - old_pos[2]) / time_elapsed)

def match_detections_to_tracks(detections, tracked_objects):
    global next_object_id
    
    # Calculate centers for all detections
    detection_centers = [calculate_center([det.xmin, det.ymin, det.xmax, det.ymax]) 
                         for det in detections]
    
    # Initialize matching structures
    matches = {}
    used_tracks = set()
    used_detections = set()
    
    # First pass: find best matches for each track
    for obj_id, obj_data in tracked_objects.items():
        best_dist = MAX_MATCH_DISTANCE
        best_det_idx = -1
        
        for det_idx, center in enumerate(detection_centers):
            if det_idx in used_detections:
                continue
                
            dist = np.sqrt((center[0]-obj_data['center'][0])**2 + 
                         (center[1]-obj_data['center'][1])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_det_idx = det_idx
                
        if best_det_idx != -1:
            matches[obj_id] = best_det_idx
            used_tracks.add(obj_id)
            used_detections.add(best_det_idx)
    
    # Second pass: assign unmatched detections to new tracks
    for det_idx, center in enumerate(detection_centers):
        if det_idx not in used_detections:
            # Check if this is really a new object or just noise
            is_new_object = True
            for obj_id, obj_data in tracked_objects.items():
                if obj_id in used_tracks:
                    continue
                dist = np.sqrt((center[0]-obj_data['center'][0])**2 + 
                             (center[1]-obj_data['center'][1])**2)
                if dist < MIN_NEW_OBJECT_DISTANCE:
                    is_new_object = False
                    break
            
            if is_new_object:
                matches[next_object_id] = det_idx
                next_object_id += 1
    
    return matches

def detection_callback(packet: DetectionPacket):
    global current_detections
    current_detections = packet.img_detections.detections if hasattr(packet, 'img_detections') else None
    process_detections()

def depth_callback(packet: DetectionPacket):
    global current_depth_frame
    current_depth_frame = packet.frame  # packet.frame is already the numpy array
    process_detections()

def process_detections():
    global current_detections, current_depth_frame, tracked_objects, last_update_time
    
    if current_detections is None or current_depth_frame is None:
        return
    
    current_time = time.time()
    time_elapsed = current_time - last_update_time
    last_update_time = current_time
    
    detections = current_detections
    depth_frame = current_depth_frame
    
    if not detections:
        tracked_objects.clear()
        return
    
    # Match detections to tracks
    matches = match_detections_to_tracks(detections, tracked_objects)
    
    # Update tracking state
    new_tracked_objects = {}
    for obj_id, det_idx in matches.items():
        det = detections[det_idx]
        position = calculate_3d_position([det.xmin, det.ymin, det.xmax, det.ymax], depth_frame)
        center = calculate_center([det.xmin, det.ymin, det.xmax, det.ymax])

        # Get previous state if this is an existing object
        old_position = tracked_objects.get(obj_id, {}).get('position', None)
        velocity = calculate_velocity(old_position, position, time_elapsed)
        
        # Apply smoothing to depth (z-coordinate)
        if obj_id in tracked_objects:
            prev_z = tracked_objects[obj_id]['position'][2]
            smoothed_z = prev_z + DEPTH_SMOOTHING_FACTOR * (position[2] - prev_z)
            position = (position[0], position[1], smoothed_z)
        
        new_tracked_objects[obj_id] = {
            'center': center,
            'position': position,
            'velocity': velocity,
            'last_seen': current_time
        }
        
        print(f"ID: {obj_id}, Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m, "
              f"Velocity: ({velocity[0]:.1f}, {velocity[1]:.1f}, {velocity[2]:.1f}) m/s")
    
    tracked_objects = new_tracked_objects

with OakCamera() as oak:
    color = oak.create_camera('color')
    stereo = oak.create_stereo('800p')  # Create stereo depth camera
    nn = oak.create_nn('vehicle-detection-0202', color, decode_fn=decode)
    
    # Configure stereo depth
    stereo.config_stereo(align=color)  # Align depth to color camera
    
    # Set up separate callbacks
    oak.callback(nn.out.passthrough, detection_callback)
    oak.callback(stereo.out.depth, depth_callback)
    
    # Visualize
    visualizer = oak.visualize(nn.out.passthrough, fps=True)
    visualizer.detections(thickness=2).text(auto_scale=True)
    
    oak.start(blocking=True)