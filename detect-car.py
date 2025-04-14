import numpy as np
from depthai import NNData
from depthai_sdk import OakCamera
from depthai_sdk.classes import Detections, DetectionPacket
import time
from collections import defaultdict

# Tracking configuration
MAX_MATCH_DISTANCE = 0.1  # Normalized distance threshold for matching
MIN_NEW_OBJECT_DISTANCE = 0.05  # Minimum distance to consider a new object

# Tracking state
tracked_objects = {}
next_object_id = 1
last_update_time = time.time()

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

def calculate_velocity(old_pos, new_pos, time_elapsed):
    if old_pos is None or time_elapsed == 0:
        return (0, 0)
    return ((new_pos[0] - old_pos[0]) / time_elapsed,
            (new_pos[1] - old_pos[1]) / time_elapsed)

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
                
            dist = np.sqrt((center[0]-obj_data['position'][0])**2 + 
                         (center[1]-obj_data['position'][1])**2)
            
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
                dist = np.sqrt((center[0]-obj_data['position'][0])**2 + 
                             (center[1]-obj_data['position'][1])**2)
                if dist < MIN_NEW_OBJECT_DISTANCE:
                    is_new_object = False
                    break
            
            if is_new_object:
                matches[next_object_id] = det_idx
                next_object_id += 1
    
    return matches

def callback(packet: DetectionPacket):
    global last_update_time, tracked_objects, next_object_id
    
    current_time = time.time()
    time_elapsed = current_time - last_update_time
    last_update_time = current_time
    
    detections = packet.img_detections.detections
    if not detections:
        tracked_objects.clear()
        return
    
    # Match detections to tracks
    matches = match_detections_to_tracks(detections, tracked_objects)
    
    # Update tracking state
    new_tracked_objects = {}
    for obj_id, det_idx in matches.items():
        det = detections[det_idx]
        center = calculate_center([det.xmin, det.ymin, det.xmax, det.ymax])
        
        # Get previous state if this is an existing object
        old_position = tracked_objects.get(obj_id, {}).get('position', None)
        velocity = calculate_velocity(old_position, center, time_elapsed)
        
        new_tracked_objects[obj_id] = {
            'position': center,
            'velocity': velocity,
            'last_seen': current_time
        }
        
        print(f"ID: {obj_id}, Position: ({center[0]:.3f}, {center[1]:.3f}), "
              f"Velocity: ({velocity[0]:.1f}, {velocity[1]:.1f}) px/s")
    
    tracked_objects = new_tracked_objects

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('vehicle-detection-0202', color, decode_fn=decode)
    
    oak.callback(nn.out.main, callback)
    visualizer = oak.visualize(nn.out.passthrough, fps=True)
    visualizer.detections(thickness=2).text(auto_scale=True)
    
    oak.start(blocking=True)