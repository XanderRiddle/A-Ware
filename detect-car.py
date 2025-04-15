import numpy as np
import time
from depthai import NNData
from depthai_sdk import OakCamera
from depthai_sdk.classes import Detections, DetectionPacket

# --- Config ---
MAX_MATCH_DIST = 0.1
MIN_NEW_OBJ_DIST = 0.05
DEPTH_SMOOTHING = 0.2

# --- State ---
tracked = {}
next_id = 1
last_time = time.time()
detections, depth_frame = None, None

# --- Utility Functions ---
def decode(nn_data: NNData) -> Detections:
    dets = Detections(nn_data)
    for det in np.array(nn_data.getFirstLayerFp16()).reshape((1, 1, -1, 7))[0][0]:
        if det[2] > 0.5:
            dets.add(int(det[1]), det[2], det[3:])
    return dets

def center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def get_3d_pos(bbox, depth):
    h, w = depth.shape
    cx, cy = int(center(bbox)[0] * w), int(center(bbox)[1] * h)
    z = depth[cy, cx] / 1000.0
    fov = 70
    f = w / (2 * np.tan(np.radians(fov / 2)))
    x = (cx - w/2) * z / f
    y = (cy - h/2) * z / f
    return (x, y, z)

def velocity(old, new, dt):
    return tuple((n - o) / dt if dt > 0 else 0 for o, n in zip(old or (0, 0, 0), new))

def match_tracks(dets, tracks):
    global next_id
    det_centers = [center([d.xmin, d.ymin, d.xmax, d.ymax]) for d in dets]
    matches, used_dets = {}, set()

    for tid, data in tracks.items():
        dists = [np.linalg.norm(np.subtract(c, data['center'])) for c in det_centers]
        best_idx = np.argmin(dists) if dists else -1
        if best_idx != -1 and dists[best_idx] < MAX_MATCH_DIST and best_idx not in used_dets:
            matches[tid] = best_idx
            used_dets.add(best_idx)

    for i, c in enumerate(det_centers):
        if i in used_dets: continue
        if all(np.linalg.norm(np.subtract(c, t['center'])) >= MIN_NEW_OBJ_DIST for t in tracks.values()):
            matches[next_id] = i
            next_id += 1
    return matches

# --- Callbacks ---
def detection_cb(pkt: DetectionPacket):
    global detections
    detections = pkt.img_detections.detections if hasattr(pkt, 'img_detections') else None
    update_tracks()

def depth_cb(pkt: DetectionPacket):
    global depth_frame
    depth_frame = pkt.frame
    update_tracks()

def update_tracks():
    global detections, depth_frame, tracked, last_time
    if detections is None or depth_frame is None: return
    dt, last_time = time.time() - last_time, time.time()

    if not detections:
        tracked.clear()
        return

    matches = match_tracks(detections, tracked)
    updated = {}
    for tid, idx in matches.items():
        det = detections[idx]
        bbox = [det.xmin, det.ymin, det.xmax, det.ymax]
        pos = get_3d_pos(bbox, depth_frame)
        cent = center(bbox)
        prev = tracked.get(tid, {})
        vel = velocity(prev.get('position'), pos, dt)

        if 'position' in prev:
            pos = (pos[0], pos[1], prev['position'][2] + DEPTH_SMOOTHING * (pos[2] - prev['position'][2]))

        updated[tid] = {'center': cent, 'position': pos, 'velocity': vel}
        print(f"ID {tid}: Pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})m Vel=({vel[0]:.1f},{vel[1]:.1f},{vel[2]:.1f})m/s")

    tracked = updated

# --- Main ---
with OakCamera() as oak:
    color = oak.create_camera('color')
    stereo = oak.create_stereo('800p')
    nn = oak.create_nn('vehicle-detection-0202', color, decode_fn=decode)
    stereo.config_stereo(align=color)

    oak.callback(nn.out.passthrough, detection_cb)
    oak.callback(stereo.out.depth, depth_cb)

    oak.visualize(nn.out.passthrough, fps=True).detections(thickness=2).text(auto_scale=True)
    oak.start(blocking=True)
