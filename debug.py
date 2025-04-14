import numpy as np
from depthai import NNData
from depthai_sdk import OakCamera
from depthai_sdk.classes import Detections, DetectionPacket


def decode(nn_data: NNData) -> Detections:
    # vehicle-detection-0202 outputs: [image_id, label, conf, x_min, y_min, x_max, y_max]
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


def callback(packet: DetectionPacket):
    detections: Detections = packet.img_detections

    for det in detections.detections:
        print(
            f"Label: {det.label}, Confidence: {det.confidence:.2f}, "
            f"Xmin: {det.xmin:.3f}, Ymin: {det.ymin:.3f}, "
            f"Xmax: {det.xmax:.3f}, Ymax: {det.ymax:.3f}"
        )


with OakCamera() as oak:
    color = oak.create_camera('color')

    # Use detection model without tracking
    nn = oak.create_nn('vehicle-detection-0202', color, decode_fn=decode)

    # Visualize the detections and also handle in callback
    oak.callback(nn.out.main, callback)
    visualizer = oak.visualize(nn.out.passthrough, fps=True)
    visualizer.detections(
        thickness=2,
    ).text(
        auto_scale=True
    )

    oak.start(blocking=True)