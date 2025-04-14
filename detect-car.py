from depthai_sdk import OakCamera

with OakCamera() as oak:  # Remove 'replay' to use live feed
    # Create color camera (live feed)
    color = oak.create_camera('color')

    # Download & run pretrained vehicle detection model with tracking
    nn = oak.create_nn('vehicle-detection-0202', color, tracker=True)

    # Visualize tracklets with FPS and recording
    visualizer = oak.visualize(nn.out.tracker, fps=True, record_path='./car_tracking.avi')
    visualizer.tracking(line_thickness=5).text(auto_scale=True)

    # Start the app in blocking mode
    oak.start(blocking=True)