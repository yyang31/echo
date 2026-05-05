import cv2
import subprocess
import glob
import sys
import numpy as np

# --- CONFIGURATION ---
BASELINE = 6.5  # <--- MEASURE THE DISTANCE BETWEEN YOUR CAMERAS IN CM

def get_real_camera_indexes():
    real_indexes = []
    devices = glob.glob('/dev/video*')
    for dev in sorted(devices):
        index = int(dev.replace('/dev/video', ''))
        try:
            output = subprocess.check_output(['v4l2-ctl', '-d', dev, '--list-formats'], stderr=subprocess.STDOUT).decode('utf-8').lower()
            if any(fmt in output for fmt in ['yuyv', 'mjpg', 'nv12', 'h264']):
                real_indexes.append(index)
        except Exception: pass
    return real_indexes

def main():
    # 1. Load Calibration Data
    try:
        data = np.load("camera_params.npz")
        focal_length = data['focal_length']
        print(f"Loaded Focal Length: {focal_length:.2f}")
    except FileNotFoundError:
        print("Error: 'camera_params.npz' not found. Run your calibration script first!")
        sys.exit(1)

    valid_cameras = get_real_camera_indexes()
    if len(valid_cameras) < 2:
        print("Error: Two cameras required.")
        sys.exit(1)

    cap_1 = cv2.VideoCapture(valid_cameras[0])
    cap_2 = cv2.VideoCapture(valid_cameras[1])

    # Initialize Stereo Matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*5, 
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2
    )

    print("Running Depth Stream. Press 'q' to quit.")

    while True:
        success_1, frame_1 = cap_1.read()
        success_2, frame_2 = cap_2.read()

        if not success_1 or not success_2:
            break

        gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

        # 2. Compute Disparity
        # SGBM returns disparity * 16, so we divide to get true pixel disparity
        disparity = stereo.compute(gray_1, gray_2).astype(np.float32) / 16.0

        # 3. Calculate Depth
        # Avoid division by zero
        disparity[disparity <= 0] = 0.1
        depth_map = (focal_length * BASELINE) / disparity

        # 4. Visualization
        # Convert depth to a 0-255 image for display
        depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # Apply a colormap so it's easier to see (Optional)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        cv2.imshow("Real-Time Depth (Heatmap)", depth_colormap)
        cv2.imshow("Left Camera", frame_1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_1.release()
    cap_2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()