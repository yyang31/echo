import cv2
import subprocess
import glob
import sys
import time
import numpy as np
from pathlib import Path

from ultralytics import YOLO

# --- CONFIGURATION ---
BASELINE = 6.5  # <--- MEASURE THE DISTANCE BETWEEN YOUR CAMERAS IN CM
MODEL_PATH = Path("yolo/runs/door_window_stair_yolo/weights/best.pt")
CONFIDENCE_THRESHOLD = 0.35
YOLO_IMGSZ = 640
INFER_EVERY_N_FRAMES = 2  # Increase to improve FPS (e.g., 2 or 3)

# Fixed depth visualization range for stable colors across frames
DEPTH_MIN_CM = 30.0
DEPTH_MAX_CM = 400.0

# Human-friendly distance bands
DEPTH_NEAR_CM = 100.0
DEPTH_MID_CM = 250.0

def get_real_camera_indexes():
    real_indexes = []
    devices = glob.glob('/dev/video*')
    for dev in sorted(devices):
        index = int(dev.replace('/dev/video', ''))
        try:
            output = subprocess.check_output(['v4l2-ctl', '-d', dev, '--list-formats'], stderr=subprocess.STDOUT).decode('utf-8').lower()
            if any(fmt in output for fmt in ['yuyv', 'mjpg', 'nv12', 'h264']):
                real_indexes.append(index)
        except FileNotFoundError:
            print("Warning: v4l2-ctl not found. Install v4l-utils to auto-detect cameras.")
            return real_indexes
        except subprocess.CalledProcessError as exc:
            print(f"Warning: camera probe failed for {dev}: {exc}")
    return real_indexes


def clamp(value, low, high):
    return max(low, min(high, value))


def depth_band_label(depth_cm: float) -> str:
    if depth_cm < DEPTH_NEAR_CM:
        return "Near"
    if depth_cm < DEPTH_MID_CM:
        return "Mid"
    return "Far"


def load_model(model_path: Path):
    if not model_path.exists():
        print(f"Error: model not found at '{model_path}'.")
        sys.exit(1)
    return YOLO(str(model_path))

def main():
    left_window = "Left Camera"
    depth_window = "Real-Time Depth (Heatmap)"
    yolo_window = "YOLO Predictions"

    model = load_model(MODEL_PATH)

    # 1. Load Calibration Data
    try:
        data = np.load("camera_params.npz")
        focal_raw = data['focal_length']
        focal_length = float(np.asarray(focal_raw).reshape(-1)[0])
        print(f"Loaded Focal Length: {focal_length:.2f}")
    except FileNotFoundError:
        print("Error: 'camera_params.npz' not found. Run your calibration script first!")
        sys.exit(1)
    except (KeyError, ValueError, TypeError, IndexError):
        print("Error: invalid or missing 'focal_length' in camera_params.npz")
        sys.exit(1)

    valid_cameras = get_real_camera_indexes()
    if len(valid_cameras) < 2:
        print("Error: Two cameras required.")
        sys.exit(1)

    cap_1 = cv2.VideoCapture(valid_cameras[0])
    cap_2 = cv2.VideoCapture(valid_cameras[1])

    if not cap_1.isOpened() or not cap_2.isOpened():
        print(f"Error: failed to open cameras {valid_cameras[0]} and/or {valid_cameras[1]}")
        cap_1.release()
        cap_2.release()
        sys.exit(1)

    # Initialize Stereo Matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*5, 
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2
    )

    print("Running Depth + YOLO stream.")
    print("Press 'y' to toggle YOLO inference on/off.")
    print("Press 'd' to toggle disparity/depth heatmap on/off.")
    print("Press 'q' to quit.")

    frame_count = 0
    last_annotated = None
    yolo_enabled = False
    disparity_enabled = True
    yolo_window_visible = False
    frame_size_warning_shown = False

    last_yolo_error_log_ts = 0.0
    yolo_error_log_interval_sec = 1.0

    fps = 0.0
    fps_frame_counter = 0
    fps_last_ts = time.time()

    infer_stride = max(1, int(INFER_EVERY_N_FRAMES))
    if INFER_EVERY_N_FRAMES < 1:
        print("Warning: INFER_EVERY_N_FRAMES must be >= 1. Using 1.")

    while True:
        success_1, frame_1 = cap_1.read()
        success_2, frame_2 = cap_2.read()

        if not success_1 or not success_2:
            print("Error: failed to read frame(s) from camera stream.")
            break

        if frame_1.shape[:2] != frame_2.shape[:2]:
            if not frame_size_warning_shown:
                print(
                    f"Warning: camera frame size mismatch ({frame_1.shape[1]}x{frame_1.shape[0]} vs "
                    f"{frame_2.shape[1]}x{frame_2.shape[0]}). Resizing right frame to match left frame."
                )
                frame_size_warning_shown = True
            frame_2 = cv2.resize(frame_2, (frame_1.shape[1], frame_1.shape[0]), interpolation=cv2.INTER_LINEAR)

        need_depth = disparity_enabled or yolo_enabled
        depth_map = None
        depth_colormap = None

        if need_depth:
            gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
            gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

            # 2. Compute Disparity
            # SGBM returns disparity * 16, so we divide to get true pixel disparity
            disparity = stereo.compute(gray_1, gray_2).astype(np.float32) / 16.0

            # 3. Calculate Depth
            # Avoid division by zero
            disparity[disparity <= 0] = 0.1
            depth_map = (focal_length * BASELINE) / disparity

            if disparity_enabled:
                # 4. Fixed-range depth visualization for consistent color meaning.
                depth_clipped = np.clip(depth_map, DEPTH_MIN_CM, DEPTH_MAX_CM)
                depth_norm = ((depth_clipped - DEPTH_MIN_CM) / (DEPTH_MAX_CM - DEPTH_MIN_CM) * 255.0).astype(np.uint8)
                depth_norm = 255 - depth_norm  # Closer = hotter colors.
                depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

                center_y = depth_map.shape[0] // 2
                center_x = depth_map.shape[1] // 2
                center_depth_cm = float(depth_map[center_y, center_x])
                center_band = depth_band_label(center_depth_cm)

                legend_lines = [
                    f"Range: {DEPTH_MIN_CM:.0f}-{DEPTH_MAX_CM:.0f} cm",
                    f"Near < {DEPTH_NEAR_CM:.0f} cm",
                    f"Mid  {DEPTH_NEAR_CM:.0f}-{DEPTH_MID_CM:.0f} cm",
                    f"Far  >= {DEPTH_MID_CM:.0f} cm",
                    f"Center: {center_depth_cm:.1f} cm ({center_band})",
                ]

                for idx, text in enumerate(legend_lines):
                    y = 25 + idx * 22
                    cv2.putText(
                        depth_colormap,
                        text,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

        frame_count += 1
        annotated_frame = frame_1
        do_infer = yolo_enabled and (frame_count % infer_stride) == 0

        if yolo_enabled:
            if do_infer:
                try:
                    results = model.predict(
                        source=frame_1,
                        conf=CONFIDENCE_THRESHOLD,
                        imgsz=YOLO_IMGSZ,
                        verbose=False,
                    )
                    result = results[0]
                    annotated_frame = result.plot()

                    if result.boxes is not None and len(result.boxes) > 0:
                        names = result.names
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cls_id = int(box.cls[0].item())

                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            cx = clamp(cx, 0, depth_map.shape[1] - 1)
                            cy = clamp(cy, 0, depth_map.shape[0] - 1)
                            depth_cm = float(depth_map[cy, cx])

                            label = f"{names[cls_id]}: {depth_cm:.1f} cm"
                            cv2.putText(
                                annotated_frame,
                                label,
                                (int(x1), max(int(y1) - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )

                    last_annotated = annotated_frame
                except Exception as exc:
                    now_ts = time.time()
                    if (now_ts - last_yolo_error_log_ts) >= yolo_error_log_interval_sec:
                        print(f"Warning: YOLO inference failed: {exc}")
                        last_yolo_error_log_ts = now_ts
                    last_annotated = None
                    annotated_frame = frame_1
            elif last_annotated is not None:
                annotated_frame = last_annotated
        else:
            last_annotated = None

        if disparity_enabled and depth_colormap is not None:
            cv2.imshow(depth_window, depth_colormap)

        fps_frame_counter += 1
        now_ts = time.time()
        elapsed = now_ts - fps_last_ts
        if elapsed >= 1.0:
            fps = fps_frame_counter / elapsed
            fps_frame_counter = 0
            fps_last_ts = now_ts

        left_display = frame_1.copy()
        cv2.putText(
            left_display,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow(left_window, left_display)

        if yolo_enabled:
            cv2.imshow(yolo_window, annotated_frame)
            yolo_window_visible = True
        elif yolo_window_visible:
            cv2.destroyWindow(yolo_window)
            yolo_window_visible = False

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key in (ord('y'), ord('Y')):
            yolo_enabled = not yolo_enabled
            state = "enabled" if yolo_enabled else "disabled"
            print(f"YOLO inference {state}")
        if key in (ord('d'), ord('D')):
            disparity_enabled = not disparity_enabled
            if disparity_enabled:
                print("Disparity/depth heatmap enabled")
            else:
                cv2.destroyWindow(depth_window)
                print("Disparity/depth heatmap disabled")

    cap_1.release()
    cap_2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()