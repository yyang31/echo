import cv2
import sys
import time
import numpy as np
from pathlib import Path

from ultralytics import YOLO
import threading
import math

import simpleaudio as sa

# --- CONFIGURATION ---
BASELINE = 6.5  # <--- MEASURE THE DISTANCE BETWEEN YOUR CAMERAS IN CM
MODEL_PATH = Path("yolo/runs/door_window_stair_yolo/weights/best.pt")
CONFIDENCE_THRESHOLD = 0.35
YOLO_IMGSZ = 640
INFER_EVERY_N_FRAMES = 2  # Increase to improve FPS (e.g., 2 or 3)
LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = 2

# Fixed depth visualization range for stable colors across frames
DEPTH_MIN_CM = 30.0
DEPTH_MAX_CM = 400.0

# Human-friendly distance bands
DEPTH_NEAR_CM = 100.0
DEPTH_MID_CM = 250.0

# Sound runtime state (module-level so thread and main can share)
sound_enabled = False
sound_thread = None
sound_stop_event = None
latest_center_depth = None


def depth_to_sound_params(depth):
    """Map a depth (cm) to (frequency Hz, amplitude 0..1, duration s, period s).

    Closer -> higher pitch and louder; distances are clamped to DEPTH_MIN_CM..DEPTH_MAX_CM.
    """
    if depth is None:
        return None
    try:
        d = float(depth)
    except Exception:
        return None

    if not np.isfinite(d):
        return None

    d = float(np.clip(d, DEPTH_MIN_CM, DEPTH_MAX_CM))

    # Frequency mapping: near -> high, far -> low (lowered for less shrill beeps)
    F_NEAR_HZ = 900.0
    F_FAR_HZ = 300.0
    t = (d - DEPTH_MIN_CM) / (DEPTH_MAX_CM - DEPTH_MIN_CM)
    freq = F_NEAR_HZ + (F_FAR_HZ - F_NEAR_HZ) * t

    # Amplitude mapping (safety caps)
    A_NEAR = 0.6
    A_FAR = 0.05
    amp = float(A_NEAR + (A_FAR - A_NEAR) * t)
    amp = float(np.clip(amp, 0.0, 1.0))

    # Period / duration mapping: closer -> more frequent beeps
    MIN_PERIOD = 0.25
    MAX_PERIOD = 1.0
    period = float(MIN_PERIOD + (MAX_PERIOD - MIN_PERIOD) * t)
    duration = min(0.12, period * 0.6)

    return freq, amp, duration, period


def sound_thread_func(stop_event):
    """Background thread that emits discrete beeps based on `latest_center_depth`.

    The thread exits when `stop_event` is set.
    """
    sample_rate = 44100

    while not stop_event.is_set():
        depth = latest_center_depth
        params = depth_to_sound_params(depth)
        if params is None:
            # nothing to play right now
            stop_event.wait(0.2)
            continue

        freq, amp, duration, period = params

        # Generate sine wave
        samples_count = max(1, int(sample_rate * duration))
        t = np.linspace(0, duration, samples_count, False)
        waveform = np.sin(2 * math.pi * freq * t)

        # Apply simple fade-in/out envelope to reduce clicks
        env_len = int(0.01 * sample_rate)
        if env_len * 2 < samples_count:
            env = np.ones_like(waveform)
            env[:env_len] = np.linspace(0.0, 1.0, env_len)
            env[-env_len:] = np.linspace(1.0, 0.0, env_len)
            waveform *= env

        # Scale to 16-bit signed integers
        max_amp = 2 ** 15 - 1
        audio = (waveform * amp * max_amp).astype(np.int16)

        try:
            play_obj = sa.play_buffer(audio.tobytes(), 1, 2, sample_rate)
            # wait for tone to finish or exit early if stop_event set
            while play_obj.is_playing():
                if stop_event.wait(0.05):
                    try:
                        play_obj.stop()
                    except Exception:
                        pass
                    return
                # otherwise loop until finished
        except Exception as exc:
            print(f"Warning: sound playback failed: {exc}")
            return

        # wait until next beep, but exit early if stop_event
        remaining = max(0.0, period - duration)
        stop_event.wait(remaining)


def compute_depth_map(frame_1, frame_2, stereo, focal_length, baseline,
                      use_rectification, rect_maps, depth_history,
                      disparity_enabled, stereo_flip, last_center_depth,
                      last_center_disparity, ORDER_CHECK_INTERVAL,
                      MIN_VALID_PIXELS_FOR_OK, frame_count):
    """
    Compute disparity and depth map from stereo frame pair.
    Returns: (depth_map, depth_colormap, depth_history, last_center_depth, last_center_disparity)
    """
    # Apply rectification maps if available
    if use_rectification:
        m1x, m1y, m2x, m2y = rect_maps
        frame_1 = cv2.remap(frame_1, m1x, m1y, interpolation=cv2.INTER_LINEAR)
        frame_2 = cv2.remap(frame_2, m2x, m2y, interpolation=cv2.INTER_LINEAR)

    gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    # Compute Disparity
    if stereo_flip:
        disparity = stereo.compute(gray_2, gray_1).astype(np.float32) / 16.0
    else:
        disparity = stereo.compute(gray_1, gray_2).astype(np.float32) / 16.0

    valid_mask = disparity > 0.0
    valid_count = int(np.sum(valid_mask))

    # Periodic re-check of best camera ordering
    do_order_check = (frame_count % ORDER_CHECK_INTERVAL) == 0 or valid_count < MIN_VALID_PIXELS_FOR_OK
    if do_order_check:
        try:
            h, w = gray_1.shape
            cy = h // 2
            cx = w // 2
            roi_h = max(32, h // 4)
            roi_w = max(32, w // 4)
            y0 = max(0, cy - roi_h // 2)
            y1 = min(h, cy + roi_h // 2)
            x0 = max(0, cx - roi_w // 2)
            x1 = min(w, cx + roi_w // 2)

            g1_roi = gray_1[y0:y1, x0:x1]
            g2_roi = gray_2[y0:y1, x0:x1]

            small1 = cv2.resize(g1_roi, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            small2 = cv2.resize(g2_roi, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            disp_a = stereo.compute(small1, small2).astype(np.float32) / 16.0
            disp_b = stereo.compute(small2, small1).astype(np.float32) / 16.0

            valid_a = int(np.sum(disp_a > 0.0))
            valid_b = int(np.sum(disp_b > 0.0))

            if valid_b > valid_a * 1.3 and not stereo_flip:
                stereo_flip = True
                print(f"Auto-switch: using (cap_2, cap_1) - center valid {valid_b} vs {valid_a}")
            elif valid_a > valid_b * 1.3 and stereo_flip:
                stereo_flip = False
                print(f"Auto-switch: using (cap_1, cap_2) - center valid {valid_a} vs {valid_b}")
        except Exception:
            pass

    # Calculate Depth
    depth_map = np.full_like(disparity, np.nan)
    depth_map[valid_mask] = (focal_length * baseline) / disparity[valid_mask]

    # Temporal smoothing
    if depth_history is None:
        depth_history = depth_map.copy()
    else:
        alpha = 0.5
        valid_cur = ~np.isnan(depth_map)
        valid_hist = ~np.isnan(depth_history)
        both_valid = valid_cur & valid_hist

        if np.any(both_valid):
            depth_map[both_valid] = (
                alpha * depth_map[both_valid] + (1 - alpha) * depth_history[both_valid]
            )

        depth_history[valid_cur] = depth_map[valid_cur]

    depth_colormap = None
    if disparity_enabled:
        valid_depth_mask = ~np.isnan(depth_map)
        depth_clipped = np.clip(depth_map, DEPTH_MIN_CM, DEPTH_MAX_CM)

        depth_norm = np.zeros(depth_clipped.shape, dtype=np.uint8)
        if np.any(valid_depth_mask):
            scaled = (depth_clipped[valid_depth_mask] - DEPTH_MIN_CM) / (DEPTH_MAX_CM - DEPTH_MIN_CM)
            scaled = np.clip(scaled, 0.0, 1.0)
            depth_vals = (scaled * 255.0).astype(np.uint8)
            depth_norm[valid_depth_mask] = 255 - depth_vals

        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

        center_y = depth_map.shape[0] // 2
        center_x = depth_map.shape[1] // 2

        win = 11
        hy = win // 2
        h, w = disparity.shape
        y0 = max(0, center_y - hy)
        y1 = min(h, center_y + hy + 1)
        x0 = max(0, center_x - hy)
        x1 = min(w, center_x + hy + 1)

        disp_win = disparity[y0:y1, x0:x1]
        valid_win = disp_win > 0.0
        valid_count_win = int(np.sum(valid_win))

        center_disp = None
        if valid_count_win > 0:
            center_disp = float(np.median(disp_win[valid_win]))
            depth_vals_win = None
            if h > 0:
                dm_win = depth_map[y0:y1, x0:x1]
                valid_dm = ~np.isnan(dm_win)
                if np.any(valid_dm):
                    depth_vals_win = float(np.median(dm_win[valid_dm]))

        if center_disp is not None:
            depth_from_disp = (focal_length * baseline) / center_disp if center_disp != 0 else float('nan')
            last_center_depth = depth_vals_win if depth_vals_win is not None else depth_from_disp
            last_center_disparity = center_disp

        center_band = "N/A"
        if last_center_depth is not None:
            if last_center_depth <= DEPTH_NEAR_CM:
                center_band = "Near"
            elif last_center_depth <= DEPTH_MID_CM:
                center_band = "Mid"
            else:
                center_band = "Far"

        cv2.putText(
            depth_colormap,
            f"Center: {last_center_depth:.1f} cm ({center_band})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return depth_map, depth_colormap, depth_history, last_center_depth, last_center_disparity

def run_yolo_inference(model, frame_1, depth_map, yolo_enabled, do_infer,
                       last_annotated, confidence_threshold, yolo_imgsz):
    """
    Run YOLO inference and annotate frame with detections and depth.
    Returns: (annotated_frame, last_annotated)
    """
    annotated_frame = frame_1
    
    if yolo_enabled:
        if do_infer:
            try:
                results = model.predict(
                    source=frame_1,
                    conf=confidence_threshold,
                    imgsz=yolo_imgsz,
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
                        cx = max(0, min(depth_map.shape[1] - 1, cx))
                        cy = max(0, min(depth_map.shape[0] - 1, cy))
                        depth_cm = float(depth_map[cy, cx])
                        
                        if not np.isnan(depth_cm):
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
                        else:
                            label = f"{names[cls_id]}"
                            cv2.putText(
                                annotated_frame,
                                label,
                                (int(x1), max(int(y1) - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (100, 100, 255),
                                2,
                            )

                last_annotated = annotated_frame
            except Exception as exc:
                print(f"Warning: YOLO inference failed: {exc}")
                last_annotated = None
                annotated_frame = frame_1
        elif last_annotated is not None:
            annotated_frame = last_annotated

    return annotated_frame, last_annotated


def main():
    global BASELINE, sound_enabled, sound_thread, sound_stop_event, latest_center_depth
    left_window = "Left Camera"
    depth_window = "Real-Time Depth (Heatmap)"
    yolo_window = "YOLO Predictions"

    if not MODEL_PATH.exists():
        print(f"Error: model not found at '{MODEL_PATH}'.")
        sys.exit(1)
    else:
        model = YOLO(str(MODEL_PATH))

    # Load stereo calibration data
    try:
        data = np.load("camera_params.npz", allow_pickle=True)
    except FileNotFoundError:
        print("Error: 'camera_params.npz' not found. Run stereo_calibrate.py first!")
        sys.exit(1)

    # Fallbacks
    focal_length = None
    rect_maps = None
    calibrated_size = None
    use_rectification = False
    stereo_rms = None
    try:
        if 'focal_length' in data:
            focal_length = float(np.asarray(data['focal_length']).reshape(-1)[0])
            print(f"Loaded Focal Length: {focal_length:.2f}")
        baseline_cm_loaded = BASELINE
        if 'baseline_cm' in data:
            baseline_cm_loaded = float(np.asarray(data['baseline_cm']).reshape(-1)[0])
            print(f"Loaded Baseline (cm): {baseline_cm_loaded:.2f}")

        if 'imageSize' in data:
            calibrated_size = tuple(int(v) for v in np.asarray(data['imageSize']).reshape(-1)[:2])
            print(f"Loaded calibration image size: {calibrated_size[0]}x{calibrated_size[1]}")

        # Load rectification maps if present
        if 'map1x' in data and 'map1y' in data and 'map2x' in data and 'map2y' in data:
            map1x = data['map1x']
            map1y = data['map1y']
            map2x = data['map2x']
            map2y = data['map2y']
            rect_maps = (map1x, map1y, map2x, map2y)
            print("Loaded rectification maps from camera_params.npz")

        if 'stereo_rms' in data:
            stereo_rms = float(np.asarray(data['stereo_rms']).reshape(-1)[0])
            print(f"Loaded stereo RMS: {stereo_rms:.4f}")

        if focal_length is None:
            raise KeyError('focal_length')
    except Exception as exc:
        print(f"Error: invalid or missing calibration data in camera_params.npz: {exc}")
        sys.exit(1)

    BASELINE = baseline_cm_loaded

    # Sanitize calibration values before using them in depth or rectification.
    if focal_length is None or not np.isfinite(focal_length) or focal_length <= 0:
        if 'K1' in data:
            K1 = np.asarray(data['K1'])
            if K1.shape == (3, 3) and np.isfinite(K1[0, 0]):
                focal_length = float(abs(K1[0, 0]))
                print(f"Warning: replacing invalid focal length with |K1[0,0]| = {focal_length:.2f}")

    if stereo_rms is not None and np.isfinite(stereo_rms) and stereo_rms > 1.5:
        print("Warning: stereo RMS is high; disabling rectification maps for now.")
        rect_maps = None
    elif focal_length is None or not np.isfinite(focal_length) or focal_length <= 0:
        print("Warning: focal length is invalid; disabling rectification maps and depth computation.")
        rect_maps = None

    use_rectification = rect_maps is not None

    cap_1 = cv2.VideoCapture(LEFT_CAMERA_INDEX)
    cap_2 = cv2.VideoCapture(RIGHT_CAMERA_INDEX)

    if calibrated_size is not None:
        cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, calibrated_size[0])
        cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, calibrated_size[1])
        cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, calibrated_size[0])
        cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, calibrated_size[1])

    if not cap_1.isOpened() or not cap_2.isOpened():
        print(f"Error: failed to open cameras {LEFT_CAMERA_INDEX} and/or {RIGHT_CAMERA_INDEX}")
        cap_1.release()
        cap_2.release()
        sys.exit(1)

    # Initialize Stereo Matcher (tuned for more reliable matches)
    num_disp = 16 * 6  # must be divisible by 16; increase if you need larger depth range
    block = 7  # odd, between 3..11 - balances detail vs stability
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # Default disparity order: use (cap_2, cap_1) - same as pressing 'r' once.
    # This matches the user's observed better behavior when moving the chessboard.
    stereo_flip = True
    # How often (frames) to re-check ordering at runtime and thresholds
    ORDER_CHECK_INTERVAL = 120
    MIN_VALID_PIXELS_FOR_OK = 2000
    
    # For temporal smoothing of depth
    depth_history = None

    print("Running Depth + YOLO stream.")
    print("Press 'y' to toggle YOLO inference on/off.")
    print("Press 'd' to toggle disparity/depth heatmap on/off.")
    print("Press 'q' to quit.")
    print(f"Baseline (cm): {BASELINE}")
    print(f"Focal length (px): {focal_length:.2f}")

    frame_count = 0
    last_annotated = None
    yolo_enabled = False
    disparity_enabled = False
    yolo_window_visible = False
    frame_size_warning_shown = False
    last_center_depth = None
    last_center_disparity = None

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

        if calibrated_size is not None and (frame_1.shape[1], frame_1.shape[0]) != calibrated_size:
            if not frame_size_warning_shown:
                print(
                    f"Warning: live camera size {(frame_1.shape[1], frame_1.shape[0])} does not match calibration size {calibrated_size}. "
                    "Resizing both frames to calibration size before rectification."
                )
                frame_size_warning_shown = True
            frame_1 = cv2.resize(frame_1, calibrated_size, interpolation=cv2.INTER_LINEAR)
            frame_2 = cv2.resize(frame_2, calibrated_size, interpolation=cv2.INTER_LINEAR)

        need_depth = disparity_enabled or yolo_enabled
        depth_map = None
        depth_colormap = None

        if need_depth:
            depth_map, depth_colormap, depth_history, last_center_depth, last_center_disparity = compute_depth_map(
                frame_1, frame_2, stereo, focal_length, BASELINE,
                use_rectification, rect_maps, depth_history,
                disparity_enabled, stereo_flip, last_center_depth,
                last_center_disparity, ORDER_CHECK_INTERVAL,
                MIN_VALID_PIXELS_FOR_OK, frame_count
            )

                # publish latest center depth for audio thread
            latest_center_depth = last_center_depth

        frame_count += 1
        do_infer = yolo_enabled and (frame_count % infer_stride) == 0
        annotated_frame, last_annotated = run_yolo_inference(
            model, frame_1, depth_map, yolo_enabled, do_infer,
            last_annotated, CONFIDENCE_THRESHOLD, YOLO_IMGSZ
        )

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
        if key in (ord('r'), ord('R')):
            stereo_flip = not stereo_flip
            state = "(cap_2, cap_1)" if stereo_flip else "(cap_1, cap_2)"
            print(f"Toggled disparity order, now using {state}")
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
        if key in (ord('s'), ord('S')):
            sound_enabled = not sound_enabled
            if sound_enabled:
                sound_stop_event = threading.Event()
                sound_thread = threading.Thread(target=sound_thread_func, args=(sound_stop_event,), daemon=True)
                sound_thread.start()
                print("Sound enabled")
            else:
                if sound_stop_event is not None:
                    sound_stop_event.set()
                if sound_thread is not None:
                    sound_thread.join(timeout=1.0)
                sound_thread = None
                sound_stop_event = None
                print("Sound disabled")

    # Ensure sound thread is stopped on exit
    if sound_stop_event is not None:
        try:
            sound_stop_event.set()
        except Exception:
            pass
    if sound_thread is not None:
        try:
            sound_thread.join(timeout=1.0)
        except Exception:
            pass

    cap_1.release()
    cap_2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()