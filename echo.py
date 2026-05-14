import cv2
import sys
import time
import numpy as np
from pathlib import Path

from ultralytics import YOLO
import threading
import math

import simpleaudio as sa
import pyttsx3

# --- CONFIGURATION ---
BASELINE = 12  # distance between the cameras in cm
MODEL_PATH = Path("yolo/runs/door_window_stair_yolo/weights/best.pt")
CONFIDENCE_THRESHOLD = 0.35
YOLO_IMGSZ = 640
INFER_EVERY_N_FRAMES = 2
LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = 2
CENTER_SIZE = 50

# Fixed depth visualization range
DEPTH_MIN_CM = 30.0
DEPTH_MAX_CM = 400.0

# TTS label speaking timeout (seconds)
TTS_LABEL_TIMEOUT = 5.0

# Sound runtime state
sound_stop_event = None
latest_center_depth = None
latest_yolo_label = None


def depth_to_sound_params(depth):
    """Map a depth (cm) to (frequency Hz, amplitude 0..1, duration s, period s)."""
    if depth is None:
        return None
    try:
        d = float(depth)
    except Exception:
        return None

    if not np.isfinite(d):
        return None

    d = float(np.clip(d, DEPTH_MIN_CM, DEPTH_MAX_CM))

    # Frequency mapping: near -> high, far -> low
    F_NEAR_HZ = 900.0
    F_FAR_HZ = 300.0
    t = (d - DEPTH_MIN_CM) / (DEPTH_MAX_CM - DEPTH_MIN_CM)
    freq = F_NEAR_HZ + (F_FAR_HZ - F_NEAR_HZ) * t

    # Amplitude mapping
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


def beep_thread_func(stop_event):
    """Background thread that emits discrete beeps based on `latest_center_depth`."""
    sample_rate = 44100

    while not stop_event.is_set():
        depth = latest_center_depth
        params = depth_to_sound_params(depth)
        if params is None:
            stop_event.wait(0.2)
            continue

        freq, amp, duration, period = params

        # Generate sine wave
        samples_count = max(1, int(sample_rate * duration))
        t = np.linspace(0, duration, samples_count, False)
        waveform = np.sin(2 * math.pi * freq * t)

        # Apply envelope to reduce clicks
        env_len = int(0.01 * sample_rate)
        if env_len * 2 < samples_count:
            env = np.ones_like(waveform)
            env[:env_len] = np.linspace(0.0, 1.0, env_len)
            env[-env_len:] = np.linspace(1.0, 0.0, env_len)
            waveform *= env

        # Scale to 16-bit signed integers
        max_amp = 2 ** 15 - 1
        audio = (waveform * amp * max_amp).astype(np.int16)

        if latest_center_depth:
            try:
                play_obj = sa.play_buffer(audio.tobytes(), 1, 2, sample_rate)
                while play_obj.is_playing():
                    if stop_event.wait(0.05):
                        try:
                            play_obj.stop()
                        except Exception:
                            pass
                        return
            except Exception as exc:
                print(f"Warning: sound playback failed: {exc}")
                return

        remaining = max(0.0, period - duration)
        stop_event.wait(remaining)


def tts_thread_func(stop_event):
    """Background thread that handles TTS label speaking with timeout."""
    global latest_yolo_label
    last_spoken_label = None
    label_speak_start_time = None
    tts_engine = pyttsx3.init()

    while not stop_event.is_set():
        # Handle TTS label speaking with timeout
        if latest_yolo_label:
            if latest_yolo_label != last_spoken_label:
                last_spoken_label = latest_yolo_label
                label_speak_start_time = time.time()
                try:
                    tts_engine.say(latest_yolo_label)
                    tts_engine.runAndWait()
                except Exception as exc:
                    print(f"Warning: TTS playback failed: {exc}")
            else:
                if label_speak_start_time is not None:
                    elapsed_time = time.time() - label_speak_start_time
                    if elapsed_time >= TTS_LABEL_TIMEOUT:
                        last_spoken_label = None
                        label_speak_start_time = None
        else:
            last_spoken_label = None
            label_speak_start_time = None

        stop_event.wait(0.1)


def compute_depth_map(frame_1, frame_2, stereo, focal_length, baseline,
                      use_rectification, rect_maps, depth_history):
    """Compute disparity and depth map from stereo frame pair."""
    # Apply rectification maps if available
    if use_rectification:
        m1x, m1y, m2x, m2y = rect_maps
        frame_1 = cv2.remap(frame_1, m1x, m1y, interpolation=cv2.INTER_LINEAR)
        frame_2 = cv2.remap(frame_2, m2x, m2y, interpolation=cv2.INTER_LINEAR)

    gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    # Compute Disparity
    disparity = stereo.compute(gray_2, gray_1).astype(np.float32) / 16.0

    valid_mask = disparity > 0.0

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

    # Calculate center depth at geometric center
    center_y = depth_map.shape[0] // 2
    center_x = depth_map.shape[1] // 2

    hy = CENTER_SIZE // 2
    h, w = disparity.shape
    y0 = max(0, center_y - hy)
    y1 = min(h, center_y + hy + 1)
    x0 = max(0, center_x - hy)
    x1 = min(w, center_x + hy + 1)

    disp_win = disparity[y0:y1, x0:x1]
    valid_win = disp_win > 0.0
    last_center_depth = None

    if np.any(valid_win):
        dm_win = depth_map[y0:y1, x0:x1]
        valid_dm = ~np.isnan(dm_win)
        if np.any(valid_dm):
            last_center_depth = float(np.median(dm_win[valid_dm]))

    return depth_map, depth_history, last_center_depth, disparity


def disparity_to_display(disparity):
    """Convert raw disparity values into a colorized image for display."""
    display = np.zeros((*disparity.shape, 3), dtype=np.uint8)
    valid_mask = np.isfinite(disparity) & (disparity > 0.0)

    if not np.any(valid_mask):
        return display

    valid_values = disparity[valid_mask]
    disp_min = float(np.min(valid_values))
    disp_max = float(np.max(valid_values))

    if not np.isfinite(disp_min) or not np.isfinite(disp_max) or disp_max <= disp_min:
        normalized = np.zeros_like(disparity, dtype=np.uint8)
    else:
        normalized = np.zeros_like(disparity, dtype=np.uint8)
        scaled = (disparity[valid_mask] - disp_min) / (disp_max - disp_min)
        normalized[valid_mask] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    display = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    display[~valid_mask] = 0
    return display


def run_yolo_inference(model, frame_1, depth_map, confidence_threshold, yolo_imgsz):
    """Run YOLO inference and annotate frame with detections and depth."""
    try:
        results = model.predict(
            source=frame_1,
            conf=confidence_threshold,
            imgsz=yolo_imgsz,
            verbose=False,
        )
        result = results[0]
        annotated_frame = result.plot()

        labels = []
        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names
            seen_labels = set()
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                class_name = names[cls_id]
                if class_name not in seen_labels:
                    labels.append(class_name)
                    seen_labels.add(class_name)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cx = max(0, min(depth_map.shape[1] - 1, cx))
                cy = max(0, min(depth_map.shape[0] - 1, cy))
                depth_cm = float(depth_map[cy, cx])

                if not np.isnan(depth_cm):
                    label = f"{class_name}: {depth_cm:.1f} cm"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (int(x1), max(int(y1) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

        return annotated_frame, labels
    except Exception as exc:
        print(f"Warning: YOLO inference failed: {exc}")
        return frame_1, []


def main():
    global sound_stop_event, latest_center_depth, latest_yolo_label, BASELINE
    camera_window_name = "Environment Camera & Hearing Object-detector"
    disparity_window_name = "Disparity Map"

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

    # Load calibration parameters
    focal_length = None
    rect_maps = None
    calibrated_size = None
    use_rectification = False
    try:
        if 'focal_length' in data:
            focal_length = float(np.asarray(data['focal_length']).reshape(-1)[0])
            print(f"Loaded Focal Length: {focal_length:.2f}")

        baseline_cm_loaded = BASELINE
        if 'baseline_cm' in data:
            baseline_cm_loaded = float(np.asarray(data['baseline_cm']).reshape(-1)[0])

        if 'imageSize' in data:
            calibrated_size = tuple(int(v) for v in np.asarray(data['imageSize']).reshape(-1)[:2])

        # Load rectification maps if present
        if 'map1x' in data and 'map1y' in data and 'map2x' in data and 'map2y' in data:
            map1x = data['map1x']
            map1y = data['map1y']
            map2x = data['map2x']
            map2y = data['map2y']
            rect_maps = (map1x, map1y, map2x, map2y)

        if 'stereo_rms' in data:
            stereo_rms = float(np.asarray(data['stereo_rms']).reshape(-1)[0])
            if stereo_rms is not None and np.isfinite(stereo_rms) and stereo_rms > 1.5:
                rect_maps = None

        if focal_length is None:
            raise KeyError('focal_length')
    except Exception as exc:
        print(f"Error: invalid or missing calibration data in camera_params.npz: {exc}")
        sys.exit(1)

    BASELINE = baseline_cm_loaded
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

    # Initialize Stereo Matcher
    num_disp = 16 * 6
    block = 7
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

    # For temporal smoothing of depth
    depth_history = None

    print("Running YOLO inference with sound.")
    print("Press 'd' to toggle display windows.")
    print("Press 'q' to quit.")

    frame_count = 0
    display_enabled = True
    infer_stride = max(1, int(INFER_EVERY_N_FRAMES))

    # Start beep and TTS threads
    sound_stop_event = threading.Event()
    beep_thread = threading.Thread(target=beep_thread_func, args=(sound_stop_event,), daemon=True)
    tts_thread = threading.Thread(target=tts_thread_func, args=(sound_stop_event,), daemon=True)
    beep_thread.start()
    tts_thread.start()

    while True:
        success_1, frame_1 = cap_1.read()
        success_2, frame_2 = cap_2.read()

        if not success_1 or not success_2:
            print("Error: failed to read frame(s) from camera stream.")
            break

        if frame_1.shape[:2] != frame_2.shape[:2]:
            frame_2 = cv2.resize(frame_2, (frame_1.shape[1], frame_1.shape[0]), interpolation=cv2.INTER_LINEAR)

        if calibrated_size is not None and (frame_1.shape[1], frame_1.shape[0]) != calibrated_size:
            frame_1 = cv2.resize(frame_1, calibrated_size, interpolation=cv2.INTER_LINEAR)
            frame_2 = cv2.resize(frame_2, calibrated_size, interpolation=cv2.INTER_LINEAR)

        depth_map, depth_history, last_center_depth, disparity = compute_depth_map(
            frame_1, frame_2, stereo, focal_length, BASELINE,
            use_rectification, rect_maps, depth_history
        )
        latest_center_depth = last_center_depth

        frame_count += 1
        do_infer = (frame_count % infer_stride) == 0
        if do_infer:
            annotated_frame, detected_labels = run_yolo_inference(
                model, frame_1, depth_map, CONFIDENCE_THRESHOLD, YOLO_IMGSZ
            )
            latest_yolo_label = detected_labels[0] if detected_labels else None
        else:
            annotated_frame = frame_1

        # Draw center depth region box
        h, w = depth_map.shape
        # Shift measurement point right by baseline/2 (converted to pixels)
        baseline_offset_px = int((BASELINE * focal_length) / (CENTER_SIZE * 2))
        center_y = h // 2
        center_x = w // 2 + baseline_offset_px
        center_x = max(0, min(center_x, w - 1))  # Clamp to valid range
        hy = CENTER_SIZE // 2
        y0 = max(0, center_y - hy)
        y1 = min(h, center_y + hy + 1)
        x0 = max(0, center_x - hy)
        x1 = min(w, center_x + hy + 1)
        cv2.rectangle(annotated_frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
        
        # Display center depth value above the box
        if last_center_depth is not None and not np.isnan(last_center_depth):
            depth_text = f"Center: {last_center_depth:.1f} cm"
            cv2.putText(
                annotated_frame,
                depth_text,
                (x0, max(y0 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        if display_enabled:
            cv2.imshow(camera_window_name, annotated_frame)

            disparity_display = disparity_to_display(disparity)

            # Reuse the same center-depth annotation position on the disparity view.
            cv2.rectangle(disparity_display, (x0, y0), (x1, y1), (0, 0, 255), 2)

            if last_center_depth is not None and not np.isnan(last_center_depth):
                depth_text = f"Center: {last_center_depth:.1f} cm"
                cv2.putText(
                    disparity_display,
                    depth_text,
                    (x0, max(y0 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(disparity_window_name, disparity_display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key in (ord('d'), ord('D')):
            display_enabled = not display_enabled
            if not display_enabled:
                cv2.destroyAllWindows()
                print("Display disabled")
            else:
                print("Display enabled")

    # Stop threads
    if sound_stop_event is not None:
        try:
            sound_stop_event.set()
        except Exception:
            pass
    if beep_thread is not None:
        try:
            beep_thread.join(timeout=1.0)
        except Exception:
            pass
    if tts_thread is not None:
        try:
            tts_thread.join(timeout=1.0)
        except Exception:
            pass

    cap_1.release()
    cap_2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
