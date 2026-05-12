import cv2
import numpy as np
import sys

# --- SETTINGS ---
CHESSBOARD_SIZE = (9, 6)  # internal corners
SQUARE_SIZE = 2.5  # cm
MIN_FRAMES = 12
LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = 2
AUTO_SAVE_INTERVAL_SEC = 0.8
AUTO_SAVE_MIN_CENTER_SHIFT_PX = 25.0
AUTO_SAVE_MIN_SCALE_CHANGE_PX = 15.0

CHESSBOARD_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)
CLASSIC_CHESSBOARD_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)
SB_CHESSBOARD_FLAGS = 0


def detect_chessboard(gray):
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE, flags=SB_CHESSBOARD_FLAGS)
        if found:
            return found, corners.astype(np.float32)

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None, CLASSIC_CHESSBOARD_FLAGS)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CHESSBOARD_CRITERIA)
    return found, corners

def capture_pairs(left_idx=None, right_idx=None):
    if left_idx is None or right_idx is None:
        left_idx = LEFT_CAMERA_INDEX if left_idx is None else left_idx
        right_idx = RIGHT_CAMERA_INDEX if right_idx is None else right_idx

    print(f"Using cameras {left_idx} and {right_idx}")

    capL = cv2.VideoCapture(left_idx)
    capR = cv2.VideoCapture(right_idx)
    if not capL.isOpened() or not capR.isOpened():
        print("Error: failed to open cameras")
        sys.exit(1)

    objpoints = []
    imgpointsL = []
    imgpointsR = []

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    print("Auto-save is enabled: valid chessboard pairs will be saved automatically with a short cooldown.")
    print("Press 'q' to finish and calibrate (requires at least {} pairs).".format(MIN_FRAMES))

    count = 0
    last_auto_save_ts = 0.0
    last_saved_center = None
    last_saved_scale = None
    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            print("Failed to read from cameras")
            break

        disp = np.hstack((frameL, frameR))
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = detect_chessboard(grayL)
        foundR, cornersR = detect_chessboard(grayR)

        vis = disp.copy()
        if foundL:
            cv2.drawChessboardCorners(vis[:, :frameL.shape[1]], CHESSBOARD_SIZE, cornersL, foundL)
        if foundR:
            cv2.drawChessboardCorners(vis[:, frameL.shape[1]:], CHESSBOARD_SIZE, cornersR, foundR)

        cv2.imshow('Stereo Capture (L | R)', vis)
        k = cv2.waitKey(1) & 0xFF
        now_ts = cv2.getTickCount() / cv2.getTickFrequency()
        should_auto_save = False
        if foundL and foundR and (now_ts - last_auto_save_ts) >= AUTO_SAVE_INTERVAL_SEC:
            cornersL_xy = cornersL.reshape(-1, 2)
            cornersR_xy = cornersR.reshape(-1, 2)
            centerL = cornersL_xy.mean(axis=0)
            centerR = cornersR_xy.mean(axis=0)
            center = (centerL + centerR) / 2.0

            scaleL = float(np.mean(np.linalg.norm(cornersL_xy - centerL, axis=1)))
            scaleR = float(np.mean(np.linalg.norm(cornersR_xy - centerR, axis=1)))
            scale = (scaleL + scaleR) / 2.0

            if last_saved_center is None or last_saved_scale is None:
                should_auto_save = True
            else:
                center_shift = float(np.linalg.norm(center - last_saved_center))
                scale_change = abs(scale - last_saved_scale)
                should_auto_save = (
                    center_shift >= AUTO_SAVE_MIN_CENTER_SHIFT_PX
                    or scale_change >= AUTO_SAVE_MIN_SCALE_CHANGE_PX
                )

        if should_auto_save:
            imgpointsL.append(cornersL.copy())
            imgpointsR.append(cornersR.copy())
            objpoints.append(objp.copy())
            count += 1
            print(f"Auto-saved pair #{count}")
            last_auto_save_ts = now_ts
            cornersL_xy = cornersL.reshape(-1, 2)
            cornersR_xy = cornersR.reshape(-1, 2)
            centerL = cornersL_xy.mean(axis=0)
            centerR = cornersR_xy.mean(axis=0)
            last_saved_center = (centerL + centerR) / 2.0
            scaleL = float(np.mean(np.linalg.norm(cornersL_xy - centerL, axis=1)))
            scaleR = float(np.mean(np.linalg.norm(cornersR_xy - centerR, axis=1)))
            last_saved_scale = (scaleL + scaleR) / 2.0

        if k == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    return objpoints, imgpointsL, imgpointsR, frameL.shape[1::-1]


def calibrate_stereo():
    objpoints, imgpointsL, imgpointsR, imageSize = capture_pairs()
    if len(objpoints) < MIN_FRAMES:
        print(f"Not enough pairs collected ({len(objpoints)}). Need at least {MIN_FRAMES}.")
        return

    print("Calibrating individual cameras...")
    retL, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpointsL, imageSize, None, None)
    retR, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpointsR, imageSize, None, None)
    print(f"Left camera RMS reprojection error: {retL:.4f}")
    print(f"Right camera RMS reprojection error: {retR:.4f}")

    print("Running stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        K1,
        D1,
        K2,
        D2,
        imageSize,
        criteria=criteria,
        flags=flags,
    )
    print(f"Stereo RMS reprojection error: {ret:.4f}")

    if not np.isfinite(ret) or ret > 1.5:
        print("Warning: stereo calibration error is high. The saved rectification may be poor.")

    if np.isfinite(ret):
        if ret <= 1.0:
            quality = "usable"
        elif ret <= 1.5:
            quality = "marginal"
        else:
            quality = "poor"
        print(f"Calibration quality: {quality}")

    print("Computing rectification maps...")
    # Keep the full field of view instead of aggressively cropping it away.
    # alpha=1 preserves more of the image and is safer for debugging calibration quality.
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, imageSize, R, T, alpha=1.0)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, imageSize, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, imageSize, cv2.CV_32FC1)

    focal_length = float(abs(P1[0, 0]))
    if not np.isfinite(focal_length) or focal_length <= 0:
        focal_length = float(abs(K1[0, 0]))

    baseline_cm = float(abs(np.asarray(T).reshape(-1)[0]))  # T is in same units as SQUARE_SIZE (cm)

    if not np.isfinite(focal_length) or focal_length <= 0:
        print("Error: invalid focal length estimated from calibration. Not saving parameters.")
        return

    np.savez("camera_params.npz",
             K1=K1, D1=D1, K2=K2, D2=D2,
             R=R, T=T,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
             focal_length=focal_length, baseline_cm=baseline_cm,
             left_rms=retL, right_rms=retR, stereo_rms=ret,
             imageSize=imageSize)

    print(f"Saved stereo calibration to camera_params.npz (focal={focal_length:.2f}px, baseline={baseline_cm:.2f}cm)")
    print("Use this calibration only if the stereo quality is usable or marginal; poor results usually need better chessboard coverage.")


if __name__ == '__main__':
    calibrate_stereo()
