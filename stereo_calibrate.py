import cv2
import numpy as np
import sys

# --- SETTINGS ---
CHESSBOARD_SIZE = (9, 6)  # internal corners
SQUARE_SIZE = 2.5  # cm
MIN_FRAMES = 30
LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = 2

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
    
    # Lock exposure here if possible
    capL.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    capR.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    if not capL.isOpened() or not capR.isOpened():
        print("Error: failed to open cameras")
        sys.exit(1)

    objpoints = []
    imgpointsL = []
    imgpointsR = []

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    print("\n--- CALIBRATION MODE ---")
    print("1. Hold the board completely still.")
    print("2. Press 'c' to capture a frame.")
    print("3. Move the board to a new angle/corner.")
    print(f"4. Press 'q' to finish and calibrate (requires at least {MIN_FRAMES} pairs).\n")

    count = 0
    while True:
        # Synchronized Grabbing
        capL.grab()
        capR.grab()
        
        okL, frameL = capL.retrieve()
        okR, frameR = capR.retrieve()
        
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

        # Manual Capture
        if k == ord('c') and foundL and foundR:
            imgpointsL.append(cornersL.copy())
            imgpointsR.append(cornersR.copy())
            objpoints.append(objp.copy())
            count += 1
            print(f"Captured perfectly still pair #{count}")
            
            # Flash the screen green to indicate successful capture
            flash = np.zeros_like(vis)
            flash[:,:] = (0, 255, 0)
            cv2.imshow('Stereo Capture (L | R)', cv2.addWeighted(vis, 0.5, flash, 0.5, 0))
            cv2.waitKey(100) # Pause briefly

        elif k == ord('q'):
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
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    
    ret, K1_opt, D1_opt, K2_opt, D2_opt, R, T, E, F = cv2.stereoCalibrate(
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
        if ret <= 0.5:
            quality = "excellent"
        elif ret <= 1.0:
            quality = "usable"
        elif ret <= 1.5:
            quality = "marginal"
        else:
            quality = "poor"
        print(f"Calibration quality: {quality}")

    print("Computing rectification maps...")
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1_opt, D1_opt, K2_opt, D2_opt, imageSize, R, T, alpha=1.0)
    map1x, map1y = cv2.initUndistortRectifyMap(K1_opt, D1_opt, R1, P1, imageSize, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2_opt, D2_opt, R2, P2, imageSize, cv2.CV_32FC1)

    focal_length = float(abs(P1[0, 0]))
    if not np.isfinite(focal_length) or focal_length <= 0:
        focal_length = float(abs(K1_opt[0, 0]))

    baseline_cm = float(abs(np.asarray(T).reshape(-1)[0])) 

    if not np.isfinite(focal_length) or focal_length <= 0:
        print("Error: invalid focal length estimated from calibration. Not saving parameters.")
        return

    np.savez("camera_params.npz",
             K1=K1_opt, D1=D1_opt, K2=K2_opt, D2=D2_opt,
             R=R, T=T,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
             focal_length=focal_length, baseline_cm=baseline_cm,
             left_rms=retL, right_rms=retR, stereo_rms=ret,
             imageSize=imageSize)

    print(f"Saved stereo calibration to camera_params.npz (focal={focal_length:.2f}px, baseline={baseline_cm:.2f}cm)")

if __name__ == '__main__':
    calibrate_stereo()