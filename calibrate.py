import cv2
import numpy as np
import glob

# --- SETTINGS ---
CHESSBOARD_SIZE = (9, 6) # Internal corners
SQUARE_SIZE = 2.5        # cm, mm, or inches (whichever you prefer)

def calibrate():
    # Prepare coordinates (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints = [] # 3d points in real world
    imgpoints = [] # 2d points in image plane

    cap = cv2.VideoCapture(0) # Use the index of your primary camera
    print("Point the camera at the chessboard. Press 's' to save a frame, 'q' to finish.")

    count = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find corners for visual feedback
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        display_frame = frame.copy()
        if found:
            cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, found)
        
        cv2.imshow("Calibration", display_frame)
        key = cv2.waitKey(1)

        if key == ord('s') and found:
            objpoints.append(objp)
            imgpoints.append(corners)
            count += 1
            print(f"Frame {count} saved!")
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 10:
        print("Calculating...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        # Extract focal length (average of fx and fy)
        focal_length = (mtx[0, 0] + mtx[1, 1]) / 2
        
        # Save to file
        np.savez("camera_params.npz", focal_length=focal_length)
        print(f"\nSUCCESS!")
        print(f"Focal Length: {focal_length:.2f} pixels")
        print("Parameters saved to 'camera_params.npz'")
    else:
        print("Not enough frames captured. Try to get at least 10-15.")

if __name__ == "__main__":
    calibrate()