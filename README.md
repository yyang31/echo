# E.C.H.O
## Environment Camera & Hearing Object-detector

## Virtual Environment:

create
```sh
python3 -m venv /home/stanley/Repos/echo/.venv --system-site-packages
```

activate
```sh
source /home/stanley/Repos/echo/.venv/bin/activate
python /home/stanley/Repos/echo/echo.py
```

## Credit

Kaggle Dataset
https://www.kaggle.com/datasets/nderalparslan/dwsonder

StereoSGBM
https://www.youtube.com/watch?v=gffZ3S9pBUE

Camera Calibration
https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html

Doors, Windows and Stairs Dataset
https://www.kaggle.com/datasets/nderalparslan/dwsonder?resource=download-directory

Simpleaudio Python Library
https://simpleaudio.readthedocs.io/en/latest/

pyttsx3 Python Library - text to speech
https://pyttsx3.readthedocs.io/en/latest/

In `echo.py`, press `s` to toggle the beep + label speech loop while YOLO is enabled.