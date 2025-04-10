import numpy as np
import cv2
import libcamera
from picamera2 import Picamera2

# Load pre-trained face and smile detection cascade classifiers
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')

# Initialize the Picamera2 object
picam2 = Picamera2()

# Configure the camera for preview with RGB888 format and 640x480 resolution
config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)},
                                            # Optional raw configuration
                                            # raw={"format": 'SRGGB12', "size": (1920, 1080)}
                                            )
config["transform"] = libcamera.Transform(hflip=0, vflip=1)  # Apply vertical flip
picam2.configure(config)
picam2.start()

try:
    # Main loop to capture frames and detect faces and smiles
    while True:
        img = picam2.capture_array()  # Capture a frame from the camera
        img = cv2.flip(img, 1)  # Flip the frame horizontally
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Detect faces in the grayscale image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30)
        )

        # Draw rectangles around detected faces and detect smiles within each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw face rectangle
            roi_gray = gray[y:y+h, x:x+w]  # Region of interest for face in grayscale
            roi_color = img[y:y+h, x:x+w]  # Region of interest for face in color

            # Detect smiles in the face region
            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.3,
                minNeighbors=15,
                minSize=(25, 25)
            )

            # Draw rectangles around detected smiles
            for (xx, yy, ww, hh) in smile:
                cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

        # Display the frame with detected faces and smiles
        cv2.imshow('video', img)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up resources
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()