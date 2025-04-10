import cv2
import libcamera
from picamera2 import Picamera2

# Load the pre-trained face detection cascade classifier
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

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
    # Main loop to capture frames and detect faces
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

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame with detected faces
        cv2.imshow('video', img)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up resources
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()