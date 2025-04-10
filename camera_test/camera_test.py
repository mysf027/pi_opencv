import cv2
import numpy as np
import libcamera
from time import sleep
from picamera2 import Picamera2

# Initialize the Picamera2 object
picam2 = Picamera2()

# Create a preview configuration for the camera
# The main stream is set to RGB888 format with a resolution of 640x480
config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)},
                                            # Uncomment the raw configuration if needed
                                            # raw={"format": 'SRGGB12', "size": (1920, 1080)}
                                            )

# Apply a transformation to the camera output: flip vertically (vflip=1)
config["transform"] = libcamera.Transform(hflip=0, vflip=1)

# Configure the camera with the specified settings
picam2.configure(config)

# Start the camera
picam2.start()

try:
    # Main loop to continuously capture and process frames
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()
        
        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Display the original frame in a window titled "frame"
        cv2.imshow("frame", frame)
        
        # Display the grayscale frame in a window titled "gray"
        cv2.imshow('gray', gray)
        
        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the camera
    picam2.stop()
    
    # Close the camera connection
    picam2.close()
    
    # Destroy all OpenCV windows to clean up resources
    cv2.destroyAllWindows()