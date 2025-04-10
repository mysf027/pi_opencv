from picamera2 import Picamera2, Preview
import time
import libcamera  

# Initialize the Picamera2 object
picam2 = Picamera2()

# Create a preview configuration for the camera
config = picam2.create_preview_configuration()

# Apply transformations to the camera output: flip horizontally (hflip=1) and vertically (vflip=1)
config["transform"] = libcamera.Transform(hflip=1, vflip=1)

# Configure the camera with the specified settings
picam2.configure(config)

# Start the camera preview using the QTGL backend (for graphical display)
picam2.start_preview(Preview.QTGL)

# Start the camera
picam2.start()

# Wait for 2 seconds to allow the camera to warm up and the preview to be displayed
time.sleep(2) 

# Capture an image and save it to a file named "test.jpg"
picam2.capture_file("test.jpg")

# Stop the camera
picam2.stop()

# Close the camera connection to free up resources
picam2.close()