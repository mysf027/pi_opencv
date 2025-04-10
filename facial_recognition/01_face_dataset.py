import cv2
import numpy as np
import libcamera
from time import sleep
from picamera2 import Picamera2

# Load face detection classifier
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')


# Initialize camera
picam2 = Picamera2()
config=picam2.create_preview_configuration(main={"format":'RGB888',"size":(640,480)},
                                        #raw={"format":'SRGGB12',"size":(1920,1080)}
                                           )
config["transform"] = libcamera.Transform(hflip = 0, vflip = 1)
picam2.configure(config)
picam2.start()

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get user input for dataset labeling
face_name = input('\n enter user name end press <return> ==> ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize face count
count = 0

while(True):
    # Capture frame from camera
    img =  picam2.capture_array()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30)
        )
    
    # Process detected faces
    for (x,y,w,h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save face image to dataset
        cv2.imwrite(f"dataset/{face_name}_{count}.jpg", gray[y:y+h,x:x+w])
        # Display result
        cv2.imshow('image', img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
print("\n [INFO] Exiting Program and clean up stuff")
picam2.stop()
picam2.close()
cv2.destroyAllWindows()



