import cv2
import numpy as np
import libcamera
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (640, 480)}
)
config["transform"] = libcamera.Transform(hflip=0, vflip=1)
picam2.configure(config)
picam2.start()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def load_name_mapping():
    name_map = {}
    try:
        with open('trainer/name_mapping.txt', 'r') as f:
            for line in f:
                id_, name = line.strip().split(':')
                name_map[int(id_)] = name
    except Exception as e:
        print(f"error:{str(e)}")
        name_map = {}
    return name_map

name_map = load_name_mapping()
font = cv2.FONT_HERSHEY_SIMPLEX

min_face_size = (100, 100)  
confidence_threshold = 50    

while True:
    img = picam2.capture_array()
    img = cv2.flip(img, 1) 
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=min_face_size
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
 
        if confidence < confidence_threshold:
            label = name_map.get(id_, f"֪ID: {id_}")
            confidence_text = f" {100 - confidence:.1f}%"
        else:
            label = "unknow"
            confidence_text = f" {100 - confidence:.1f}%"
        
        cv2.putText(img, label, (x+5, y-10), font, 0.8, (0, 255, 0), 2)
        cv2.putText(img, confidence_text, (x+5, y+h-10), font, 0.6, (0, 255, 0), 1)

    cv2.imshow('facialrecognition', img)

    if cv2.waitKey(1) == ord('q'):
        break

picam2.stop()
picam2.close()
cv2.destroyAllWindows()
print("\n[INFO] exit")
