# Import required libraries
import cv2
import numpy as np
import os
from PIL import Image

# Initialize paths and models
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # Get all JPG files in dataset
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    
    # Create name-ID mapping
    name_id_map = {} 
    current_id = 0
    face_samples = []
    ids = []

    # Process each image
    for imagePath in imagePaths:
        try:
            # Parse filename to get name and ID
            filename = os.path.split(imagePath)[-1]
            name_part = filename.rsplit('_', 1)[0]
            
            # Update name-ID mapping
            if name_part not in name_id_map:
                name_id_map[name_part] = current_id
                current_id += 1
            
            user_id = name_id_map[name_part]

            # Convert image to grayscale numpy array
            pil_image = Image.open(imagePath).convert('L')
            img_np = np.array(pil_image, 'uint8')

            # Detect faces in image
            faces = detector.detectMultiScale(img_np)

            # Add face samples and IDs
            for (x, y, w, h) in faces:
                face_samples.append(img_np[y:y+h, x:x+w])
                ids.append(user_id)

        except Exception as e:
            print(f"error: {str(e)}")
            continue

    return face_samples, ids, name_id_map

# Main execution flow
print("\n [INFO] waiting...")
faces, ids, name_map = getImagesAndLabels(path)

# Create output directory
os.makedirs('trainer', exist_ok=True)

# Train and save model
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')

# Save name-ID mapping
with open('trainer/name_mapping.txt', 'w') as f:
    for name, id in name_map.items():
        f.write(f"{id}:{name}\n")

print(f"\n [INFO]finish")
