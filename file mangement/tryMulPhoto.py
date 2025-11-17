import os
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Initialize the InsightFace app
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Perform face analysis
    faces = app.get(img)
    
    # Draw bounding boxes on the image (optional)
    rimg = app.draw_on(img, faces)
    
    # Save the output image (optional)
    output_path = image_path.replace(".jpg", "_output.jpg")
    cv2.imwrite(output_path, rimg)

    # Process face data (optional): for example, extract embeddings
    face_data = []
    for face in faces:
        # You can extract face embeddings or other data here
        face_embedding = face.embedding
        face_data.append(face_embedding)
    
    return face_data

def process_multiple_images(image_folder):
    # Loop through the images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            print(f"Processing {image_path}...")
            face_data = process_image(image_path)
            print(f"Processed {filename}, found {len(face_data)} faces.")

# Folder containing the images
image_folder = r"D:\project\photo\tryFifty"

# Process all images in the folder
process_multiple_images(image_folder)
