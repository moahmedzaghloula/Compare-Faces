#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
import numpy as np
import os
from io import BytesIO
from PIL import Image

app = FastAPI()
UPLOAD_FOLDER = 'uploads'

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def preprocess_image_from_bytes(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image_rgb = np.array(image)
    return image_rgb

def detect_and_align_face(image_rgb):
    detector = MTCNN()
    results = detector.detect_faces(image_rgb)
    if results:
        x, y, width, height = results[0]['box']
        margin = 0.2
        x_min, y_min = max(x - int(margin * width), 0), max(y - int(margin * height), 0)
        x_max, y_max = min(x + width + int(margin * width), image_rgb.shape[1]), min(y + height + int(margin * height), image_rgb.shape[0])
        face = image_rgb[y_min:y_max, x_min:x_max]
        aligned_face = cv2.resize(face, (160, 160))
        return aligned_face
    else:
        return None

def get_face_embedding(face):
    embedder = FaceNet()
    embedding = embedder.embeddings([face])[0]
    return embedding

def is_same_person(embedding1, embedding2, threshold=0.5):
    distance = cosine(embedding1, embedding2)
    return distance < threshold

@app.post("/compare_faces")
async def compare_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    image1_bytes = await image1.read()
    image2_bytes = await image2.read()

    image1_rgb = preprocess_image_from_bytes(image1_bytes)
    image2_rgb = preprocess_image_from_bytes(image2_bytes)

    face1 = detect_and_align_face(image1_rgb)
    face2 = detect_and_align_face(image2_rgb)

    if face1 is None or face2 is None:
        raise HTTPException(status_code=400, detail="Face not detected in one or both images")

    embedding1 = get_face_embedding(face1)
    embedding2 = get_face_embedding(face2)

    result = is_same_person(embedding1, embedding2)

    if result:
        return JSONResponse(content={'result': 'They are the same person.'})
    else:
        return JSONResponse(content={'result': 'They are not the same person.'})

if __name__ == '__main__':
    import uvicorn
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    uvicorn.run(app, host='0.0.0.0', port=5000)
