import os
import cv2
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt

load_dotenv()

IBM_WATSON_API_KEY = os.getenv("IBM_WATSON_API_KEY")
IBM_WATSON_URL = os.getenv("IBM_WATSON_URL")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_URL = os.getenv("AZURE_URL")

def take_photo(filename='photo.jpg'):
    cam_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = cam_capture.read()
        key = cv2.waitKey(100)
        if ret:
            cv2.imshow("Camera", frame)
        if key == 13 or key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(filename, frame)
            break
    cam_capture.release()
    cv2.destroyAllWindows()

def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    return ((left, top), (right, bottom))

def drawCrossesOnFace(draw, faceLandmarks, size=2, color='white'):
    fl = faceLandmarks.as_dict()
    for l in fl.keys():
        x = fl[l]['x']
        y = fl[l]['y']
        draw.line(((x - size, y - size), (x + size, y + size)), fill=color)
        draw.line(((x - size, y + size), (x + size, y - size)), fill=color)
    pass

def print_fiducial_points(image, face):
    img = Image.open(image)
    draw = ImageDraw.Draw(img)
    draw.rectangle(getRectangle(face), outline='red')
    drawCrossesOnFace(draw, face.face_landmarks, size=2, color='white')
    # cv2.imshow("Pontos Fiduciais", np.array(img))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.figure(figsize=(20,10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Pontos Fiduciais')

def facial_recognition(face_client, image):
    recognition = face_client.face.detect_with_stream(image=image, return_face_landmarks=True, return_face_attributes=["age", "gender"])
    face = recognition[0]
    print_fiducial_points(image, face)
    face_attributes = face.face_attributes
    return face_attributes.age, face_attributes.gender