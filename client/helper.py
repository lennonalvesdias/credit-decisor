import os
import cv2
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

IBM_WATSON_API_KEY = os.getenv("IBM_WATSON_API_KEY")
IBM_WATSON_URL = os.getenv("IBM_WATSON_URL")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_URL = os.getenv("AZURE_URL")

def take_photo():
    cam_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = cam_capture.read()
        if cv2.waitKey(1) == 13:
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
  cv2.imgshow(img)

def facial_recognition(face_client, image):
  recognition = face_client.face.detect_with_stream(image=image, return_face_landmarks=True, return_face_attributes=["age", "gender"])
  face = recognition[0]
  print_fiducial_points(image, face)
  face_attributes = face.face_attributes
  return face_attributes.age, face_attributes.gender