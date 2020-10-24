import cv2
import numpy as np
import sys
import json
import requests

image = "pics/test_face.jpg"

if len(sys.argv) >= 2:
  image = sys.argv[1]

color_img=cv2.imread(image)
shape = '['+str(color_img.shape)[1:-1]+']'
print(shape)
imagedata = json.dumps(color_img.tolist())

data = '{"inputs":[{"name":"input","datatype":"UINT8","shape":'+shape+',"data":'+imagedata+'}]}'

r = requests.post('http://localhost:8000/v2/models/tnn_face_detection/infer', data=data)
print(r.text)
