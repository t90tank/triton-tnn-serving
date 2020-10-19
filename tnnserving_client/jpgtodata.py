# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# from torchvision import transforms

# # loader使用torchvision中自带的transforms函数
# loader = transforms.Compose([
#     transforms.ToTensor()])  

# unloader = transforms.ToPILImage()

# def PIL_to_tensor(image):
#     image = loader(image).unsqueeze(0)
#     return image.to(torch.float16)

# t = PIL_to_tensor(Image.open('pics/tiger_cat.jpg'))
# print(t)
# print(t.size())

import cv2
import numpy as np
color_img=cv2.imread('pics/tiger_cat_2.jpg')
print(color_img.shape)
ss = '['
for i in range(color_img.shape[0]):
  if i != 0: ss += ','
  ss += '['
  for j in range(color_img.shape[1]):
    if j != 0: ss += ','
    ss += '['
    for k in range(color_img.shape[2]):
      if k != 0: ss += ','
      ss += str(color_img[i][j][k])
      # ss += "0"
    ss += ']'
  ss += ']'
ss += ']'
# ss = '['
# for k in range(color_img.shape[2]):
#   if k != 0: ss += ','
#   ss += '['
#   for j in range(color_img.shape[1]):
#     if j != 0: ss += ','
#     ss += '['
#     for i in range(color_img.shape[0]):
#       if i != 0: ss += ','
#       ss += str(color_img[i][j][k])
#       # ss += "0"
#     ss += ']'
#   ss += ']'
# ss += ']'

f = open("data.txt", 'w')
print(ss, file = f)
# print(color_img.shape)
# ss = str(color_img)
# ss = ss.replace(']\n', '],')
# print(ss)