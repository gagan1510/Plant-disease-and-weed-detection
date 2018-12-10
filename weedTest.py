import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img = 'imageTest123.jpg'

mainImage = cv2.imread(img)
image = cv2.cvtColor(mainImage, cv2.COLOR_BGR2RGB)
xInit, yInit, _ = mainImage.shape

cv2.rectangle(image, (int(xInit/20),int(yInit/20)), 
              (int(xInit/3.1),int(yInit/3)), (0,0,0), cv2.FILLED)

cv2.rectangle(image, (int(xInit/2.8),int(yInit- yInit/3.5)), 
              (int(xInit/1.45),int(yInit/2.7)), (0,0,0), cv2.FILLED)

cv2.rectangle(image, (int(xInit/1.36),int(yInit/20)), 
              (int(xInit/1),int(yInit/3)), (0,0,0), cv2.FILLED)

cv2.rectangle(image, (int(xInit/0.99999),int(yInit- yInit/3.5)), 
              (int(xInit/0.78),int(yInit/2.7)), (0,0,0), cv2.FILLED)

mask = cv2.inRange(image, (36,0,0) , (86,255,255))

testImage = cv2.imread(img)
imask = mask>0
green = np.zeros_like(testImage, np.uint8)
testImage[imask] = image[imask]+120

cv2.imshow(testImage, cmap='Greys_r')