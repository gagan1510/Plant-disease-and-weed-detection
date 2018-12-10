import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img = 'imageTest.jpeg'

mainImage = cv2.imread(img)
image = cv2.cvtColor(mainImage, cv2.COLOR_BGR2RGB)
xInit, yInit, _ = mainImage.shape

resized_image = cv2.resize(image, (yInit, xInit))

y, X, _ = resized_image.shape

gsImg = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

blur = cv2.GaussianBlur(gsImg, (55,55), 0)

ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((50,50),np.uint8)
closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

image, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

def find_contour(cnts):
    contains = []
    y_ri,x_ri = im_bw_otsu.shape
    for cc in cnts:
        yn = cv2.pointPolygonTest(cc,(x_ri//2,y_ri//2),False)
        contains.append(yn)

    val = [contains.index(temp) for temp in contains if temp>0]
    print(contains)
    return val[0]

black_img = np.empty([y,X,3],dtype=np.uint8)
black_img.fill(1)

index = find_contour(contours)
cnt = contours[index]
mask = cv2.drawContours(black_img, [cnt] , 0, (255,255,255), -1)

maskedImg = cv2.bitwise_and(resized_image, mask)

white_pix = [255,255,255]
black_pix = [0,0,0]

final_img = maskedImg
h,w,channels = final_img.shape
for x in range(0,w):
    for y in range(0,h):
        channels_xy = final_img[y,x]
        if all(channels_xy == black_pix):    
            final_img[y,x] = white_pix

params = cv2.SimpleBlobDetector_Params()
params.blobColor = (170+42+42)/3
params.filterByColor = True

detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(final_img, None)

im_with_keypoints = cv2.drawKeypoints(final_img, keypoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow(im_with_keypoints)