#!/usr/bin/python3

import numpy as np
import cv2 as cv
import glob
import imutils

# Image paths
image_paths = glob.glob('unstitchedImages/*.JPG')
images = []

for image in image_paths:
    img = cv.imread(image)
    img = cv.resize(img, (0, 0), fx=0.3, fy=0.3)
    images.append(img)

    cv.imshow(f'Image: {image}', img)
    cv.waitKey(0)

imageStitcher = cv.Stitcher_create()

error, stitched_img = imageStitcher.stitch(images)

if not error:
    cv.imwrite('stitched-image.JPG', stitched_img)
    cv.imshow('stitched-image.JPG', stitched_img)
    cv.waitKey(0)