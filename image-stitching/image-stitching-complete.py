#!/usr/bin/env python3

import cv2 as cv
import glob
import imutils
import numpy as np

image_paths = glob.glob('unstitchedImages/*JPG')
images = []

for image in image_paths:
    img = cv.imread(image)

    img = cv.resize(img, (0, 0), fx=0.3, fy=0.3)
    images.append(img)
    cv.imshow(f'Image {image}', img)
    cv.waitKey(0)

imageStitcher = cv.Stitcher_create()

error, stitched_img = imageStitcher.stitch(images)

if not error:
    cv.imshow('Stitched Image', stitched_img)
    cv.waitKey(0)

    stitched_img = cv.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv.BORDER_CONSTANT, (0, 0, 0))

    gray_img = cv.cvtColor(stitched_img, cv.COLOR_BGR2GRAY)
    thresh_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY)[1]

    cv.imshow('Threshold Image', thresh_img)
    cv.waitKey(0)

    # Find the contours
    contours = cv.findContours(thresh_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv.contourArea)

    mask = np.zeros(thresh_img.shape, dtype='uint8')
    x, y, width, height = cv.boundingRect(areaOI)
    cv.rectangle(mask, (x, y), (x + width, y + height), 255, -1)

    minRectangle = mask.copy()
    subtract = mask.copy()

    while cv.countNonZero(subtract) > 0:
        minRectangle = cv.erode(minRectangle, None)
        subtract = cv.subtract(minRectangle, thresh_img)

    contours = cv.findContours(minRectangle.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    areaOI = max(contours, key=cv.contourArea)

    cv.imshow('Min Rectangle Image', minRectangle)
    cv.waitKey(0)

    x, y, width, height = cv.boundingRect(areaOI)

    stitched_img = stitched_img[y:y + height, x:x + width]

    cv.imshow('Final Stitched Image', stitched_img)
    cv.waitKey(0)

else:
    print('Images could not be stitched!')