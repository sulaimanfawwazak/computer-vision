#!/usr/bin/env python3

import cv2 as cv
import numpy as np

# Make SIFT instance/object
sift = cv.SIFT_create()

# Features Matcher (Brute Force Matcher)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True) # L2: Euclidean Distance

# Load the images
img_1 = cv.imread('/home/pwnwas/Downloads/images/DSC02931.JPG')
img_2 = cv.imread('/home/pwnwas/Downloads/images/DSC02932.JPG')

# Resize the images
img_1 = cv.resize(img_1, (0, 0), fx=0.3, fy=0.3)
img_2 = cv.resize(img_2, (0, 0), fx=0.3, fy=0.3)

# Show the images
cv.imshow('Image 1', img_1)
cv.imshow('Image 2', img_2)
cv.waitKey(0)

# Convert the color space of the images
img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

# Keypoints & Descriptor
keypoints_1, descriptors_1 = sift.detectAndCompute(img_1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img_2, None)

# Find the matches
matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

# Draw the lines connecting the features
result_img = cv.drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches[:100], img_2, flags=2)

# Show the resulting image
cv.imshow('Resulting image', result_img)
cv.waitKey(0)