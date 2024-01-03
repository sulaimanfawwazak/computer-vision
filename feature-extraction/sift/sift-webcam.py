#!/usr/bin/env python3

import cv2 as cv
import time

# Make SIFT instance
sift = cv.SIFT_create()

# Make Feature Matcher
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

vid_capture = cv.VideoCapture(0)

while vid_capture.isOpened():
    # Read the images
    _, video_1 = vid_capture.read()

    # Make a copy of the video
    video_2 = video_1

    start = time.time()

    # Convert the color space
    video_1 = cv.cvtColor(video_1, cv.COLOR_BGR2GRAY)
    video_2 = cv.cvtColor(video_2, cv.COLOR_BGR2GRAY)

    # Keypoints & Descriptors
    keypoints_1, descriptors_1 = sift.detectAndCompute(video_1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(video_2, None)

    # Match the feature
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x:x.distance)

    # Calculate the time elapsed
    end = time.time()
    timeElapsed = end - start

    # Calculate the FPS
    fps = 1/timeElapsed

    # Display the resulting SIFT process
    result_video = cv.drawMatches(video_1, keypoints_1, video_2, keypoints_2, matches[200:500], video_2, flags=2)
    cv.putText(result_video, f'FPS: {int(fps)}', (20, 450), cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
    cv.imshow('SIFT', result_video)

    if cv.waitKey(5) == ord(' '):
        break