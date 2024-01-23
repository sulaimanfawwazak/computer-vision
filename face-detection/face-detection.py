import cv2 as cv
import math

def faceTracking():
    res1 = (320, 240)
    res2 = (640, 480)
    res3 = (1280, 720)
    res = res3

    faceCascade = cv.CascadeClassifier('haarcascade_frontal_default.xml')
    
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, res[0])
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, res[1])

    frame_counter = 0
    current_id = 0
    face_trackers = {} # Creates a dictionary of faces

    WIDTH = res[0]/2
    HEIGHT = res[1]/2
    EYE_DEPTH = 2
    horizontal_FOV = 62/2
    vertical_FOV = 49/2
    ppcm = WIDTH*2/15.5
    term = False

    while not term:
        ret, frame = capture.read()
        frame_counter += 1
        if frame_counter % 1 == 0:
            greyscale = cv.cvtColor(cv.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                greyscale,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (30, 30),
                flags= cv.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                center = (int(x+w/2), int(y+h/2))
                face_id_match = False
                for face_id in face_trackers.keys():
                    (tx, ty, tw, th, n, u) = face_trackers.get(face_id)
                    if tx <= center[0] <= tx+tw and ty <= center[1] <= ty+th:
                        if n < 50: n += 1
                        face_trackers.update({face_id:(x, y, w, h, n, True)})
                        face_id_match = True
                        break
                if not face_id_match:
                    face_trackers.update({current_id:(x, y, w, h, 1, True)})
                    current_id += 1

        track_id = -1
        face_id_to_delete = []
        for face_id in face_trackers.keys():
            (tx, ty, tw, th, n, u) = face_trackers.get(face_id)
            if not u: n -= 1
            if n < 1: face_id_to_delete.append(face_id)
            else:
                face_trackers.update({face_id:(tx, ty, tw, th, n, False)})
                if n < 25:
                    pass
                else:
                    track_id = face_id

        for face_id in face_id_to_delete:
            face_trackers.pop(face_id, None)

        if track_id != -1:

            # Determine who to track
            (x, y, w, h, n, u) = face_trackers.get(track_id)
            center = (int(x+w/2), int(y+h/2))
            hAngle








