import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

video = cv2.VideoCapture(0)
# cv2.namedWindow('Result', cv2.WINDOW_KEEPRATIO)

akaze = cv2.AKAZE_create()
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
obj = OBJ('fox.obj', swapyz=True)

prev_frame = None
prev_kp = None
prev_des = None
prev_M = None

while True:
    ret, frame = video.read()
    height, width = frame.shape[:2]
    canvas = frame.copy()
    if not ret:
        break
    if prev_frame is None:
        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_kp, prev_des = akaze.detectAndCompute(prev_frame, None)
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = akaze.detectAndCompute(gray_frame, None)
    matches = matcher.knnMatch(prev_des, des, 2)
    good = []
    prev_pt = []
    pt = []
    nn_match_ratio = 0.7
    for m, n in matches:
        if m.distance < nn_match_ratio * n.distance:
            good.append([m])
            pt1 = prev_kp[m.queryIdx].pt
            pt2 = kp[m.trainIdx].pt
            prev_pt.append(pt1)
            pt.append(pt2)
    if len(good) > 12:
        prev_pt = np.float32(prev_pt).reshape(-1, 1, 2)
        pt = np.float32(pt).reshape(-1, 1, 2)
        M, st = cv2.findHomography(prev_pt, pt, cv2.RANSAC, 5.0)
        if prev_M is None:
            prev_M = M
            continue
        new_M = np.dot(M, prev_M)
        projection = projection_matrix(np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]]), new_M)
        res = render(frame, obj, projection, gray_frame, False)

        cv2.imshow('Result', res)
        if cv2.waitKey(1) == ord('q') & 0xFF:
            break

        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_kp, prev_des = akaze.detectAndCompute(prev_frame, None)
        prev_M = new_M

cv2.destroyAllWindows()
