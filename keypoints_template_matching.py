import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *


model_img = cv2.imread('model.jpg', 0)
model_img = cv2.pyrDown(model_img)
model_img = cv2.pyrDown(model_img)

akaze = cv2.AKAZE_create()
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
kp_model, des_model = akaze.detectAndCompute(model_img, None)
obj = OBJ('key.obj', swapyz=True)


cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_frame, des_frame = akaze.detectAndCompute(gray, None)
    matches = matcher.knnMatch(des_model, des_frame, 2)
    good = []
    nn_match_ratio = 0.9
    for m, n in matches:
        if m.distance < nn_match_ratio * n.distance:
            good.append(m)

    if len(good) > 21:
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = model_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        if M is not None:
            projection = projection_matrix(np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]]), M)
            frame = render(frame, obj, projection, model_img, False)

    cv2.imshow('Result', frame)
    if cv2.waitKey(1) == ord('q') & 0xFF:
        break

cv2.destroyAllWindows()
