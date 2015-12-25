import sys
import glob
import cv2
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--web', action='store_true', default=False)
parser.add_argument('-o', default='output.txt')

namespace = parser.parse_args(sys.argv[1:])
print namespace

images = []

if namespace.web:
    video = cv2.VideoCapture(0)

    while True:
        flag, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.flip(gray, 1)
        cv2.imshow('frame', gray)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('w'):
            images.append(gray)
            if len(images) >= 20:
                break
        elif k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
else:
    img_names = glob.glob('*.jpg')
    images = [cv2.imread(fn, 0) for fn in img_names]

square_size = 0.3
pattern_size = (8, 6)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []
h, w = 0, 0
shape = images[0].shape[::-1]
k = 1

for img in images:
    print "Img:", k
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), term)

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

    k += 1
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, shape, None, None)

print mtx
print dist

