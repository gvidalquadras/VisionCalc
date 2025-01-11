import cv2
import os
from typing import List
import numpy as np
import imageio
import copy
import glob

def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames] # On the future imageio.v2.imread

def get_chessboard_points(chessboard_shape, dx, dy):
    
    points = []
    
    for i in range(chessboard_shape[1]):
        for j in range(chessboard_shape[0]):
            points.append(np.array([i*dy, j*dx, 0]))
            
    return np.array(points, dtype = np.float32)

def write_image(counter_image, img):
    cv2.imwrite(f"../calibration/corners/detected_{counter_image}.jpg",img) # f"../data/left_detected/detected_{counter_image}.jpg"


imgs_path = [path for path in glob.glob( "../calibration/images/*jpg" )] # "../data/left/*jpg" 
imgs = load_images(imgs_path)

corners = [cv2.findChessboardCorners(img, (8,6)) for img in imgs]
corners_copy = copy.deepcopy(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

corners_refined = [cv2.cornerSubPix(i, cor[1], (8, 6), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

imgs_copy = copy.deepcopy(imgs)
drawed_corners = [cv2.drawChessboardCorners( imgs_copy[i], (8,6), corners_refined[i], corners[i][0])  for i in range(1, len(corners)) if i!= 14]

os.makedirs("../calibration/corners", exist_ok = True) # "../data/left_detected"

for i in range(0, len(imgs_copy)): 
    if i != 14:
        cv2.imwrite(f"../calibration/corners/detected_{i}.jpg",imgs_copy[i])


chessboard_points = [get_chessboard_points((8, 6), 30, 30) for i in range(1, len(corners_refined))] # One per each image with corners detected


valid_corners = [cor[1] for cor in corners if cor[0]]
valid_corners = np.asarray(valid_corners, dtype=np.float32)


rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, np.concatenate([corners_refined[:14], corners_refined[15:]]), imgs_gray[-1].shape[::-1], None, None) # corners_refined[1:]
extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)
print(len(corners_refined))