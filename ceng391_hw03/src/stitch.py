# 260201043-Merve Malak
# I got help from this page while doing this part of the homework : 
# https://stackoverflow.com/questions/68836179/how-make-panoramic-view-of-serie-of-pictures-using-python-opencv-orb-descripto
import cv2 as cv
import numpy as np


def warpImages(img1, img2, h):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts_img2_tmp = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts_img2 = cv.perspectiveTransform(pts_img2_tmp, h)
    points = np.concatenate((pts_img1, pts_img2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    new_img = cv.warpPerspective(img2, H_translation.dot(h), (x_max - x_min, y_max - y_min))
    new_img[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = img1

    return new_img

def panorama():
    sift = cv.SIFT_create()
    img2 = cv.imread("../data/goldengate/goldengate-03.png")
    for i in range(6):
        if (i == 3):
            continue
        else:
            img1 = cv.imread(f"../data/goldengate/goldengate-0{i}.png")
            gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
            img1_kp, des1 = sift.detectAndCompute(gray1,None) 
            img2_kp, des2 = sift.detectAndCompute(gray2, None)
            matches1_2 = cv.BFMatcher().match(des1, des2)
            kpoints1  = np.float32([img1_kp[m.queryIdx].pt for m in matches1_2]).reshape(-1, 1, 2)  
            kpoints2 = np.float32([img2_kp[m.trainIdx].pt for m in matches1_2]).reshape(-1, 1, 2)
            h, mask = cv.findHomography(kpoints1, kpoints2, cv.RANSAC)
            img2 = warpImages(img2, img1, h)
    cv.imwrite("../data/panorama.png", img2)


panorama()