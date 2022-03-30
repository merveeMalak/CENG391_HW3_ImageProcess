#260201043-Merve Malak
# I got help from this page while doing this part of the homework : 
#https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

for i in range(5):
    file_sift_image1 = open(f"sift_0{i}.txt")
    img1 = cv.imread(f"../data/goldengate/goldengate-0{i}.png", 0)
    points1= []
    for line in (file_sift_image1.readlines()):
        if line.startswith("Descriptor"):
            break
        else:
            points1.append(line.strip())
    image1_keypoints = []
    for line in points1:
        point = line.split(',')
        kp = cv.KeyPoint(x=float(point[0]), y=float(point[1]), size=float(point[2]), angle=float(point[3]),
            response=float(point[4]), octave=int(point[5]), class_id=int(point[6]))
        image1_keypoints.append(kp)

    file_sift_image2 =  open(f"sift_0{i+1}.txt")
    img2 = cv.imread(f"../data/goldengate/goldengate-0{i+1}.png", 0)
    points2 = []
    for line in (file_sift_image2.readlines()):
        if line.startswith("Descriptor"):
            break
        else:
            points2.append(line.strip())
    image2_keypoints = []
    for line in points2:
        point = line.split(',')
        kp = cv.KeyPoint(x=float(point[0]), y=float(point[1]), size=float(point[2]), angle=float(point[3]),
            response=float(point[4]), octave=int(point[5]), class_id=int(point[6]))
        image2_keypoints.append(kp)


    file_tentative_correspondences =open(f"tentative_correspondences_{i}-{i+1}.txt", "r")
    tentative_correspondences=[]
    for line in file_tentative_correspondences.readlines():
        crp_point = line.split(",")
        crp = cv.DMatch(int(crp_point[0]), int(crp_point[1]), int(crp_point[2]), float(crp_point[3]))
        tentative_correspondences.append(crp)
    file_tentative_correspondences.close()
    kpoints1 = np.zeros((len(tentative_correspondences), 2), dtype=np.float32)
    kpoints2 = np.zeros((len(tentative_correspondences), 2), dtype=np.float32)
    for j, match in enumerate(tentative_correspondences):
        kpoints1[j, :] = image1_keypoints[match.queryIdx].pt
        kpoints2[j, :] = image2_keypoints[match.trainIdx].pt

    hm, mask = cv.findHomography(kpoints1, kpoints2, cv.RANSAC)
    inliers = []
    matches_mask = mask.ravel().tolist()
    for k in range(len(matches_mask)):
        if matches_mask[k] == 1:
            inliers.append(tentative_correspondences[k])
    file_inliers = open(f"../data/inliers_{i}-{i+1}.txt", "w")
    file_inliers.write(str(inliers))
    file_inliers.close()
    file_h = open(f"../data/h_{i}-{i+1}.txt", "w")
    file_h.write(str(hm))
    file_h.close()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,hm)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matches_mask, # draw only inliers
                   flags = 2)
    img3 = cv.drawMatches(img1,image1_keypoints,img2,image2_keypoints,tentative_correspondences,None,**draw_params)
    cv.imwrite(f"../data/inliers_{i}-{i+1}.png", img3)


