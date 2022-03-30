#260201043-Merve Malak
import cv2 as cv
import numpy as np

def detect_keypoints(image_number):
    img = cv.imread(f"../data/goldengate/goldengate-0{image_number}.png")
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp,des = sift.detectAndCompute(gray,None)
    return kp,des

def draw_and_write_sift(image_number):
    img = cv.imread(f"../data/goldengate/goldengate-0{image_number}.png")
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kp,des = detect_keypoints(image_number)
    img=cv.drawKeypoints(gray,kp,img)
    cv.imwrite(f'sift_keypoints0{image_number}.png',img)
    f = open(f"sift_0{image_number}.txt", "w")
    for point in kp:
        p = str(point.pt[0]) + "," + str(point.pt[1]) + "," + str(point.size) + "," + str(point.angle) + "," + str(
        point.response) + "," + str(point.octave) + "," + str(point.class_id) + "\n"
        f.write(p)
    f.write(f"Descriptor:\n {des}")
    f.close()

def bfc_matches(image_number1,image_number2):
    img1 = cv.imread(f"../data/goldengate/goldengate-0{image_number1}.png")
    img2 = cv.imread(f"../data/goldengate/goldengate-0{image_number2}.png")
    gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp1, des1  = sift.detectAndCompute(gray1,None)        
    kp2, des2 = sift.detectAndCompute(gray2,None)
    bf = cv.BFMatcher()
    matches = bf.match(des1,des2)
    kmatches = sorted(matches, key = lambda x:x.distance)
    return kmatches

def draw_and_write_tentative_correspendences(image_number1, image_number2):
    img1 = cv.imread(f"../data/goldengate/goldengate-0{image_number1}.png")
    img2 = cv.imread(f"../data/goldengate/goldengate-0{image_number2}.png")
    matches = bfc_matches(image_number1, image_number2)
    f = open(f"tentative_correspondences_{image_number1}-{image_number2}.txt", "w")
    for match in matches:
        m = str(match.queryIdx) + "," + str(match.trainIdx) + "," + str(match.imgIdx) + "," + str(match.distance)+"\n" 
        f.write(m)
    f.close() 
    kp1, _ = detect_keypoints(image_number1)
    kp2, _ = detect_keypoints(image_number2)
    matched_img = cv.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
    cv.imwrite(f"tentative_correspondences_{image_number1}-{image_number2}.png", matched_img)


def main():
    for i in range(6):
        draw_and_write_sift(0)
    for i in range(5):
        draw_and_write_tentative_correspendences(i, i+1)

main()