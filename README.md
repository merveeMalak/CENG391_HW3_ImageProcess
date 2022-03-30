# CENG391_HW3_ImageProcess

Exercise 1 Image Processing and Feature Detection (40 points)

Please do the following exercises by a single Python script named as src/detect and match.py. You may use OpenCV for feature detection and descriptor computation.

a. Detect SIFT interest points on the six images of the Golden Gate Bridge that are in the folder data.

b. Draw the SIFT interest points on each image and store the resulting images in the same folder with names as sift keypoints i.png, where i is the image number.

c. Calculate SIFT descriptor matches between consecutive pairs of images by brute force matching, for example between goldengate-00.png and goldengate-01.png, between goldengate-01.png and goldengate-02.png, and so on.

d. Draw these tentative correspondences on a match image and save the resulting images in the same folder with names as tentative correspondences i-j.png, where i and j are image numbers.

e. Save the SIFT interest points, descriptors, and tentative correspondences as text files in the same folder with names as sift i.txt and tentative correspondences i-j.txt.

Exercise 2 RANSAC (40 points)

Please do the following exercises by a single Python script named as src/ransac.py. You may use OpenCV for homography computation with RANSAC.

a. Read the keypoints and tentative correspondences for each image pair and match them by RANSAC.

b. You may use RANSAC from OpenCV, implement RANSAC yourself for 10 bonus points.

c. Save the resulting homography matrices in files within the folder data with names such as h i-j.txt, where i and j are image numbers.

d. Do not forget about normalization and the final estimation over all inliers. You may optionally perform guided matching.

e. Draw and save the resulting final inlier correspondences in files in the data folder with names as inliers i-j.png and inliers i-j.txt.

Exercise 3 Basic Stitching (20 points)

Please do the following exercises by a single Python script named as src/stitch.py. You may use OpenCV function warp perspective for image warping.

a. Stitch all the images by calculating a homography matrix from each image to one of the center images goldengate-02.png or goldengate-03.png and warping the images to this coordinate system.

b. Save the resulting image in the folder data named as panorama.png.

c. To blend multiple images just overwrite or average intensities of overlapping pixels.
