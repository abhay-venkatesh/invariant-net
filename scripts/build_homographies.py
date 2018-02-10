"""
Build Homographies.

Algorithm:

    Traverse 
        view0
        view1

    Compute homographies between
        view0 image
        view1 image

WARNING: USE OPENCV 2.4 FOR THIS SCRIPT.
NOTE: RUN FROM THE MAIN DIRECTORY TO GET THE PATHS TO WORK
"""
import numpy as np 
import cv2
import itertools
from matplotlib import pyplot as plt
import os

class HomographyComputer:
    def __init__(self):
        self.H = np.identity(3)
        pass

    def hamming_homography(self, A_path, B_path):
        try:
            img1 = cv2.imread(A_path) # Original image, queryImage
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.imread(B_path) # Rotated image, trainImage
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Create ORB detector with 1000 keypoints with a scaling pyramid factor
            # of 1.2
            orb = cv2.ORB(1000, 1.2)

            # Detect keypoints of original image
            (kp1,des1) = orb.detectAndCompute(img1, None)

            # Detect keypoints of rotated image
            (kp2,des2) = orb.detectAndCompute(img2, None)

            # Create matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Do matching
            matches = bf.match(des1,des2)

            # Sort the matches based on distance.  Least distance
            # is better
            matches = sorted(matches, key=lambda val: val.distance)

            img1_matches = []
            img2_matches = []
            length_of_matches = min(len(matches), 10)
            for match in matches[:length_of_matches]:
                img1_matches.append(kp1[match.queryIdx].pt)
                img2_matches.append(kp2[match.trainIdx].pt)

            img1_matches = np.array(img1_matches)
            img2_matches = np.array(img2_matches)

            self.H, status = cv2.findHomography(img1_matches, img2_matches)
            return True
            
        except:
            print(A_path)
            print(B_path)
            self.H = np.identity(3)
            return False

    def apply_homography(self, A_path, out_path, should_write=True):
        if should_write:
            im_src = cv2.imread(A_path)
            im_out = cv2.warpPerspective(im_src, self.H, 
                                         (im_src.shape[1],im_src.shape[0]))
            cv2.imwrite(out_path, im_out)
        else:
            pass

view0_path = './datasets/Unreal-20View-11class/view0/'
view1_path = './datasets/Unreal-20View-11class/view1/'

homography1_path = './datasets/Unreal-20View-11class/homography1/'
homography1_images_path = homography1_path + "images/"
homography1_ground_truths_path = homography1_path + "ground_truths/"
if not os.path.exists(homography1_path):
    os.makedirs(homography1_path)
if not os.path.exists(homography1_images_path):
    os.makedirs(homography1_images_path)
if not os.path.exists(homography1_ground_truths_path):
    os.makedirs(homography1_ground_truths_path)

view0_image_directory = view0_path + "images/"
view1_image_directory = view1_path + "images/"
homography1_length = min(len(next(os.walk(view0_image_directory))[2]),
                         len(next(os.walk(view1_image_directory))[2]))

view0_ground_truth_directory = view0_path + "ground_truths/"
view1_ground_truth_directory = view1_path + "ground_truths/"

hc = HomographyComputer()
j = 0
for i in range(homography1_length):

    view0_image_path = view0_image_directory + "pic" + str(i) + ".png"
    view1_image_path = view1_image_directory + "pic" + str(i) + ".png"
    homography1_image_path = homography1_images_path + "pic" + str(j) + ".png"

    view0_ground_truth_path = view0_ground_truth_directory + \
                              "seg" + str(i) + ".png"
    view1_ground_truth_path = view1_ground_truth_directory + \
                              "seg" + str(i) + ".png"
    homography1_ground_truth_path = homography1_ground_truths_path + \
                                    "seg" + str(j) + ".png"

    if not os.path.exists(homography1_image_path):
        should_write = hc.hamming_homography(view0_image_path, view1_image_path)
        hc.apply_homography(view0_image_path, homography1_image_path, should_write)
        hc.apply_homography(view0_ground_truth_path, 
                            homography1_ground_truth_path,
                            should_write)

    if should_write:
        j += 1