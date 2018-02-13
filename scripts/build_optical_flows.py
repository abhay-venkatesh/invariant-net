"""

NOTE: Use OpenCV 2.4, Run from MAIN DIRECTORY

"""
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

class FlowComputer:
    def __init__(self):
        pass

    def compute_flow(self, image_1_path, image_2_path, 
                     ground_truth_1_path, output_image_path, output_ground_truth_path):
        frame1 = cv2.imread(image_1_path)
        frame2 = cv2.imread(image_2_path)
        old_gt = cv2.imread(ground_truth_1_path)
        old_gt = cv2.cvtColor(old_gt,cv2.COLOR_BGR2GRAY)
        previous_frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, 0.5, 3, 15, 3, 5, 1.2, 0)

        height = flow.shape[0]
        width = flow.shape[1]
        R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
        pixel_map = R2 + flow

        pixel_map_x = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                pixel_map_x[i][j] = pixel_map[i][j][0]

        pixel_map_y = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                pixel_map_y[i][j] = pixel_map[i][j][1]

        pixel_map_x_32 = pixel_map_x.astype('float32')
        pixel_map_y_32 = pixel_map_y.astype('float32')

        new_gt = cv2.remap(old_gt, pixel_map_x_32, pixel_map_y_32, cv2.INTER_NEAREST)
        cv2.imwrite(output_ground_truth_path, new_gt)
        cv2.imwrite(output_image_path, frame2)
   

for k in range(0,1):
    # PARAMETER: Change the two view folder locations
    view0_path = './datasets/Unreal-20View-11class/view' + str(k) + '/'
    view1_path = './datasets/Unreal-20View-11class/view' + str(k+1) + '/'

    # PARAMETER: Change the homography1_path to the desired output folder
    homography1_path = './datasets/Unreal-20View-11class/opticalflow' + str(k+1) + '/'
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

    fc = FlowComputer()
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

        print("Working on " + homography1_image_path)
        if not os.path.exists(homography1_image_path):
            fc.compute_flow(view0_image_path, view1_image_path, 
                            view0_ground_truth_path, homography1_image_path,
                            homography1_ground_truth_path)
            j += 1