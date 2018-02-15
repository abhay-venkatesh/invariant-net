"""
Algorithm:

    for i = 0 to i = 499:
        For each opticalflow directory:
            pic one img/seg pair at random
            write to flow1/ dir with image/seg      

NOTE: RUN FROM MAIN DIRECTORY FOR PATHS
"""
import cv2
import numpy as np
import os
import csv
from shutil import copyfile
from random import randint

# Build output directories
DATASET_NAME = 'UnrealFlows'
output_directory = './datasets/' + DATASET_NAME + '/' 

def copy_file(dir_name, file_name, counter, i):
    file_path = dir_name + '/' + file_name
    print("Working on " + file_path)
    seg_write_path = './datasets/' + DATASET_NAME + '/view' + str(i) + \
              '/ground_truths/seg' + str(counter) + '.png'
    print("Writing to " + seg_write_path)
    copyfile(file_path, seg_write_path)
    image_file_path = './datasets/' + 'Unreal-20View-11class' + '/opticalflow' + str(i) + \
                    '/images/' + file_name.replace('seg', 'pic')
    image_write_path = './datasets/' + DATASET_NAME + '/view' + str(i) + \
                    '/images/pic' + str(counter) + '.png'
    copyfile(image_file_path, image_write_path)

if not os.path.exists(output_directory):
    os.makedirs(output_directory) 

for i in range(1,20):

    subsequence_directory = './datasets/' + DATASET_NAME + '/view' + \
                             str(i) + '/'
    if not os.path.exists(subsequence_directory):
        os.makedirs(subsequence_directory)
    ground_truths_directory = subsequence_directory + 'ground_truths/'
    if not os.path.exists(ground_truths_directory):
        os.makedirs(ground_truths_directory) 
    scene_image_directory = subsequence_directory + '/images/'
    if not os.path.exists(scene_image_directory):
        os.makedirs(scene_image_directory) 

    source_directory = "./datasets/Unreal-20View-11class/opticalflow" + str(i) + "/"

    
    # Walk through the DatasetSource and pick samples
    for dirName, subdirList, fileList in os.walk(source_directory):
        print('Found directory: %s' % dirName)

        counter = 0
        for fname in fileList:
            if randint(0,4) == 1:
                desired_file_suffix = 'seg' + str(i) + '.png'

                image_write_path = './datasets/' + DATASET_NAME + '/view' + str(i) + \
                                   '/images/pic' + str(counter) + '.png'
                if not os.path.exists(image_write_path):
                    copy_file(dirName, fname, counter, i)
                    counter += 1

                if counter == 500:
                    break

        