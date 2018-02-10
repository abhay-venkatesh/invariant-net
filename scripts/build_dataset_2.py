"""
Script that prepares a dataset for training ThetaDFSegNet.
We will split the dataset into 20 pieces - one piece per view.
We will call it Unreal-20View-11class.

NOTE: Run this from the scripts folder to make sure all the paths work.
"""
import json
import cv2
import numpy as np
from ast import literal_eval
from shutil import copyfile
import os
import csv

DATASET_NAME = 'Unreal-20View-11class'
WIDTH = 480
HEIGHT = 320

# Match an i
def match_color(object_mask, target_color, tolerance=3):
    match_region = np.ones(object_mask.shape[0:2], dtype=bool)
    for c in range(3):
        min_val = target_color[c] - tolerance
        max_val = target_color[c] + tolerance
        channel_region = ((object_mask[:,:,c] >= min_val) & 
                                            (object_mask[:,:,c] <= max_val))
        match_region &= channel_region

    if match_region.sum() != 0:
        return match_region
    else:
        return None

# Get color to class and class to number maps
# Used in convert_image
color2class = json.load(open('../dat/reducedColorsToClasses.json','r'))
class2num = json.load(open('../dat/reducedClassesToInt.json','r'))
color_map = {}
for color in color2class:
    color_map[literal_eval(color)] = class2num[color2class[color]]              

def convert_image(dir_name, file_name, counter, i):
    """ 
        Args:
            dir_name: Name of directory to output to
            file_name: Name of file getting converted
            counter: Index of the file in the dataset
            i: View Index

        Output:
            Converts colored segmentation to grayscale,
            and writes to the output directory. 

        Writes the converted image to the data folder as well.
    """
    converted_dir_name = os.path.abspath(dir_name)
    file_path = converted_dir_name + '\\' + file_name
    print(file_path)

    seg_img = cv2.imread(file_path)
    seg_img = cv2.cvtColor(seg_img,cv2.COLOR_BGR2RGB)
    seg_img = cv2.resize(seg_img, (WIDTH, HEIGHT), 
                         interpolation=cv2.INTER_NEAREST)
    [m,n] = seg_img.shape[:2]
    seg = np.zeros((m,n))

    print("Working on" + file_path)

    for key in color_map:
        match_region=match_color(seg_img,key)
        if not (match_region is None):
            seg = ((np.multiply(seg, ~match_region)) + 
                    match_region*color_map[key])

    seg_write_path = '../datasets/' + DATASET_NAME + '/view' + str(i) + \
              '/ground_truths/seg' + str(counter) + '.png' 

    print("Writing to " + seg_write_path)

    cv2.imwrite(seg_write_path, seg*8)

    image_write_path = '../datasets/' + DATASET_NAME + '/view' + str(i) + \
                    '/images/pic' + str(counter) + '.png'
    image = cv2.imread(file_path.replace('seg','pic'))
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(image_write_path, image)             

def build():

    # Build output directories
    output_directory = '../datasets/' + DATASET_NAME + '/' 
    if not os.path.exists(output_directory):
        os.makedirs(output_directory) 

    for i in range(1,2):

        subsequence_directory = '../datasets/' + DATASET_NAME + '/view' + \
                                 str(i) + '/'
        if not os.path.exists(subsequence_directory):
            os.makedirs(subsequence_directory)
        ground_truths_directory = subsequence_directory + 'ground_truths/'
        if not os.path.exists(ground_truths_directory):
            os.makedirs(ground_truths_directory) 
        scene_image_directory = subsequence_directory + '/images/'
        if not os.path.exists(scene_image_directory):
            os.makedirs(scene_image_directory) 
            
        with open(subsequence_directory + 'id-round-mappings', 'w', newline='') as csvfile:
            id_round_mapping_writer = csv.writer(csvfile, delimiter=',')

            source_directory = "../../../../UnrealEngineSource/"

            counter = 0
            # Walk through the DatasetSource and pick samples
            for dirName, subdirList, fileList in os.walk(source_directory):
                print('Found directory: %s' % dirName)

                for fname in fileList:
                    desired_file_suffix = 'seg' + str(i) + '.png'
                    if desired_file_suffix in fname:
                        round_substring_index = dirName.find("round")
                        round_number_index = round_substring_index + 5

                        id_round_mapping_writer.writerow([counter, dirName[round_number_index:]])
                        image_write_path = '../datasets/' + DATASET_NAME + '/view' + str(i) + \
                                           '/images/pic' + str(counter) + '.png'
                        if not os.path.exists(image_write_path):
                            convert_image(dirName, fname, counter, i)
                        counter += 1

def main():
    build() 

if __name__ == "__main__":
    main()
