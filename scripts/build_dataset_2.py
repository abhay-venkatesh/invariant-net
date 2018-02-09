"""
Script that prepares a dataset for training ThetaDFSegNet.
We will split the dataset into 20 pieces - one piece per view.
We will call it Unreal-20View-11class.
"""
import json
import cv2
import numpy as np
from ast import literal_eval
from shutil import copyfile
import os

DATASET_NAME = 'Unreal-20View-11class'

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

def convert_image(dirName, fname, counter):
    ''' 
        Args:
            dirName: Name of directory to output to
            fname: Name of file getting converted
            counter: Index of the file in the dataset

        Output:
            Converts colored segmentation to grayscale,
            and writes to the output directory. 

    Writes the converted image to the data folder as well.
    Output image has pixel values from 0-27, 0 if no class
    and 1-27 are classes as described in the finalClassesToInt json file. '''
    convertedPath = os.path.abspath(dirName)
    filePath = convertedPath + '\\' + fname
    print(filePath)

    img = cv2.imread(filePath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    [m,n] = img.shape[:2]
    res = np.zeros((m,n))
    print("Working on" + filePath)

    for key in color_map:
        match_region=match_color(img,key)
        if not (match_region is None):
            res = ((np.multiply(res, ~match_region)) + 
                    match_region*color_map[key])

    outfile = '../datasets/' + DATASET_NAME + \
                        '/ground_truths/seg' + str(counter) + '.png' 
    cv2.imwrite(outfile,res*8)
    image_outfile = '../datasets/' + DATASET_NAME + \
                                    '/images/pic' + str(counter) + '.png'
    copyfile(filePath.replace('seg','pic'), image_outfile) 

def build():

    # Build output directories
    output_directory = '../datasets/' + DATASET_NAME + '/' 
    if not os.path.exists(output_directory):
        os.makedirs(output_directory) 

    for i in range(20):

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

        source_directory = "../../../../UnrealEngineSourceResized/batch0/"

        counter = 0
        # Walk through the DatasetSource and pick samples
        for dirName, subdirList, fileList in os.walk(source_directory):
            print('Found directory: %s' % dirName)

            for fname in fileList:
                desired_file_suffix = str(i) + '.jpg'
                if desired_file_suffix in fname:
                    convert_image(dirName, fname, counter)
                    counter += 1


def print_file_walk():
    source_directory = "../../../../UnrealEngineSource/"

    # Walk through the DatasetSource and pick samples
    counter = 0
    non_street_view_count = 0
    for dirName, subdirList, fileList in os.walk(source_directory):
        sequence_counter = 0
        for fname in fileList:
            # For first 50 scenes, print the whole sequence of scenes
            if non_street_view_count < 50 and "seg" in fname:
                print(dirName + '\\' + fname)
                sequence_counter += 1
                if sequence_counter == 20:
                    counter += 20
                    sequence_counter = 0
                    non_street_view_count += 1
            elif fname == 'seg0.png':
                print(dirName + '\\' + fname)
                counter += 1

def main():
    build() 
    # print_file_walk()

if __name__ == "__main__":
    main()
