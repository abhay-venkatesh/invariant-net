import numpy as np
import cv2
import os
import random
from utils.ImageResizer import ImageResizer 
from utils.RecordFileGenerator import RecordFileGenerator
from random import randint

class MultiBatchDatasetReader:
    """ 
        Helper class to SegNet that handles data reading, conversion 
        and all things related to data 
    """

    def __init__(self, directory, WIDTH, HEIGHT, current_step, batch_size,
                 resize=False):

        # Save variables
        self.batch_size = batch_size
        self.current_step = current_step
        self.base_directory = directory
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.num_train = 0
        self.num_val = 0
        self.num_test = 0

        # Prepare record files for each view
        for i in range(1,20):
            curr_directory = self.base_directory + 'view' + str(i) + '/'
            rfg = RecordFileGenerator(curr_directory)
            curr_num_train, curr_num_val, curr_num_test = rfg.create_files()
            self.num_train += curr_num_train
            self.num_val += curr_num_val
            self.num_test += curr_num_test

        # Say we have 100 training images
        # Say we are at training step 10
        # Then, our index in our training images should be 
        # (10 * 5) % 100 = 50
        # This corresponds to, 
        self.train_index = (current_step * self.batch_size) % self.num_train

        # Read dataset items
        self.current_view = 1
        self.curr_directory = self.base_directory + 'view' + str(self.current_view) + '/'
        self.training_data = open(self.curr_directory + 'train.txt').readlines()
        self.validation_data = open(self.curr_directory + 'val.txt').readlines()
        self.test_data = open(self.curr_directory + 'test.txt').readlines()

    def shuffle_training_data(self):
        lines = open(self.curr_directory + 'train.txt').readlines()
        random.shuffle(lines)
        open(self.curr_directory + 'train.txt', 'w').writelines(lines)
        self.training_data = open(self.curr_directory + 'train.txt').readlines()

    def next_training_batch(self):
        images = []
        ground_truths = []

        for i in range(self.batch_size):

            if self.train_index == 0:
                self.shuffle_training_data()

            # Load image
            image_directory = self.curr_directory + 'images/'
            print("Loading images from " + self.curr_directory)
            image_file = self.training_data[self.train_index].rstrip()
            image = cv2.imread(image_directory + image_file)
            image = np.float32(image)
            # With 0.5 probability, flip the image/ground truth pair
            # for data augmentation
            random_number = randint(0,1)
            if random_number == 1:
                image = np.fliplr(image)
            images.append(image)

            # Load ground truth
            ground_truth_directory = self.curr_directory + 'ground_truths/'
            ground_truth_file = image_file.replace('pic', 'seg')
            ground_truth = cv2.imread((ground_truth_directory + 
                                       ground_truth_file), cv2.IMREAD_GRAYSCALE)
            ground_truth = ground_truth/8
            if random_number == 1:
                ground_truth = np.fliplr(ground_truth)
            ground_truths.append(ground_truth)

            # Update training index
            self.train_index += 1
            self.train_index %= self.num_train

        self.current_view += 1
        if self.current_view == 20:
            self.current_view = 1
        self.curr_directory = self.base_directory + 'view' + str(self.current_view) + '/'

        returning_view = self.current_view - 1
        if returning_view == 0:
            returning_view = 19
            
        return images, ground_truths, returning_view


    def next_val_batch(self):
        images = []
        ground_truths = []

        for i in range(self.batch_size):
            # Load image
            image_directory = self.curr_directory + 'images/'
            image_file = random.choice(self.validation_data).rstrip()
            image = cv2.imread(image_directory + image_file)
            image = np.float32(image)
            images.append(image)

            # Load ground truth
            ground_truth_directory = self.curr_directory + 'ground_truths/'
            ground_truth_file = image_file.replace('pic', 'seg')
            ground_truth = cv2.imread((ground_truth_directory + 
                                       ground_truth_file), cv2.IMREAD_GRAYSCALE)
            ground_truth = ground_truth/8
            ground_truths.append(ground_truth)

        self.current_view += 1
        if self.current_view == 20:
            self.current_view = 1
        self.curr_directory = self.base_directory + 'view' + str(self.current_view) + '/'

        returning_view = self.current_view - 1
        if returning_view == 0:
            returning_view = 19
            
        return images, ground_truths, returning_view


def main():
    pass

if __name__ == "__main__":
    main()
