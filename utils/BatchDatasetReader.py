import numpy as np
import cv2
import os
import random
from utils.ImageResizer import ImageResizer 
from utils.RecordFileGenerator import RecordFileGenerator
from random import randint

class BatchDatasetReader:
    """ 
        Helper class to SegNet that handles data reading, conversion 
        and all things related to data 
    """

    def __init__(self, directory, WIDTH, HEIGHT, current_step, batch_size
                 resize=False):

        # Save variables
        self.batch_size = batch_size
        self.current_step = current_step
        self.directory = directory
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        # Prepare dataset record files
        rfg = RecordFileGenerator(directory)
        self.num_train, self.num_val, self.num_test = rfg.create_files()

        # Say we have 100 training images
        # Say we are at training step 10
        # Then, our index in our training images should be 
        # (10 * 5) % 100 = 50
        # This corresponds to, 
        self.train_index = (current_step * self.batch_size) % self.num_train

        # Resize data if needed
        if resize == True:
            ground_truth_directory = directory + 'ground_truths/'
            ground_truth_output_directory = directory + 'ground_truths_resized/'
            if not os.path.exists(ground_truth_output_directory):  
                print("Resizing dataset........ ")
                ir = ImageResizer(ground_truth_directory, 
                                  ground_truth_output_directory)
                ir.resize_ground_truths(WIDTH, HEIGHT)
                image_directory = directory + 'images/'
                images_output_directory = directory + 'images_resized/'
                ir = ImageResizer(image_directory, images_output_directory)
                ir.resize_images(WIDTH, HEIGHT)
                print("Finished resizing dataset. ")

        # Read dataset items
        self.training_data = open(directory + 'train.txt').readlines()
        self.validation_data = open(directory + 'val.txt').readlines()
        self.test_data = open(directory + 'test.txt').readlines()

    def shuffle_training_data(self):
        lines = open(self.directory + 'train.txt').readlines()
        random.shuffle(lines)
        open(self.directory + 'train.txt', 'w').writelines(lines)
        self.training_data = open(self.directory + 'train.txt').readlines()

    def next_training_batch(self):
        images = []
        ground_truths = []

        for i in range(self.batch_size):

            if self.train_index == 0:
                self.shuffle_training_data()

            # Load image
            image_directory = self.directory + 'images/'
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
            ground_truth_directory = self.directory + 'ground_truths/'
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
            
        return images, ground_truths


    def next_val_batch(self):
        images = []
        ground_truths = []

        for i in range(self.batch_size):
            # Load image
            image_directory = self.directory + 'images/'
            image_file = random.choice(self.validation_data).rstrip()
            image = cv2.imread(image_directory + image_file)
            image = np.float32(image)
            images.append(image)

            # Load ground truth
            ground_truth_directory = self.directory + 'ground_truths/'
            ground_truth_file = image_file.replace('pic', 'seg')
            ground_truth = cv2.imread((ground_truth_directory + 
                                       ground_truth_file), cv2.IMREAD_GRAYSCALE)
            ground_truth = ground_truth/8
            ground_truths.append(ground_truth)

        return images, ground_truths


def main():
    pass

if __name__ == "__main__":
    main()
