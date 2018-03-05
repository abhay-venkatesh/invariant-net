from random import randint
import numpy as np
import cv2

class ValDatasetReader:
  ''' Helper class to SegNet that handles data reading, conversion 
      and all things related to data. Non-Batch Version. '''

  def __init__(self, WIDTH, HEIGHT, dataset_directory):
    self.dataset_directory = dataset_directory
    val_data_file = dataset_directory + 'val.txt'
    self.val_data = open(val_data_file).readlines()
    self.val_data_size = len(self.val_data)
    self.val_index = 0
    self.WIDTH = WIDTH
    self.HEIGHT = HEIGHT

  def next_val_pair(self):
    # Load image
    image_directory = self.dataset_directory + 'images/'
    image_file = self.val_data[self.val_index].rstrip()
    self.val_index += 1
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
    image = np.float32(image)
    random_number = randint(0,1)
    if random_number == 1:
        image = np.fliplr(image)

    # Load ground truth
    ground_truth_directory = self.dataset_directory + 'ground_truths/'
    ground_truth_file = image_file.replace('pic', 'seg')
    ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
    ground_truth= cv2.resize(ground_truth, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
    ground_truth = ground_truth/8
    if random_number == 1:
        ground_truth = np.fliplr(ground_truth)

    return image, ground_truth, image_file

def main():
    pass
if __name__ == "__main__":
    main()
