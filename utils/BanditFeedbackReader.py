import numpy as np
import cv2
import os
import random
from random import randint

class BanditFeedbackReader:
    """ 
        Helper class to SegNet that handles data reading, conversion 
        and all things related to data 
    """

    def __init__(self, directory, current_step, batch_size):

        # Save variables
        self.batch_size = batch_size
        self.current_step = current_step
        self.directory = directory

        # Get meta information
        with open(self.directory + "meta", "r") as infile:
            reader = csv.reader(infile, delimiter=",")
            for row in reader:
                if row[0] == "size":
                    self.dataset_size = int(row[1])

        # We have three types of logging information:
        #   1. Image
        #   2. Segmentation
        #   3. Feedback/Loss (Delta)
        #   4. Propensities (Probability distribution at theta = 0)
        images = []
        segmentations 

    def next_item(self):
        """
            Returns next 4-tuple with (x,y,d,p) data

        """
        pass


