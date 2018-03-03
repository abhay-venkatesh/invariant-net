import os
import math

class RecordFileGenerator:

	def __init__(self, directory):
		# Get size of dataset
		image_directory = directory + "images/"
		files = next(os.walk(image_directory))[2]
		self.dataset_size = len(files)

		# Get directory to write to
		self.directory = directory

	def create_files(self):
		''' Create train.txt, val.txt and test.txt files. '''

		# Open files
		train_path = self.directory + 'train.txt'
		val_path = self.directory + 'val.txt'
		test_path = self.directory + 'test.txt'
		with open(train_path, 'w') as trainfile:
			with open(val_path, 'w') as valfile:
				with open(test_path, 'w') as testfile:

					# Get image counts
					numTrainingImages = math.floor(self.dataset_size * 0.7)
					numValidationImages = math.floor(self.dataset_size * 0.29)
					numTestImages = math.floor(self.dataset_size * 0.01)

					# Write the images
					for i in range(numTrainingImages):
						trainfile.write('pic' + str(i) + ".png\n")
					for i in range(numValidationImages):
						valfile.write('pic' + str(i + numTrainingImages) + 
									  ".png\n")
					for i in range(numTestImages):
						testfile.write('pic' + str(i + numTrainingImages + 
									   numValidationImages) + ".png\n")

					return numTrainingImages, numValidationImages, numTestImages

	def create_trainval_only(self):
		''' Create train.txt and val.txt'''

		# Open files
		train_path = self.directory + 'train.txt'
		val_path = self.directory + 'val.txt'
		with open(train_path, 'w') as trainfile:
			with open(val_path, 'w') as valfile:
				# Get image counts
				numTrainingImages = math.floor(self.dataset_size * 0.7)
				numValidationImages = math.floor(self.dataset_size * 0.30)

				# Write the image records
				for i in range(numTrainingImages):
					trainfile.write('pic' + str(i) + ".png\n")
				for i in range(numValidationImages):
					valfile.write('pic' + str(i + numTrainingImages) + 
								  ".png\n")

				return numTrainingImages, numValidationImages

def main():
	pass

if __name__ == "__main__":
	main()

