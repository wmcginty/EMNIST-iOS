import numpy as np
import csv
from PIL import Image

def extract_images(file, output_directory):

    with open(file, 'r') as csv_file:
        for data in csv.reader(csv_file):
            # The first column is the label
            label = data[0]

            # The rest of columns are pixels
            pixels = data[1:]

            # Make those columns into a array of 8-bits pixels
            # This array will be of 1D with length 784
            # The pixel intensity values are integers from 0 to 255
            pixels = np.array(pixels, dtype='uint8')

            # Reshape the array into 28 x 28 array (2-dimensional array)
            pixels = pixels.reshape((28, 28))
            image = Image.fromarray(pixels, mode='L')
            image.save(output_directory + '/' + label + '.jpg', 'JPEG')