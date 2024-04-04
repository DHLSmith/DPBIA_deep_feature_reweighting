from PIL import Image
import math
from utils import Logger
from glob import glob

# Thanks to https://www.geeksforgeeks.org/how-to-find-width-and-height-of-an-image-using-python/ and
if __name__ == '__main__':
    # filepath = "geeksforgeeks.png"
    # img = Image.open(filepath)
    # width,height = img.size
    S = 1
    R_l = 3 / 4
    R_h = 4 / 3
    W_o = 224
    H_o = 224
    W_i = math.sqrt(S * W_o * H_o * R_h)
    H_i = math.sqrt(S * W_o * H_o / R_l)

    logger = Logger('C:/deep_learning_project/data/smallfiles.txt')
    logger.write(f"Min sizes: {W_i=}, {H_i=}\n")

    for filename in glob('C:/deep_learning_project/data/waterbird_complete95_forest2water2/**/*.jpg', recursive=True):
        img = Image.open(str(filename))
        width, height = img.size
        if width <= W_i:
            logger.write(f"Narrow: {filename}\t {width} x {height}\n")
        if height <= H_i:
            logger.write(f"Short: {filename}\t {width} x {height}\n")
