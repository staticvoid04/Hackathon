from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
import numpy as np
import cv2
import glob
import os


# This class looks for rectangular like shapes that resembles a registration plate
# and extracts it from the image
class extractPlate:
    def __init__(self, image_path, show_steps=False):
        self.image_path = image_path
        self.show_steps = show_steps
        self.empty_folder()
        self.extract()

    def empty_folder(self):
        files = glob.glob("static/Process/*.jpg")
        for file in files:
            os.remove(file)

    def extract(self):
        # Reads in image in color
        car_image_color = imread(self.image_path)
        # Reads image in gray
        car_image = imread(self.image_path, as_gray=True)
        gray_car_image = car_image * 255
        # Gets binary image
        threshold_value = threshold_otsu(gray_car_image)
        binary_car_image = gray_car_image > threshold_value
        # Labels/Finds connected regions
        label_image = measure.label(binary_car_image)
        # Shape of plate
        plate_dimensions = (0.04*label_image.shape[0],
                            0.09*label_image.shape[0],
                            0.16*label_image.shape[1],
                            0.4*label_image.shape[1])
        min_height, max_height, min_width, max_width = plate_dimensions
        # Goes through each possible plate(region) found in the image
        for region in regionprops(label_image):
            # Check if region is less than a specific size
            if region.area < 50:
                continue
            # Gets size of region
            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            region_width = max_col - min_col
            # Checks if region size is within the required plate size
            if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
                # Crops image and saves plate
                cropness = 2
                cropped_image = car_image_color[min_row+cropness:max_row -
                                                cropness, min_col+cropness:max_col-cropness]
                processed_image = self.processImage(cropped_image)
                cv2.imwrite("../temp/Plate.jpg", processed_image)
                # break

    # Shows steps of each process

    def show(self, what, image):
        scale = 0.7
        if self.show_steps:
            cv2.imshow(what, image)
            cv2.waitKey()
        cv2.imwrite("../processed/"+what+".jpg", image)
        width = int(image.shape[1]*scale)
        height = int(image.shape[0]*scale)
        image = cv2.resize(image, (width, height))
        cv2.imwrite("static/Process/"+what+".jpg", image)

    # Process the image goes through before extraction of characters

    def processImage(self, image):
        # Converts image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.show("1 - Original", image)
        # Resizes image
        image = cv2.resize(image, (480, 80))
        self.show("2 - Resized", image)
        # Otsu thresholds image
        _, thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.show("3 - Threshold", thresh)
        # Dilates image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilation = cv2.dilate(thresh, kernel, iterations=2)
        self.show("4 - Dilation", dilation)
        # Sharpens the image
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        sharpened = cv2.filter2D(dilation, -1, kernel_sharpening)
        self.show("5 - Sharpen", sharpened)
        # Inverts image
        thresh = cv2.bitwise_not(sharpened)
        self.show("6 - Binarize", thresh)
        return thresh
