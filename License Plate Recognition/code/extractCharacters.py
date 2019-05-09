from imageio import imread
import extractPlate
import cv2
import glob
import os

# This class calls the extractPlate class which extracts the plate from the image
# with that plate this class extracts the characters from the plate


class extractCharacters:
    def __init__(self, image_path, save_path="../contours/", clear_folder=True, show_steps=False):
        self.image_path = image_path
        self.save_path = save_path
        self.clear_folder = clear_folder
        self.show_steps = show_steps
        self.characters = []
        self.char_images = []
        self.char_start = []
        self.extract()

    def empty_folder(self):
        files = glob.glob("../contours/*.jpg")
        for file in files:
            os.remove(file)

    def extract(self):
        if self.clear_folder:
            self.empty_folder()
        # Extracts plate out of image
        extractPlate.extractPlate(self.image_path, show_steps=self.show_steps)
        # Reads in cropped out plate
        license_plate = cv2.imread("../temp/Plate.jpg")
        height, width, _ = license_plate.shape
        min_x, min_y = width, height
        max_x = max_y = 0
        # Converts to grayscale
        license_plate_grey = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        # Gets contours (outlines of objects) in the image
        contours, _ = cv2.findContours(license_plate_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #_, contours, _ = cv2.findContours(
            #license_plate_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Sorts contours by area
        cntsSorted = sorted(
            contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)
        idx = 0
        # Copys plate
        draw_on_plate = license_plate.copy()

        # Iterate through contours from highest contour area
        for cnt in cntsSorted:
            # Gets dimentions of contour
            x, y, w, h = cv2.boundingRect(cnt)
            # Checks height of contour
            if h < 50 and w < 50:
                continue
            idx += 1
            # Crops out character from plate
            char_image = license_plate[y:y+h, x:x+w]
            # Reszies cropped image
            char_image = cv2.resize(char_image, (50, 50))
            # Inverts image
            char_image = cv2.bitwise_not(char_image)
            # Gets image path
            char_image_path = self.save_path + str(idx) + '.jpg'
            # Saves image
            cv2.imwrite(char_image_path, char_image)
            # Draws rectangle around character
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
            cv2.rectangle(draw_on_plate, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Stores character image path
            self.char_images.append(char_image_path)
            # Stores x component of character
            # This helps decided which character is at the begining of the plate
            self.char_start.append(x)

            # Shows plate
            if self.show_steps:
                cv2.imshow('ImageWindow', draw_on_plate)
                cv2.waitKey()
            cv2.imwrite("static/Process/identified_characters.jpg",
                        draw_on_plate)

        # Saves plate
        cv2.imwrite("../processed/Plate_Contours.jpg", license_plate)
        # Sorts characters left to right
        sorted_images = [i for _, i in sorted(
            zip(self.char_start, self.char_images), key=lambda pair: pair[0])]

        # Character images are saved in list
        for image in sorted_images:
            self.characters.append(imread(image))
