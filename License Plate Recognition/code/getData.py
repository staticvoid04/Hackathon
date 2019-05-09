from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import random
import sys
import os
import glob
import extractPlate


def main():
    files = glob.glob('../cars/*.jpg')
    for i, file in enumerate(files):
        extractPlate.extractPlate(file)
        license_plate = cv2.imread("../temp/Plate.jpg")
        height, width, _ = license_plate.shape
        min_x, min_y = width, height
        max_x = max_y = 0
        license_plate_grey = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(
            license_plate_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cntsSorted = sorted(
            contours, key=lambda x: cv2.contourArea(x), reverse=True)
        idx = 0

        for cnt in cntsSorted:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 50:
                continue
            idx += 1
            char_image = license_plate[y:y+h, x:x+w]
            char_image = cv2.resize(char_image, (50, 50))
            char_image = cv2.bitwise_not(char_image)
            char_image_path = '../temp/Characters/' + \
                str(i) + "_" + str(idx) + '.jpg'
            cv2.imwrite(char_image_path, char_image)


if __name__ == '__main__':
    main()
