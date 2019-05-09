from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import random
import os
import glob

# Uses this array to iterate through all possible characters
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# use a truetype font
font = ImageFont.truetype("../font/UKNumberPlate.ttf", 68)
save_path = "../training_data/"


def main():
    print("Cleaning Folders...")
    empty_folders()
    print("Generating Data...")
    for char in characters:
        if not os.path.exists(save_path+char):
            os.makedirs(save_path+char)
        for itr in range(300):
            gen_image(char, itr)
    print("Done!")


def gen_image(character, itr):
    # Creates blank image
    img = np.zeros((50, 50, 3), np.uint8)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draws text onto image
    draw.text((random.randint(6, 9), random.randint(0, 2)), character, font=font, fill=(255, 255, 255, 0))
    # Adds rotation to text
    rotated = pil_img.rotate(random.randint(-10, 10),  expand=1)
    # Some image processing to make text look like it was extracted
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erosion = cv2.erode(np.array(rotated), kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilation = cv2.erode(np.array(erosion), kernel, iterations=1)
    # Saves image and reopens image - somehow this helps with the character looking more realistic 
    # character_image = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path+character+"/" + character+"_"+str(itr)+".jpg", dilation)
    character_image = cv2.imread(save_path+character+"/" + character+"_"+str(itr)+".jpg", 0)
    # Finds contours in generated image and extracts the character to mimic how
    # characters are extracted from the actual plate
    # _, contours, _ = cv2.findContours(character_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(character_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    x, y, w, h = cv2.boundingRect(cntsSorted[0])
    char = character_image[y:y+h, x:x+w]
    character_image = cv2.resize(char, (50, 50))
    character_image = cv2.bitwise_not(character_image)
    character_image = cv2.medianBlur(character_image, random.choice([1, 3]))
    # Saves final generated image ready for training
    cv2.imwrite(save_path+character+"/"+character+"_"+str(itr)+".jpg", character_image)


def empty_folders():
    for char in characters:
        files = glob.glob(save_path+char+"/*.jpg")
        for file in files:
            os.remove(file)


if __name__ == '__main__':
    main()
