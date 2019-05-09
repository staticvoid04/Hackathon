from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage import data, color, exposure
from imageio import imread
import scipy.ndimage
import os
import numpy as np
import glob
from skimage.feature import hog

features_list = []
features_label = []
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def train():
    # load labeled training / test data
    # loop over the 10 directories where each directory stores the images of a digit
    print("Starting...")
    for char in characters:
        # for digit in range(0, 10):
        label = char
        training_directory = '../training_data/' + str(label) + '/'
        # for filename in os.listdir(training_directory):
        for filename in glob.glob(training_directory + '*.jpg'):
            # if (filename.endswith('.jpg')):
            training_char = imread(filename)
            training_char = color.rgb2gray(training_char)
            df = hog(training_char, orientations=8,
                     pixels_per_cell=(10, 10), cells_per_block=(5, 5), block_norm='L2-Hys')

            features_list.append(df)
            features_label.append(label)

    # store features array into a numpy array
    features = np.array(features_list, 'float64')
    # split the labled dataset into training / test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, features_label)
    # train using K-NN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    # get the model accuracy
    model_score = knn.score(X_test, y_test)

    # save trained model
    joblib.dump(knn, '../models/knn_model.pkl')
    print("Trained")


if __name__ == "__main__":
    train()
