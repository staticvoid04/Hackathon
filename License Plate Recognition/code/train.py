import os
import glob
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage import data, color, exposure
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.feature import hog
from imageio import imread
import scipy.ndimage
import pickle
import sys


characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main():
    if(len(sys.argv) > 1):
        if(sys.argv[1] == 'SVM'):
            print("SVM")
            SVM()
        else:
            print("KNN")
            KNN()
    else:
        print("KNN")
        KNN()


def SVM():
    def read_training_data(training_directory):
        image_data = []
        target_data = []
        for character in characters:
            files = glob.glob(training_directory+character+'/*.jpg')
            for i, file in enumerate(files):
                image = imread(file, as_gray=True)
                binary_image = image < threshold_otsu(image)
                flat_image = binary_image.reshape(-1)
                image_data.append(flat_image)
                target_data.append(character)
                if(i == 10):
                    break

        return (np.array(image_data), np.array(target_data))

    def cross_validation(model, num_of_fold, train_data, train_label):
        accuracy_result = cross_val_score(
            model, train_data, train_label, cv=num_of_fold)
        print("Cross Validation Result for ", str(num_of_fold), " -fold")
        print(accuracy_result * 100)

    print('reading data')
    training_dataset_dir = '../training_data/'
    image_data, target_data = read_training_data(training_dataset_dir)
    print('reading data completed')

    svc_model = SVC(kernel='linear', probability=True)

    cross_validation(svc_model, 4, image_data, target_data)

    print('training model')

    svc_model.fit(image_data, target_data)

    print("model trained.saving model..")
    filename = '../models/UK_model.sav'
    pickle.dump(svc_model, open(filename, 'wb'))
    print("model saved")


def KNN():
    features_list = []
    features_label = []
    print("Training...")
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
    print("Trained")
    print("Saving Model")

    # save trained model
    joblib.dump(knn, '../models/knn_model.pkl')
    print("Model Saved")


if __name__ == "__main__":
    main()
