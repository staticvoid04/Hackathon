import extractCharacters
from skimage.feature import hog
from sklearn.externals import joblib
from skimage import color


def KNN(image_path, show_steps=False):
    knn = joblib.load('../models/knn_model.pkl')
    extractedCharacters = extractCharacters.extractCharacters(image_path, show_steps=show_steps)

    def feature_extraction(image):
        # Uses hog to extract features from image/characters
        return hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(10, 10), cells_per_block=(5, 5), block_norm='L2-Hys')

    def predict(df):
        # Reshaped image into 1D
        reshaped = df.reshape(1, -1)
        # Gets prediction of character
        predict = knn.predict(reshaped)[0]
        # Gets probability of each character being predicted
        # Not being used at the moment - mainly for testing purposes
        predict_proba = knn.predict_proba(reshaped)
        return predict

    # Gets characters from plate
    characters = extractedCharacters.characters
    # HOG of each character
    hogs = list(map(lambda character: feature_extraction(character), characters))
    # Each character is run through the KNN and the predictions are stored
    predictions = list(map(lambda character: predict(character), hogs))
    predictions = ''.join(predictions)

    return predictions
