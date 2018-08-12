import os
import warnings
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16
from keras.models import Model
warnings.simplefilter(action='ignore', category=FutureWarning)

class VGGFeatureExtraction:
    def __init__(self):
        self.model = vgg16.VGG16()
        self.model.layers.pop()
        self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-1].output)

    def extract_features_from_folder(self, directory):
        """
        extract features from each photo in the directory by vgg16 model
        :param directory: path
        :return: feature dict
        """
        features = dict()
        for name in os.listdir(directory):
            filename = directory + '/' + name
            image_id = name.split('.')[0]  # Remove .jpg
            features[image_id] = self.extract_features_from_file(filename)
            print('Image file {} is processed'.format(name))
        return features

    def extract_features_from_file(self, filename):
        image = load_img(filename, target_size=(224, 224))
        # Pre-process image
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # first dim for sample
        image = vgg16.preprocess_input(image)

        return self.model.predict(image, verbose=0)

