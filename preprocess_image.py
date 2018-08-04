import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import vgg16
from keras.models import Model

from keras.utils import plot_model

def extract_features(directory):
    """
    extract features from each photo in the directory by vgg16 model
    :param directory: path
    :return: feature dict
    """
    model = vgg16.VGG16()

    # Re-construct model: pop softmax layer
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    plot_model(model, to_file='vgg_model.png', show_shapes=True)

    # extract features from each photo. Store to (k, v) = (image_id, feature)
    features = dict()
    for name in os.listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # Pre-process image
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # first dim for sample
        image = vgg16.preprocess_input(image)
        # Pass (image_id, feature) to feature dict
        image_id = name.split('.')[0] # Remove .jpg
        feature = model.predict(image, verbose=0)
        features[image_id] = feature
        print('Image file {} is processed'.format(name))
        features
    return features

if __name__ == '__main__':
    """
    Extract feature from all images in a directory. Feaure data are dumped to a pickle file
    """
    # Input block
    #directory = 'Flicker8k_Dataset'
    directory = 'Flicker8k_Dataset_test'
    output_file = 'features_chlin.pkl'
    # End of input block

    features = extract_features(directory)
    print('Extracted features from {:d} images'.format(len(features)))

    pickle.dump(features, open('features_chlin.pkl', 'wb'))
