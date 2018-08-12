import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from image_captioning_utils.VGGFeatureExtraction import VGGFeatureExtraction

if __name__ == '__main__':
    """
    Extract feature from all images in a directory. Feature data are dumped to a pickle file
    """
    # Input block
    directory = 'Flicker8k_Dataset'
    output_file = 'features.pkl'
    # End of input block

    features = VGGFeatureExtraction().extract_features_from_folder(directory)
    print('Extracted features from {:d} images'.format(len(features)))

    pickle.dump(features, open(output_file, 'wb'))
