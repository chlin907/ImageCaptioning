import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# Image captioning data calss
class MLData:
    def __init__(self, image_set_file, description_file, feature_file=None):
        self.image_set = self.__load_file_to_set(image_set_file)
        self.description_map = self.__load_descriptions(description_file, self.image_set)
        if feature_file is None:
            self.features = None
        else:
            self.features = self.__load_photo_features(feature_file, self.image_set)

    #def __init__(self, image_set_file, description_file):
    #    self.image_set = self.__load_file_to_set(image_set_file)
    #    self.description_map = self.__load_descriptions(description_file, self.image_set)
    #    self.features = None

    # load a pre-defined list of photo identifiers
    def __load_file_to_set(self, filename):
        """
        Load the image list file to a set
        :param filename:
        :return: dataset containg all working image id's
        """
        dataset = set()
        with open(filename, 'r') as f:
            for line in f:
                if len(line) < 1:
                    continue
                identifier = line.split('.')[0]
                dataset.add(identifier)
        return dataset

    # load clean descriptions into memory
    def __load_descriptions(self, filename, dataset):
        """
        Load the description file and convert to a description map. Only those image id's in dataset are picked
        :param filename: description file
        :param dataset: dataset of all working image id
        :return:
        """
        descriptions = dict()
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.split()
                image_id, image_desc = tokens[0], tokens[1:]
                if image_id not in dataset: # skip if image id not in dataset
                    continue
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                descriptions[image_id].append(desc)
        return descriptions

    # used as class function

    # load photo features
    def __load_photo_features(self, filename, dataset):
        """
        Load photo features from pickle file. Filter photo only in dataset
        :param filename: file path
        :param dataset: dataset collecting all working image id
        :return:
        """
        # load all features
        all_features = pickle.load(open(filename, 'rb'))
        features = {k: all_features[k] for k in dataset}
        return features

    def gen_tokenizer(self):
        """
        Genenate tokenizer based on description_map
        :return: keras tokenizer
        """
        lines = []
        [lines.extend(v) for _, v in self.description_map.items()]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    def get_descriptions(self):
        return self.description_map

    def get_max_length(self):
        """
        Find the length of the longest description sentence
        :return: int
        """
        lines = []
        [lines.extend(v) for _, v in self.description_map.items()]
        return max(len(d.split()) for d in lines)

    def gen_generator(self, tokenizer, max_length, vocab_size):
        """
        Create generator for model.fit_generator(). input can be from object or training data object
        :param tokenizer: tokenizer for text data
        :param max_length: for padding.
        :param vocab_size: for seq generation
        :return: generator
        """
        """
        Create generator for model.fit_generator()
        :param filename:
        :return: dataset containg all working image id's
        """
        descriptions = self.description_map
        photos = self.features
        while 1:
            for key, desc_list in descriptions.items():
                # retrieve the photo feature
                photo = photos[key][0]
                in_img, in_seq, out_word = self.__create_sequences(desc_list, photo, tokenizer, max_length, vocab_size)
                yield [[in_img, in_seq], out_word]

    # create sequences of images, input sequences and output words for an image
    def __create_sequences(self, desc_list, photo, tokenizer, max_length, vocab_size):
        X1, X2, y = list(), list(), list()
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)


