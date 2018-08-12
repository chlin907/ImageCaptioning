import string
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras import layers

from keras.layers.merge import add

class Descriptions:
    def __init__(self, file_path):
        self.description_map = self._load_descriptions(file_path)
        self._clean_descriptions()

    def __len__(self):
        return len(self.description_map)

    def _load_descriptions(self, file_path):
        """
        Prepare a hash map with image_id: image_desc
        :param file_path: file path to load raw description
        :return: (k, v) = (image id , list of description)
        """
        mapping = dict()

        with open(file_path, 'r') as f:
            for line in f:
                if len(line) < 2:
                    continue

                # Prepare image_id and to image_desc map
                tokens = line.split()
                image_id, image_desc = tokens[0], tokens[1:]
                image_id, image_desc = image_id.split('.')[0], ' '.join(image_desc)

                if image_id not in mapping:
                    mapping[image_id] = list()

                mapping[image_id].append(image_desc)
        return mapping

    def _clean_descriptions(self):
        """
        clean up the value (desc text) of self.decription_map.
        """
        #import enchant  # for spelling correction
        #english_dictionary = enchant.Dict("en_US")

        for key, line_list in self.description_map.items():
            for i, line in enumerate(line_list):
                line = line.translate(str.maketrans('', '', string.punctuation + string.digits)) # remove punctuation and digits

                line = line.split()
                line = [word for word in line if len(word)>1] # Remove a and hanging s (from like mom's)

                # Spelling correction
                #for iword in range(len(line)):
                #    if not english_dictionary.check(line[iword]) and english_dictionary.suggest(line[iword]):
                #        line[iword] = english_dictionary.suggest(line[iword])[0]

                line = [word.lower() for word in line]  # convert to lower case
                line_list[i] = ' '.join(line)

    def get_vocabulary_set(self):
        """
        Convert self.description_map to a vocabulary set
        :return: list of all vocabulary
        """
        # build a set to contain all description strings
        all_desc = set()
        for key in self.description_map.keys():
            [all_desc.update(d.split()) for d in self.description_map[key]]

        return all_desc

    def save_descriptions(self, filename):
        """
        Save self.description_map to a text file
        :param filename: text file name to save
        """
        lines = list()
        for key, desc_list in self.description_map.items():
            for desc in desc_list:
                lines.append(key + ' ' + desc)
        data = '\n'.join(lines)
        with open(filename, 'w') as f:
            f.write(data)




# Image captioning data calss
class MLData:
    def __init__(self, image_set_file, description_file, feature_file):
        self.image_set = self.__load_file_to_set(image_set_file)
        self.description_map = self.__load_descriptions(description_file, self.image_set)
        self.features = self.__load_photo_features(feature_file, self.image_set)

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

    def get_max_length(self):
        """
        Find the length of the longest description sentence
        :return: int
        """
        lines = []
        [lines.extend(v) for _, v in self.description_map.items()]
        return max(len(d.split()) for d in lines)

    def gen_generator(self, tokenizer, max_length):
        """
        Create generator for model.fit_generator(). input can be from object or training data object
        :param tokenizer: tokenizer for text data
        :param max_length: for padding.
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
                in_img, in_seq, out_word = self.__create_sequences(desc_list, photo, tokenizer, max_length)
                yield [[in_img, in_seq], out_word]

    # create sequences of images, input sequences and output words for an image
    def __create_sequences(self, desc_list, photo, tokenizer, max_length):
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



# Image captioning model calss
class MLModel:
    def __init__(self, vocab_size, max_length, tokenizer):
        self.model = self._define_model(vocab_size, max_length, tokenizer)

    # define the captioning model
    def _define_model(self, vocab_size, max_length, tokenizer):

        """
        # original model structure
        # image feature extractor model
        image_input = layers.Input(shape=(4096,)) # 4069 is the output dim of the last VGG16 dense layer
        image_1 = layers.Dropout(0.5)(image_input)
        image_2 = layers.Dense(256, activation='relu')(image_1)
        # language sequence model
        language_input = layers.Input(shape=(max_length,))
        language_1 = layers.Embedding(vocab_size, 256, mask_zero=True)(language_input)
        language_2 = layers.Dropout(0.5)(language_1)
        language_3 = layers.LSTM(256)(language_2)
        """
        # image feature extractor model
        image_input = layers.Input(shape=(4096,)) # 4069 is the output dim of the last VGG16 dense layer
        image_1 = layers.Dense(128, activation='relu')(image_input)
        image_2 = layers.Dropout(0.4)(image_1)
        # language sequence model
        embedding_matrix, embedding_dim = self._gen_embedding(tokenizer)
        language_input = layers.Input(shape=(max_length,))
        language_1 = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, weights=[embedding_matrix],
                                      trainable=True)(language_input)
        language_2 = layers.LSTM(128, activation='relu', dropout=0.4, recurrent_dropout=0.2)(language_1)

        #language_1 = layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
        #                              trainable=True)(language_input)

        #language_2 = layers.LSTM(256, activation='relu', dropout=0.1,
        #                         recurrent_dropout=0.5, return_sequences=True)(language_1)


        # decoder model
        decoder1 = add([image_2, language_2])
        decoder2 = layers.Dense(25, activation='relu')(decoder1)
        output = layers.Dense(vocab_size, activation='softmax')(decoder2)

        model = Model(inputs=[image_input, language_input], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def _gen_embedding(self, tokenizer):
        """
        Generate the embedding_matrix and embedding_dim from a given pretrained GLOVE embedding model
        :param tokenizer: tokenizer from the training data
        :return: (embedding_matrix, embedding_dim)
        """
        # Pretrained embedding
        pretrained_embedding_model_path = './glove.6B.100d.txt'
        embedding_dim = 100 # fixed by glove.6B. This number can be obtained from the vector size in file

        embeddings_index = {}
        with open(pretrained_embedding_model_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print('Found {} word vectors from model in {}.'.format(len(embeddings_index), pretrained_embedding_model_path ))

        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, i in tokenizer.word_index.items():
            if i < vocab_size:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix, embedding_dim

    def get_model(self):
        return self.model
