import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.utils import plot_model
from keras.models import Model
from keras import layers

from keras.layers.merge import add

# Image captioning model calss
class MLModel:
    def __init__(self, vocab_size, max_length, tokenizer):
        self.model = self._define_model(vocab_size, max_length, tokenizer)

    # define the captioning model
    def _define_model(self, vocab_size, max_length, tokenizer):
        # image feature extractor model
        image_input = layers.Input(shape=(4096,)) # 4069 is the output dim of the last VGG16 dense layer
        image_1 = layers.Dropout(0.5)(image_input)
        image_2 = layers.Dense(256, activation='relu')(image_1)
        # language sequence model
        embedding_matrix, embedding_dim = self._gen_embedding(tokenizer, vocab_size)
        language_input = layers.Input(shape=(max_length,))
        language_1 = layers.Embedding(vocab_size, 256, mask_zero=True)(language_input)
        language_2 = layers.Dropout(0.5)(language_1)
        language_3 = layers.LSTM(256)(language_2)
        # decoder model
        decoder1 = add([image_2, language_3])
        decoder2 = layers.Dense(25, activation='relu')(decoder1)
        output = layers.Dense(vocab_size, activation='softmax')(decoder2)

        model = Model(inputs=[image_input, language_input], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def _gen_embedding(self, tokenizer, vocab_size):
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
