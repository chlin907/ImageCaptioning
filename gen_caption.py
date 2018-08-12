import pickle
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model

from image_captioning_utils.MLData import MLData
from image_captioning_utils.VGGFeatureExtraction import VGGFeatureExtraction

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        # Encoding and padding
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word and convert probability to integer index
        yhat = argmax(model.predict([photo, sequence], verbose=0))
        # map integer to word
        word = next((key for key, value in tokenizer.word_index.items() if value == yhat), None)

        if word is None:   # Stop when we cannot find the word
            break

        in_text += ' ' + word

        if word == 'endseq':  # stop if we hit the end
            break
    return ' '.join(in_text.split()[1:-1])


if __name__ == '__main__':
    # Beginning of input block
    train_image_set_file = './Flickr8k_text/Flickr_8k.trainImages.txt'
    description_file = './descriptions.txt'
    tokenizer_file = './tokenizer.pkl'
    model_filename = './model.h5'

    # Provide your test photo in the same directory
    photo_list = ['img_dog_in_water.jpg', 'img_dog_standing.jpg',
                  'img_people_basketball.jpg', 'img_two_dogs_in_snow.jpg',
                  'img_skier.jpg', 'img_two_dog_on_grass.jpg', 
                  'img_boy_on_swing.jpg', 'img_dog_jump_on_water.jpg',
                  'img_boy_on_field.jpg']
    # User can provide max_length or None. If none, the max length is determined from descriptions file
    max_length = None
    # End of input block

    if max_length is None:
        ml_data = MLData(train_image_set_file, description_file)
        train_descriptions = ml_data.get_descriptions()
        max_length = ml_data.get_max_length()

    print('max_length from {} = {}'.format(description_file, max_length))

    tokenizer = pickle.load(open(tokenizer_file, 'rb'))
    model = load_model(model_filename)
    for photo_filename in photo_list:
        photo_feature = VGGFeatureExtraction().extract_features_from_file(photo_filename)
        caption = generate_desc(model, tokenizer, photo_feature, max_length)
        print('Photo "{}" can be described by "{}"'.format(photo_filename, caption))

