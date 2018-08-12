import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from keras import callbacks

from image_captioning_utils.MLModel import MLModel
from image_captioning_utils.MLData import MLData

# load training dataset (6K)
if __name__ == '__main__':
    # Input block
    train_image_set_file = 'Flickr8k_text/Flickr_8k.trainImages.txt'
    val_image_set_file = 'Flickr8k_text/Flickr_8k.devImages.txt'

    #train_image_set_file = 'Flickr8k_text/Flickr_8k.trainImages.txt_reduced'
    #val_image_set_file = 'Flickr8k_text/Flickr_8k.devImages.txt_reduced'
    description_file = 'descriptions.txt'
    feature_file = 'features.pkl'
    num_epoch = 10
    run_save_tokenizer = True
    run_val = True
    # End of input block

    train_data = MLData(train_image_set_file, description_file, feature_file=feature_file)
    train_image_set = train_data.image_set
    train_descriptions = train_data.description_map
    train_features = train_data.features

    # prepare tokenizer
    tokenizer = train_data.gen_tokenizer()
    if run_save_tokenizer:
        pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

    vocab_size = len(tokenizer.word_index) + 1
    max_length = train_data.get_max_length()

    print('Train data set size =               {}'.format(len(train_image_set)))
    print('Train descriptions =                {}'.format(len(train_descriptions)))
    print('Train photos =                      {}'.format(len(train_features)))
    print('Train num of different vocabulary = {}'.format(vocab_size))
    print('Train max description length =      {}'.format(max_length))

    if run_val:
        val_data = MLData(val_image_set_file, description_file, feature_file)
        val_image_set = val_data.image_set
        val_descriptions = val_data.description_map
        val_features = val_data.features
        print('Val data set size =               {}'.format(len(val_image_set)))
        print('Val descriptions =                {}'.format(len(val_descriptions)))
        print('Val photos =                      {}'.format(len(val_features)))
        print('Val num of different vocabulary = {}'.format(vocab_size))
        print('Val max description length =      {}'.format(max_length))

    # define the model
    model = MLModel(vocab_size, max_length, tokenizer)
    model = model.get_model()

    log_dir = './log_dir'
    model_save_path = './model'
    callback_log = callbacks.TensorBoard(log_dir = './log_dir')
    callback_chkpoint = callbacks.ModelCheckpoint('model_weight.{epoch:02d}.h5',
                                                 verbose=0, save_best_only=False,
                                                 save_weights_only=False, mode='auto', period=1)
    callback_list = [callback_chkpoint, callback_log]

    train_steps = len(train_descriptions)
    train_generator = train_data.gen_generator(tokenizer, max_length, vocab_size)
    if not run_val:
        history = model.fit_generator(train_generator, epochs=num_epoch, steps_per_epoch=train_steps, verbose=1, callbacks=callback_list)
        train_loss = history.history['loss']
    else:
        val_steps = len(val_descriptions)
        val_generator = val_data.gen_generator(tokenizer, max_length, vocab_size)
        history = model.fit_generator(train_generator, epochs=num_epoch,
                                      validation_data=val_generator, validation_steps=val_steps,
                                      steps_per_epoch=train_steps, verbose=1, callbacks=callback_list)
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']


    epochs = range(1, len(train_loss) + 1)
    import matplotlib.pyplot as plt

    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    if run_val:
        plt.plot(epochs, val_loss, 'ro', label='Validation loss')
    plt.show()

