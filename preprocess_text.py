import os
from image_captioning_utils.Descriptions import Descriptions

if __name__ == '__main__':
    """
    Load Flickr8k token file and save it into a text file
    """
    # Input block
    directory = 'Flickr8k_text/'
    filename = 'Flickr8k.token.txt'
    output_file = 'descriptions.txt'
    # End of input block

    desc = Descriptions(os.path.join(directory, filename))
    desc.save_descriptions(output_file)
    vocabulary_set = desc.get_vocabulary_set()

    print('{} images with descriptions and {} pieces of vocabulary are loaded'.format(len(desc), len(vocabulary_set)))
