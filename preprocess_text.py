import string
import os

class descriptions:
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

if __name__ == '__main__':
    """
    Load Fickr8k token file and save it into a text file
    """
    # Input block
    directory = 'Flickr8k_text/'
    filename = 'Flickr8k.token.txt'
    output_file = 'descriptions_new.txt'
    # End of input block

    desc = descriptions(os.path.join(directory, filename))
    desc.save_descriptions(output_file)
    vocabulary_set = desc.get_vocabulary_set()

    print('{} images with descriptions and {} pieces of vocabulary are loaded'.format(len(desc), len(vocabulary_set)))
