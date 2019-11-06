import json
import string
import torch
import os


def load_sequences_from_file(path, voc_path=None):
    """
    Used for reading sentences or sequences of slots.
    Splits input lines into lists of words and performs word to id transformation.
    :param path: path to txt file with data
    :param voc_path: path to json file with word to id vocabulary
    :return: torch tensor with word ids of size [num_sentences, seq_length]
    """
    table = str.maketrans('', '', string.punctuation)
    word_to_idx = {}
    dict_size = 0
    corpus = []
    with open(path, "r") as fin:
        for line in fin:
            line = line.lower()
            sentence = []
            for word in line.split():
                word = word.translate(table) # remove punctuation
                if word not in word_to_idx:
                    word_to_idx[word] = dict_size
                    dict_size += 1
                sentence.append(word_to_idx[word])
            corpus.append(sentence)

    if voc_path:
        with open(voc_path, 'w+') as fout:
            json.dump(corpus, fout)
    # TODO padding
    return torch.tensor(corpus)


def load_words_from_file(path, voc_path=None):
    """
    Used for reading sentence intent labels.
    :param path: path to txt file with labels.
    :param voc_path: path to json file with labels to id vocabulary.
    :return: torch tensor with intent label ids.
    """
    label_to_idx = {}
    dict_size = 0
    label_ids = []
    with open(path, "r") as fin:
        for label in fin:
            if label not in label_to_idx:
                label_to_idx[label] = dict_size
                dict_size += 1
            label_ids.append(label_to_idx[label])
    if voc_path:
        with open(voc_path, "w+") as fout:
            json.dump(label_to_idx, fout)
    return torch.tensor(label_ids)


def load_data(inp_dir, voc_dir=None):
    """
    Reads data from directory with data.
    Saves vocabularies to voc_dir if specified.
    :param inp_dir: directory with files <label, seq.in, seq.out>.
    :param voc_dir: output directory for vocabulary files.
    :return: one input tensor and two target vectors (slots and labels)
    """
    inp_sentences = load_sequences_from_file(os.path.join(inp_dir, "seq.in"), voc_dir)
    slots = load_sequences_from_file(os.path.join(inp_dir, "seq.out"), voc_dir)
    labels = load_words_from_file(os.path.join(inp_dir, "label"), voc_dir)
    return inp_sentences, slots, labels
