#!/usr/bin/env python3
import hydra
import numpy as np
from enum import Enum
import random
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf
import os
import torch
from datasets import Dataset, load_from_disk


PARTS_MAPPING = {
    'UNKN': 0,
    'PREF': 1,
    'ROOT': 2,
    'SUFF': 3,
    'END': 4,
    'LINK': 5,
    'HYPH': 6,
    'POSTFIX': 7,
    'B-SUFF': 8,
    'B-PREF': 9,
    'B-ROOT': 10,
    #'NUMB': 11,
}

SPEECH_PARTS = [
    'X',
    'ADJ',
    'ADV',
    'INTJ',
    'NOUN',
    'PROPN',
    'VERB',
    'ADP',
    'AUX',
    'CONJ',
    'SCONJ',
    'DET',
    'NUM',
    'PART',
    'PRON',
    'PUNCT',
    'H',
    'R',
    'Q',
    'SYM',
    'PARTICIPLE',  # aux speech parts
    'GRND',
    'ADJS',
]

SPEECH_PART_MAPPING = {str(s): num for num, s in enumerate(SPEECH_PARTS)}

MASK_VALUE = 0.0

def build_speech_part_array(sp):
    output = [0. for _ in range(len(SPEECH_PARTS))]
    output[SPEECH_PART_MAPPING[str(sp)]] = 1.
    return output

LETTERS = {
    'о': 1,
    'е': 2,
    'а': 3,
    'и': 4,
    'н': 5,
    'т': 6,
    'с': 7,
    'р': 8,
    'в': 9,
    'л': 10,
    'к': 11,
    'м': 12,
    'д': 13,
    'п': 14,
    'у': 15,
    'я': 16,
    'ы': 17,
    'ь': 18,
    'г': 19,
    'з': 20,
    'б': 21,
    'ч': 22,
    'й': 23,
    'х': 24,
    'ж': 25,
    'ш': 26,
    'ю': 27,
    'ц': 28,
    'щ': 29,
    'э': 30,
    'ф': 31,
    'ъ': 32,
    'ё': 33,
    '-': 34,
}


VOWELS = {
    'а', 'и', 'е', 'ё', 'о', 'у', 'ы', 'э', 'ю', 'я'
}


class MorphemeLabel(Enum):
    UNKN = 'UNKN'
    PREF = 'PREF'
    ROOT = 'ROOT'
    SUFF = 'SUFF'
    END = 'END'
    LINK = 'LINK'
    HYPH = 'HYPH'
    POSTFIX = 'POSTFIX'
    NONE = None


class Morpheme(object):
    def __init__(self, part_text, label, begin_pos):
        self.part_text = part_text
        self.length = len(part_text)
        self.begin_pos = begin_pos
        self.label = label
        self.end_pos = self.begin_pos + self.length

    def __len__(self):
        return self.length

    def get_labels(self):
        if self.length == 1:
            return ['S-' + self.label.value]
        result = ['B-' + self.label.value]
        result += ['M-' + self.label.value for _ in self.part_text[1:-1]]
        result += ['E-' + self.label.value]
        return result

    def get_simple_labels(self):
        if (self.label == MorphemeLabel.SUFF or self.label == MorphemeLabel.PREF or self.label == MorphemeLabel.ROOT):

            result = ['B-' + self.label.value]
            if self.length > 1:
                result += [self.label.value for _ in self.part_text[1:]]
            return result
        else:
            return [self.label.value] * self.length

    def __str__(self):
        return self.part_text + ':' + self.label.value

    @property
    def unlabeled(self):
        return not self.label.value


class Word(object):
    def __init__(self, morphemes=[], speech_part='X', is_conv=0, is_short=0, is_part=0):
        self.morphemes = morphemes
        self.sp = speech_part
        self.is_conv = is_conv
        self.is_part = is_part
        self.is_short = is_short

    def append_morpheme(self, morpheme):
        self.morphemes.append(morpheme)

    def get_word(self):
        return ''.join([morpheme.part_text for morpheme in self.morphemes])

    def parts_count(self):
        return len(self.morphemes)

    def suffix_count(self):
        return len([morpheme for morpheme in self.morphemes
                    if morpheme.label == MorphemeLabel.SUFFIX])

    def get_labels(self):
        result = []
        for morpheme in self.morphemes:
            result += morpheme.get_labels()
        return result

    def get_simple_labels(self):
        result = []
        for morpheme in self.morphemes:
            result += morpheme.get_simple_labels()
        return result

    def __str__(self):
        return '/'.join([str(morpheme) for morpheme in self.morphemes])

    def __len__(self):
        return sum(len(m) for m in self.morphemes)

    @property
    def unlabeled(self):
        return all(p.unlabeled for p in self.morphemes)


def parse_morpheme(str_repr, position):
    ##print(str_repr)
    text, label = str_repr.split(':')
    return Morpheme(text, MorphemeLabel[label], position)


def parse_word(str_repr, maxlen):
    if str_repr.count('\t') == 3:
        wordform, word_parts, _, class_info = str_repr.split('\t')
        is_conv = 0
        is_short = 0
        is_part = 0
        if 'ADJF' in class_info:
            sp = 'ADJ'
        elif 'VERB' in class_info:
            sp = 'VERB'
        elif 'NOUN' in class_info:
            sp = 'NOUN'
        elif 'ADV' in class_info:
            sp = 'ADV'
        elif 'GRND' in class_info:
            sp = 'GRND'
            is_conv = 1
        elif 'PART' in class_info:
            sp = 'PARTICIPLE'
            is_part = 1
        elif 'ADJS' in class_info:
            sp = 'ADJS'
            is_short = 1
        else:
            raise Exception("Unknown class", class_info)
    elif str_repr.count('\t') == 2:
        wordform, word_parts, sp = str_repr.split('\t')
    else:
        wordform, word_parts = str_repr.split('\t')
        sp = 'X'

    if ':' in wordform or '/' in wordform:
        return None

    if len(wordform) > maxlen:
        return None

    parts = word_parts.split('/')
    morphemes = []
    global_index = 0
    for part in parts:
        morphemes.append(parse_morpheme(part, global_index))
        global_index += len(part)
    return Word(morphemes, sp, is_conv, is_short, is_part)



def _transform_classification(parse):
    parts = []
    current_part = [parse[0]]
    for num, letter in enumerate(parse[1:]):
        index = num + 1
        if letter == 'SUFF' and parse[index - 1] == 'B-SUFF':
            current_part.append(letter)
        elif letter == 'PREF' and parse[index - 1] == 'B-PREF':
            current_part.append(letter)
        elif letter == 'ROOT' and parse[index - 1] == 'B-ROOT':
            current_part.append(letter)
        elif letter != parse[index - 1] or letter.startswith('B-'):
            parts.append(current_part)
            current_part = [letter]
        else:
            current_part.append(letter)
    if current_part:
        parts.append(current_part)

    for part in parts:
        if part[0] == 'B-PREF':
            part[0] = 'PREF'
        if part[0] == 'B-SUFF':
            part[0] = 'SUFF'
        if part[0] == 'B-ROOT':
            part[0] = 'ROOT'
        if len(part) == 1:
            part[0] = 'S-' + part[0]
        else:
            part[0] = 'B-' + part[0]
            part[-1] = 'E-' + part[-1]
            for num, letter in enumerate(part[1:-1]):
                part[num+1] = 'M-' + letter
    result = []
    for part in parts:
        result += part
    return result

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def _get_parse_repr(word):
    features = []
    word_text = word.get_word()
    for index, letter in enumerate(word_text):
        letter_features = []
        vovelty = 0
        if letter in VOWELS:
            vovelty = 1
        letter_features.append(vovelty)
        if letter in LETTERS:
            letter_code = LETTERS[letter]
        #elif letter in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        #    letter_code = 35
        else:
            letter_code = 0
        letter_features += to_categorical(letter_code, num_classes=len(LETTERS) + 1).tolist()
        letter_features += build_speech_part_array(word.sp)
        #letter_features.append(word.is_part)
        #letter_features.append(word.is_conv)
        #letter_features.append(word.is_short)
        features.append(letter_features)

    X = torch.Tensor(features).float()
    Y = torch.Tensor([PARTS_MAPPING[label] for label in word.get_simple_labels()]).byte()
    return X, Y

def _prepare_words(words, verbose=True):
    result_x, result_y = [], []
    if verbose:
        print("Preparing words")
    for i, word in enumerate(words):
        word_x, word_answer = _get_parse_repr(word)
        result_x.append(word_x)
        result_y.append(word_answer)
        if i % 1000 == 0 and verbose:
            print("Prepared", i)

    return result_x, result_y

def create_buckets(features, labels, bucket_size):
    # Combine features and labels, then sort by length of features
    data = list(zip(features, labels))
    data.sort(key=lambda x: len(x[0]))  # Sort by the length of the features

    # Create buckets
    buckets = [data[i:i + bucket_size] for i in range(0, len(data), bucket_size)]

    return buckets

# Function to load data with bucketed batching
def bucketed_data(features, labels, bucket_size):
    # Create buckets
    buckets = create_buckets(features, labels, bucket_size)

    # Shuffle buckets if needed
    random.shuffle(buckets)

    # Flatten buckets into one list
    flattened_buckets = [item for bucket in buckets for item in bucket]
    unzipped = list(zip(*flattened_buckets))
    print("Unzipped len", len(unzipped))
    print("Zero", len(unzipped[0]))
    print("One", len(unzipped[1]))
    return unzipped[0], unzipped[1]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print("Config", OmegaConf.to_yaml(cfg))
    RESTRICTED_LEN = 20
    max_len = RESTRICTED_LEN

    if not os.path.exists(cfg["outputs"]["base_dir"]):
        os.makedirs(cfg["outputs"]["base_dir"])

    def load_dataset_from_file(path, max_len):
        counter = 0
        dataset = []
        with open(path, 'r') as data:
            for num, line in enumerate(data):
                word = parse_word(line.strip(), max_len)
                if word is None:
                    continue
                counter += 1
                dataset.append(word)
                max_len = max(max_len, len(dataset[-1]))
                if counter % 5000 == 0:
                    print("Loaded", counter, "train words")
        return dataset


    def save_dataset_to_file(source_path, dest_path, max_len, bucketed):
        dataset = load_dataset_from_file(source_path, max_len)
        input_data, labels = _prepare_words(dataset)
        if bucketed:
            input_data, labels = bucketed_data(input_data, labels, cfg["training"]["batch_size"])
        torch.save({'input_data': input_data, 'labels': labels}, dest_path)


    save_dataset_to_file(cfg["training"]["train_set"], cfg["outputs"]["prepared_train_set"], RESTRICTED_LEN, True)
    save_dataset_to_file(cfg["training"]["val_set"], cfg["outputs"]["prepared_val_set"], RESTRICTED_LEN, False)
    save_dataset_to_file(cfg["training"]["test_set"], cfg["outputs"]["prepared_test_set"], RESTRICTED_LEN, False)

if __name__ == "__main__":
    main()

