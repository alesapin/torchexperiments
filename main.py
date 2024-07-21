#!/usr/bin/env python3
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from argparse import ArgumentParser
from torch.nn.utils.rnn import pad_sequence
from enum import Enum
import hydra
from omegaconf import DictConfig, OmegaConf
import tqdm

if not torch.cuda.is_available():
    raise Exception("Cuda not found")
else:
    torch.set_default_device('cuda')

torch.set_printoptions(profile="full")

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

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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


def measure_quality(predicted_targets, targets, words, verbose=False):
    TP, FP, FN, equal, total = 0, 0, 0, 0, 0
    SE = ['{}-{}'.format(x, y) for x in "SE" for y in ["ROOT", "PREF", "SUFF", "END", "LINK", "None"]]
    corr_words = 0
    for corr, pred, word in zip(targets, predicted_targets, words):
        corr_len = len(corr)
        pred_len = len(pred)
        boundaries = [i for i in range(corr_len) if corr[i] in SE]
        pred_boundaries = [i for i in range(pred_len) if pred[i] in SE]
        common = [x for x in boundaries if x in pred_boundaries]
        TP += len(common)
        FN += len(boundaries) - len(common)
        FP += len(pred_boundaries) - len(common)
        equal += sum(int(x == y) for x, y in zip(corr, pred))
        total += len(corr)
        corr_words += (corr == pred)
        if corr != pred and verbose:
            print("Error in word '{}':\n correct:".format(word.get_word()), corr, '\n!=\n wrong:', pred)

    metrics = ["Precision", "Recall", "F1", "Accuracy", "Word accuracy"]
    results = [TP / (TP+FP), TP / (TP+FN), TP / (TP + 0.5*(FP+FN)),
               equal / total, corr_words / len(targets)]
    return list(zip(metrics, results))

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

    X = torch.Tensor(features)
    Y = torch.Tensor([PARTS_MAPPING[label] for label in word.get_simple_labels()]).long()
    return X, Y

def _pad_sequences(Xs, Ys, max_len):
    newXs = pad_sequence(Xs, batch_first=True, padding_value=MASK_VALUE)
    newYs = pad_sequence(Ys, batch_first=True, padding_value=MASK_VALUE)
    return newXs, newYs

def _prepare_words(words, max_len, verbose=True):
    result_x, result_y = [], []
    if verbose:
        print("Preparing words")
    for i, word in enumerate(words):
        word_x, word_answer = _get_parse_repr(word)
        result_x.append(word_x)
        result_y.append(word_answer)
        if i % 1000 == 0 and verbose:
            print("Prepared", i)

    return _pad_sequences(result_x, result_y, max_len)

class Conv1MorphModel(nn.Module):
    def __init__(self, dropouts, conv_layers, kernel_sizes):
        super(Conv1MorphModel, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.activation = 'relu'
        self.parts_mapping_len = len(PARTS_MAPPING)

        input_channels = len(LETTERS) + 1 + 1 + len(SPEECH_PARTS)
        print("Input channels", input_channels)

        for i, (drop, units, window_size) in enumerate(zip(dropouts, conv_layers, kernel_sizes)):
            print("i", i, drop, units, window_size)
            self.conv_layers.append(nn.Conv1d(in_channels=input_channels if i == 0 else conv_layers[i-1], out_channels=units, kernel_size=window_size, padding='same'))
            self.activation_layers.append(nn.ReLU() if self.activation == 'relu' else nn.Tanh())
            self.dropout_layers.append(nn.Dropout(drop))

        self.dense_layer = nn.Linear(conv_layers[-1], self.parts_mapping_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch, channels, seq_length) for Conv1d
        for conv, activation, drop in zip(self.conv_layers, self.activation_layers, self.dropout_layers):
            x = conv(x)
            x = activation(x)
            x = drop(x)

        x = x.permute(0, 2, 1)  # Change to (batch, time, features)
        x = self.dense_layer(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.encoding[:, :seq_len, :].to(x.device)
        return x

class TransformerMorphModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation='relu'):
        super(TransformerMorphModel, self).__init__()

        self.parts_mapping_len = len(PARTS_MAPPING)

        input_channels = len(LETTERS) + 1 + 1 + len(SPEECH_PARTS)
        self.embedding = nn.Linear(input_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 20)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.dense_layer = nn.Linear(d_model, self.parts_mapping_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Change to (seq_length, batch_size, d_model) for Transformer
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Change back to (batch_size, seq_length, d_model)
        x = self.dropout(x)
        x = self.dense_layer(x)

        return x

def eval_model(model, loader, data):
    model.eval()

    def one_iteration(inputs):
        outputs = model(inputs.cuda())
        return outputs

    # Disable gradient computation and reduce memory consumption.
    predicted_validation = []
    reverse_mapping = {v: k for k, v in PARTS_MAPPING.items()}
    with torch.no_grad():
        collected_outputs = []
        for inputs, labels in loader:
            outputs = one_iteration(inputs)
            argmaxed = torch.argmax(outputs, dim=-1)
            collected_outputs.append(argmaxed)

        i = 0
        for tensor in collected_outputs:
            for row in tensor:
                word = data[i]
                cutted_prediction = row[:len(word)]
                raw_parse = [reverse_mapping[int(num)] for num in cutted_prediction]
                parse = _transform_classification(raw_parse)
                predicted_validation.append(parse)
                i += 1

    if i != len(data):
        raise Exception("Not all words validated {} / {}", i, len(data))

    print(measure_quality(predicted_validation, [w.get_labels() for w in data], data, False))

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    def one_iteration(inputs, labels):
        outputs = model(inputs.cuda())
        return outputs, criterion(outputs.view(-1, outputs.shape[-1]).cuda(), labels.view(-1).cuda())

    for epoch in tqdm.tqdm(range(num_epochs)):
        running_loss = 0.0
        print (f"Started epoch #{epoch + 1}")
        counter = 0
        model.train(True)
        for inputs, labels in tqdm.tqdm(train_loader, leave=False):
            counter += 1
            if counter % 1000 == 0:
                print("Counter", counter)

            optimizer.zero_grad()
            _, loss = one_iteration(inputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}')

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print("Config", OmegaConf.to_yaml(cfg))
    RESTRICTED_LEN = 20
    max_len = RESTRICTED_LEN

    if not os.path.exists(cfg["cachedir"]):
        os.makedirs(cfg["cachedir"])

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


    def load_from_cache_or_from_file(path, name, max_len):
        cache_path = os.path.join(cfg["cachedir"], name)
        if os.path.exists(cache_path):
            checkpoint = torch.load(cache_path)
            return checkpoint["input_data"], checkpoint["labels"]
        else:
            dataset = load_dataset_from_file(path, max_len)
            input_data, labels = _prepare_words(dataset, max_len)
            torch.save({'input_data': input_data, 'labels': labels}, cache_path)
            return input_data, labels


    train_data, train_labels = load_from_cache_or_from_file(cfg["training"]["train_set"], "train_set", RESTRICTED_LEN)
    val_data, val_labels = load_from_cache_or_from_file(cfg["training"]["val_set"], "val_set", RESTRICTED_LEN)
    test_data, test_labels = load_from_cache_or_from_file(cfg["training"]["test_set"], "test_set", RESTRICTED_LEN)

    print("Maxlen", max_len)

    if cfg["model"] == "cnn":
        convolutions = cfg["cnn"]["conv_layers"]
        dropouts = cfg["cnn"]["dropouts"]
        windows = cfg["cnn"]["windows"]
        model = Conv1MorphModel(dropouts, convolutions, windows)
    elif cfg["model"] == "transformer":
        embedding_size = cfg["transformer"]["embedding_size"]
        num_heads = cfg["transformer"]["num_heads"]
        num_encoder_layers = cfg["transformer"]["num_encoder_layers"]
        dim_feedforward = cfg["transformer"]["dim_feedforward"]
        dropout = cfg["transformer"]["dropout"]
        model = TransformerMorphModel(embedding_size, num_heads, num_encoder_layers, dim_feedforward, dropout)
    else:
        raise Exception(f"Unknown model {cfg['model']}")

    batch_size = cfg["training"]["batch_size"]
    num_epochs = cfg["training"]["num_epochs"]
    learning_rate = cfg["training"]["learning_rate"]
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def get_loader(data, labels, shuffle):
        dataset = torch.utils.data.TensorDataset(data, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator(device='cuda'),)

    train_loader = get_loader(train_data, train_labels, True)

    validation_dataset = load_dataset_from_file(cfg["training"]["val_set"], RESTRICTED_LEN)
    val_loader = get_loader(val_data, val_labels, False)
    test_dataset = load_dataset_from_file(cfg["training"]["test_set"], RESTRICTED_LEN)
    test_loader = get_loader(test_data, test_labels, False)

    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    eval_model(model, val_loader, validation_dataset)
    eval_model(model, test_loader, test_dataset)


if __name__ == "__main__":
    main()

