#!/usr/bin/env python3

#!/usr/bin/env python3
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
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

    train_data_path = cfg["outputs"]["prepared_train_set"]
    if not os.path.exists(train_data_path):
        raise Exception("Data is not prepared")

    def load_from_cache_or_from_file(path):
        checkpoint = torch.load(path)
        return checkpoint["input_data"], checkpoint["labels"]

    train_data, train_labels = load_from_cache_or_from_file(train_data_path)
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
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    print("Training finished, saving model")
    model_path = cfg["outputs"]["model_path"]
    torch.save(model, model_path)
    print("Model saved to", model_path)


if __name__ == "__main__":
    main()
