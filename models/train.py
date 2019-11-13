import pandas as pd
import numpy as np

from models.dataset import ATIS, preprocess_atis, padding_map
from models.bilstm_model import BiLSTMmodel

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

SLOTS = 129
INTENTS = 26
N_EPOCHS = 100
DECAY_RATE = 0.05


def main(data_path='data/raw_data/ms-cntk-atis/'):
    # 0) Инициализация девайса и различных параметров
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    params = {'batch_size': 128,
              'shuffle': True}

    print(f'DEVICE: {device}')

    # 1) Загрузить данные
    ## 1. Разобраться с тем, как устроенны данные - текст разбит на числовые токены, лэйблы занумерованы
    ## 2. Если все не ок, написать выгрузку, иначе сразу 3.
    ## 3. прописать выгрузку здесь
    train_seq = pd.read_csv(data_path+'atis.train.query.csv', header=None)
    slot_train = pd.read_csv(data_path+'atis.train.slots.csv', header=None)
    intent_train = pd.read_csv(data_path+'atis.train.intent.csv', header=None)

    test_seq = pd.read_csv(data_path + 'atis.test.query.csv', header=None)
    slot_test= pd.read_csv(data_path + 'atis.test.slots.csv', header=None)
    intent_test = pd.read_csv(data_path + 'atis.test.intent.csv', header=None)

    vocab = pd.read_csv(data_path+'atis.dict.vocab.csv', header=None)
    vocab_size = len(vocab)


    # 2) Препроцесс
    ## 1. По сути надо все переписать в двумерные массивы
    ## 2. One-Hot Encode меток

    X_train, y_slot_train, y_intent_train = preprocess_atis(train_seq, intent_train, slot_train, INTENTS, SLOTS)
    X_test, y_slot_test, y_intent_test = preprocess_atis(test_seq, intent_test, slot_test, INTENTS, SLOTS)


    ## 3. Паддинг - так как уже есть разбитый вакубуляр, можем добить один токен, который отвеает за паддинг
    X_train = pad_sequence(X_train, batch_first=True,
                           padding_value=vocab_size).to(torch.long)  # size: tensor(batch, max_seq_len)
    y_slot_train = pad_sequence(y_slot_train, batch_first=True,
                                padding_value=-100).to(torch.long)  # size: tensor(batch, max_seq_len, SLOTS)
    X_test = pad_sequence(X_test, batch_first=True,
                          padding_value=vocab_size).to(torch.long)  # size: tensor(batch, max_seq_len)
    y_slot_test = pad_sequence(y_slot_test, batch_first=True,
                               padding_value=-100).to(torch.long)  # size: tensor(batch, max_seq_len, SLOTS)
    vocab_size += 1


    # 3) Даталоадер
    training = data.DataLoader(ATIS(X_train, y_slot_train, y_intent_train), **params)
    testing = data.DataLoader(ATIS(X_test, y_slot_test, y_intent_test), **params)

    # 4) Инициализация модели, оптимайзера
    model = BiLSTMmodel(vocab_size, embedding_dim=100, n_slots=SLOTS,
                        n_intents=INTENTS, hidden_dim=64).to(device)
    optimizer = Adam(params=model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: step / (1 + DECAY_RATE*step))
    cr_slot = nn.CrossEntropyLoss(ignore_index=-100)
    cr_intent = nn.CrossEntropyLoss()

    # 5) Обучение и валидация
    for X, slot, intent in training:
        break
    for ep in range(N_EPOCHS):
        print(f'epoch: {ep}')

        model.train()
        losses = []
        #for X, slot, intent in training:
        for i in range(32):
            lengths = padding_map(X, padding_value=vocab_size - 1)

            optimizer.zero_grad()
            X, slot, intent = X.to(device), slot.to(device), intent.to(device)
            slot_pred, intent_pred = model(X, lengths)  # (batch, seq_len, n_slots), (batch, n_intents)
            loss = cr_slot(slot_pred, slot) + cr_intent(intent_pred, intent)
            loss.backward()
            losses.append(float(loss.cpu()))
            optimizer.step()
            scheduler.step()

        train_loss = np.mean(losses)

        with torch.no_grad():
            model.eval()
            losses = []
            for X, slot, intent in testing:
                lengths = padding_map(X, padding_value=vocab_size - 1)
                X, slot, intent = X.to(device), slot.to(device), intent.to(device)
                slot_pred, intent_pred = model(X, lengths)
                loss = cr_slot(slot_pred, slot) + cr_intent(intent_pred, intent)
                losses.append(float(loss.cpu()))
        val_loss = np.mean(losses)
        print(f'train_loss: {train_loss}, val_loss: {val_loss}')


if __name__ == '__main__':
    main(data_path, emb_path)
