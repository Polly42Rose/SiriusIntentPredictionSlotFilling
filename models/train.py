import pandas as pd
import numpy as np

from models.dataset import ATIS
# from models.bilstm_model import

import torch
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
    # Эту часть надо обернуть в функцию, так как для теста то же самое !!!!!!!
    X_train = []
    y_slot_train = []
    y_intent_train = []
    for i in range(len(train_seq)):
        X_train.append(torch.tesnor(train_seq[0].iloc[i].split()))

        intent = [0] * INTENTS
        intent[intent_train[i]] = 1
        y_intent_train.append(intent)  # (batch, INTENTS)

        slot = []  # двумерная матрица (SLOTS, seq_len)
        for j in range(len(slot_train[0].iloc[i].split())):
            cur_slot = [0] * SLOTS
            cur_slot[slot_train[0].iloc[i].split()[j]] = 1
            slot.append(torch.tensor(cur_slot))
        # как здесь стоит паддить - нулями, так как все равно использум кросс энтропию,
        # а значит - эти нули не будут никак задействованы
        y_slot_train.append(slot)  # (batch, seq_len, SLOTS)
    y_intent_train = torch.tensor(y_intent_train)

    X_train, y_slot_train, y_intent_train = ...()
    X_test, y_slot_test, y_intent_test = ...()


    ## 3. Паддинг - так как уже есть разбитый вакубуляр, можем добить один токен, который отвеает за паддинг
    X_train = pad_sequence(X_train, batch_first=True,
                           padding_value=vocab_size).to(torch.long)  # size: tensor(batch, max_seq_len)
    y_slot_train = pad_sequence(y_slot_train, batch_first=True,
                           padding_value=0).to(torch.long)  # size: tensor(batch, max_seq_len, SLOTS)
    X_test = pad_sequence(X_test, batch_first=True,
                           padding_value=vocab_size).to(torch.long)  # size: tensor(batch, max_seq_len)
    y_slot_test = pad_sequence(y_slot_test, batch_first=True,
                                padding_value=0).to(torch.long)  # size: tensor(batch, max_seq_len, SLOTS)
    vocab_size += 1


    ## 4. Карта паддинга, для механизма внимания
    ### ????

    # 3) Даталоадер
    training = data.DataLoader(ATIS(X_train, y_slot_train, y_intent_train), **params)
    testing = data.DataLoader(ATIS(X_test, y_slot_test, y_intent_test), **params)

    # 4) Инициализация модели, оптимайзера
    model = MODEL().to(device)
    optimizer = Adam(lr=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: step / (1 + DECAY_RATE*step))


    # 5) Обучение и валидация
    for ep in range(N_EPOCHS):

        model.train()
        losses = []
        for X, slot, intent in training:
            optimizer.zero_grad()
            X, slot, intent = X.to(device), slot.to(device), intent.to(device)
            output = model(X)
            loss = cr_slot(output, slot) + cr_intent(output, intent)
            loss.backward()
            losses.append(float(loss.cpu()))
            optimizer.step()
            scheduler.step()

        train_loss = np.mean(losses)

        with torch.no_grad():
            model.eval()
            losses = []
            for X, slot, intent in training:
                X, slot, intent = X.to(device), slot.to(device), intent.to(device)
                output = model(X)
                loss = cr_slot(output, slot) + cr_intent(output, intent)
                losses.append(float(loss.cpu()))
        val_loss = np.mean(losses)


if __name__ == '__main__':
    main(data_path, emb_path)