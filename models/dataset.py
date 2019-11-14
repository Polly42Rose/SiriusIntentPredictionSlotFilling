from torch.utils import data
import torch
import numpy as np


class ATIS(data.Dataset):
    def __init__(self, X, slots, intents):
        self.X = X
        self.slots = slots
        self.intents = intents
        self.size = X.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.slots[idx], self.intents[idx]


def preprocess_atis(raw_seq, raw_intent, raw_slot, n_intents, n_slots):
    X = []
    y_slot = []
    y_intent = []
    for i in range(len(raw_seq)):
        X.append(torch.tensor([int(token) for token in raw_seq[0].iloc[i].split()]))

        # intent = [0] * n_intents
        # intent[raw_intent[i]] = 1
        # y_intent.append(intent)  # (batch, INTENTS)
        y_intent.append(raw_intent[0].iloc[i])  # (batch, 1)

        slot = []  # двумерная матрица (SLOTS, seq_len)
        # for j in range(len(raw_slot[0].iloc[i].split())):
        #     cur_slot = [0] * n_slots
        #     cur_slot[raw_slot[0].iloc[i].split()[j]] = 1
        #     slot.append(torch.tensor(cur_slot))
        slot = torch.tensor([int(token) for token in raw_slot[0].iloc[i].split()])

        # как здесь стоит паддить - нулями, так как все равно использум кросс энтропию,
        # а значит - эти нули не будут никак задействованы
        y_slot.append(slot)  # (batch, seq_len, SLOTS)
    y_intent = torch.tensor(y_intent)

    return X, y_slot, y_intent


def padding_map(X, padding_value):
    """

    :param X: tensor(batch, max_seq_len)
    :param padding_value: token of pad
    :return: list of len batch with values of appropriate lengths
    """
    return torch.sum((X != padding_value), dim=1)

