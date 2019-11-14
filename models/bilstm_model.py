import torch
import torch.nn as nn
from models.layers import BiLSTM


class BiLSTMmodel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=100, n_slots=129, n_intents=29, hidden_dim=64):
        super(BiLSTMmodel, self).__init__()

        self.voc_size = vocab_size
        self.emb_dim = embedding_dim
        self.hidden_size = hidden_dim
        self.n_slots = n_slots
        self.n_intetns = n_intents

        self.rnn = BiLSTM(vocab_size, embedding_dim, hidden_dim=64)
        self.slot_linear = nn.Linear(in_features=hidden_dim, out_features=n_slots)
        self.intent_linear = nn.Linear(in_features=hidden_dim, out_features=n_intents)
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        output = self.rnn(x, lengths)
        output = self.relu(output)
        slot = self.slot_linear(output)
        intent = self.intent_linear(torch.sum(output, dim=1))
        return torch.transpose(slot, 1, 2), intent
        # (batch, n_slots, seq_len), (batch, n_intents)
