import torch
import torch.nn as nn
from models.layers import BiLSTM, IntentAttention, SlotAttention, IterModel


class BiLSTMmodel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=100, n_slots=129, n_intents=29, hidden_dim=64):
        super(BiLSTMmodel, self).__init__()

        self.voc_size = vocab_size
        self.emb_dim = embedding_dim
        self.hidden_size = hidden_dim
        self.n_slots = n_slots
        self.n_intetns = n_intents

        self.rnn = BiLSTM(vocab_size, embedding_dim, hidden_dim=64)
        self.att_intent = IntentAttention(n_features=hidden_dim)
        self.att_slot = SlotAttention(n_features=hidden_dim)

        self.iter_sf_id = IterModel(hidden_dim, n_intents, n_slots)

        # For temporary testing of layers
        self.relu = nn.ReLU()
        self.slot_linear = nn.Linear(in_features=hidden_dim, out_features=n_slots)
        self.intent_linear = nn.Linear(in_features=hidden_dim, out_features=n_intents)

    def forward(self, x, lengths):
        output = self.rnn(x, lengths)
        slot = self.att_slot(output)
        intent = self.att_intent(output)

        intent, slot = self.iter_sf_id(output, slot, intent)
        return torch.transpose(slot, 1, 2), intent

        ### LSTM + attention + ReLU + Linear
        # output = self.rnn(x, lengths)
        # slot = self.att_slot(output)
        # intent = self.att_intent(output)
        # slot = self.relu(slot)
        # intent = self.relu(intent)
        # slot = self.slot_linear(slot)
        # intent = self.intent_linear(intent)
        # return torch.transpose(slot, 1, 2), intent

        ### LSTM + ReLU + Linear
        # output = self.rnn(x, lengths)
        # output = self.relu(output)
        # slot = self.slot_linear(output)
        # intent = self.intent_linear(torch.sum(output, dim=1))
        # return torch.transpose(slot, 1, 2), intent
        # # (batch, n_slots, seq_len), (batch, n_intents)
