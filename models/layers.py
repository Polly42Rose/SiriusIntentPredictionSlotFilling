import torch
import torch.nn as nn

torch.manual_seed(1)


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim=64):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab_size-1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

    def forward(self, x):
        embeds = self.word_embeds(x).view(len(x[0]), len(x), -1)
        h_out, (_, _) = self.lstm(embeds)
        return h_out
