import torch
import torch.nn as nn

torch.manual_seed(1)

# pack padding sequence
class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim=64):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab_size-1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

    def forward(self, x):
        embeds = nn.utils.rnn.pad_sequence(x, batch_first=True)
        embeds = self.word_embeds(embeds).view(len(embeds[0]), len(embeds), -1)
        h_out, (_, _) = self.lstm(embeds)
        return h_out
