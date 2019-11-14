import torch
import torch.nn as nn

torch.manual_seed(1)


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim=64):
        """
        :param vocab_size: number of words in the vocabulary
        :param embedding_dim: size of the embedding vector
        :param hidden_dim: layer size of the bilstm network (size of vector h)
        """
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab_size-1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x, lengths):
        """
        :param x: tensor of word idx of size (batch_size, seq_length)
        :param lengths: tensort of lengths
        :return: bilstm otput of size (batch_size, seq_len, hidden_size)
        https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch
        """
        orig_len = x.shape[1]
        embeds = self.word_embeds(x)

        lengths, sort_idx = lengths.sort(0, descending=True)
        embeds = embeds[sort_idx]

        lstm_input = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)
        h_out, (_, _) = self.lstm(lstm_input)
        h_out, _ = nn.utils.rnn.pad_packed_sequence(h_out, batch_first=True, total_length=orig_len)

        _, unsort_idx = sort_idx.sort(0)
        h_out = h_out[unsort_idx]

        return h_out


class SlotAttention(nn.Module):
    def __init__(self, n_features=64):
        super(SlotAttention, self).__init__()
        self.attention = nn.Linear(n_features, n_features)

    def forward(self, x):
        """
        :param x: hidden states of LSTM (batch_size, seq_len, hidden_size)
        :return: slot attention vector of size (batch_size, seq_len, hidden_size)
        """
        weights = self.attention(x)  # (batch_size, hidden_size, hidden_size) weight of attention
        output = torch.matmul(x, weights)
        output = torch.matmul(output, torch.transpose(x, 1, 2))
        output = torch.matmul(output, x)
        return output


class IntentAttention(nn.Module):
    def __init__(self, n_features=64):
        super(IntentAttention, self).__init__()
        self.attention = nn.Linear(n_features, n_features)

    def forward(self, x):
        """

        :param x: hidden states of LSTM (batch_size, seq_len, hidden_size)
        :return: intent vector of size (batch_size, hidden_size)
        """
        weights = self.attention(x)  # (batch_size, hidden_size, hidden_size) weight of attention
        output = torch.matmul(x, weights)
        output = torch.matmul(output, torch.transpose(x, 1, 2))
        output = torch.matmul(output, x)
        output = torch.sum(output, 1)
        return output
