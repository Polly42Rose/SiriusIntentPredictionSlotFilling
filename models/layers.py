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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

    def forward(self, x, length):
        """
        :param x: tensor of word idx of size (batch_size, seq_length)
        :return: bilstm otput of size (seq_len, batch_size, hidden_size)
        https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch
        """
        embeds = self.word_embeds(x)
        lstm_input = nn.utils.rnn.pack_padded_sequence(embeds, length)
        h_out, (_, _) = self.lstm(lstm_input)
        h_out, _ = nn.utils.rnn.pad_packed_sequence(h_out)
        return h_out


class SlotAttention(nn.Module):
    def __init__(self, n_features=64):
        super(SlotAttention, self).__init__()


    def forward(self, x):
        """

        :param x: hidden states of LSTM
        :return:
        """



class IntentAttention(nn.Module):
    def __init__(self):
        super(IntentAttention, self).__init__()

    def forward(self, x):
        pass
