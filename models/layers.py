import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# torch.manual_seed(1)

# torch.manual_seed(1)
iteration_num = 3


# pack padding sequence
class IterModel(nn.Module):
    def __init__(self, hid_size, n_labels, n_slots):
        super(IterModel, self).__init__()
        self.n_labels = n_labels    
        self.n_slots = n_slots
        self.W_SF = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.V_SF = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.V1_ID = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.V2_ID = nn.Linear(in_features=hid_size, out_features=hid_size, bias=True)
        self.W_ID = nn.Linear(in_features=hid_size, out_features=1, bias=None)
        self.W_inte_ans = nn.Linear(in_features=hid_size * 2, out_features=n_labels, bias=None)
        self.W_slot_ans = nn.Linear(in_features=hid_size * 2, out_features=n_slots, bias=None)

    def forward(self, h, c_slot, c_inte):
        """
        params: h - hidden states [batch_size, len_sents, hid_size]
                c_slot - slot context vector [batch_size, len_sents, hid_size]
                c_inte - intent context vector [batch_size, hid_size]
        return: intent_output - intent features prepared for softmax [batch_size, n_labels]
                slot_output - slot features prepared for softmax [batch_size, len_sents, n_slots]
        """
        r_inte = c_inte
        batch_size = c_slot.shape[0]
        len_sents = c_slot.shape[1]

        for iter in range(iteration_num):
            print('SF')
            # SF subnet
            f = torch.tanh(c_slot + self.W_SF(r_inte).unsqueeze(1))  # [batch_size, len_sents, hid_size]
            f = self.V_SF(f)  # [batch_size, len_sents, hid_size]
            f = torch.sum(f, dim=1)  # [batch_size, hid_size]
            r_slot = f.unsqueeze(1) * c_slot  # [batch_size, len_sents, hid_size]


            # ID subnet
            print('ID')
            slot_features = self.V1_ID(r_slot)  # [batch_size, len_sents, hid_size]
            hid_features = self.V2_ID(h)  # [batch_size, len_sents, hid_size]
            alphas = []
            for i in tqdm(range(slot_features.shape[1])):
                e_i = slot_features[:, i, :].unsqueeze(1)  # [batch_size, 1, hid_size]
                e_i = self.W_ID(torch.tanh(e_i + hid_features))  # [batch_size, len_sents, 1]
                e_i = e_i.squeeze(-1)  # [batch_size, len_sents]
                alpha_i = torch.nn.functional.softmax(e_i, dim=1)  #  # [batch_size, len_sents]
                alpha_i = alpha_i[:, i]  # [batch_size]
                alphas.append(alpha_i)
            alphas = torch.stack(alphas)  # [len_seq, batch]
            alphas = torch.transpose(alphas, 0, 1)  # [batch, len_seq]

            #  r_slot - [batch_size, len_sents, hid_size]
            #  alphas - [batch, len_seq]

            r = r_slot * alphas.unsqueeze(-1)  # [batch_size, len_sents, hid_size]
            r = torch.sum(r, dim=1)  # [batch_size, hid_size]
            r_inte = r + c_inte

        intent_output = self.W_inte_ans(torch.cat((r_inte, h[:, len_sents - 1, :]), dim=1))  # [batch_size, n_intents]
        slot_output = torch.cat((h, r_slot), dim=-1)   # [batch_size, seq_len, 2*hidden_size]
        slot_output = self.W_slot_ans(slot_output)   # [batch_size, seq_len, n_slots]

        return slot_output, intent_output


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
        weights = self.attention(x)  # (batch_size, seq_len, hidden_size) - temporary weight
        weights = torch.matmul(weights, torch.transpose(x, 1, 2))  # (batch_size, hidden_size, hidden_size) - att matrix
        weights = F.softmax(weights, dim=2)
        output = torch.matmul(weights, x)
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
        weights = self.attention(x)  # (batch_size, seq_len, hidden_size) - temporary weight
        # output = torch.matmul(x, weights)
        weights = torch.matmul(weights, torch.transpose(x, 1, 2))  # (batch_size, seq_len, seq_len) - att matrix
        weights = F.softmax(weights, dim=2)
        output = torch.matmul(weights, x)
        output = torch.sum(output, 1)
        return output
