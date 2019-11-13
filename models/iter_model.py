import torch
import torch.nn as nn

torch.manual_seed(1)
iteration_num = 3

class IterModel(nn.Module):
    def __init__(self, hid_size, n_labels, n_slots):
        super(IterModel, self).__init__()
        self.n_labels = n_labels
        self.n_slots = n_slots
        self.W_SF = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.V_SF = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.V1_ID = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.V2_ID = nn.Linear(in_features=hid_size, out_features=hid_size, bias=True)
        self.W_ID = nn.Linear(in_features=hid_size, out_features=hid_size, bias=None)
        self.W_inte_ans = nn.Linear(in_features=hid_size * 2, out_features=n_labels, bias=None)

    def forward(self, h, c_slot, c_inte):
        """
        params: h - hidden states [batch_size, len_sents, hid_size]
                c_slot - slot context vector [batch_size, len_sents, hid_size]
                c_inte - intent context vector [batch_size, hid_size]
        return: r_inte - probabilities of each intent [n_labels]
        """
        r_inte = c_inte
        batch_size = c_slot.shape[0]
        len_sents = c_slot.shape[1]

        for iter in range(iteration_num):
            # SF subnet
            f = self.V_SF(torch.tanh(c_slot[:, 0, :] + self.W_SF(r_inte).float()))
            for i in range(1, c_slot.shape[1]):
                f += self.V_SF(nn.functional.tanh(c_slot[:, i, :] + self.W_SF(r_inte)))
            r_slot = torch.stack(
                [torch.stack([f[b] * c_slot[b, i, :] for i in range(len_sents)]) for b in range(batch_size)])

            # ID subnet
            hid_features = self.V1_ID(r_slot)
            slot_features = self.V2_ID(h)
            sum_for_softmax = torch.stack(
                [torch.stack([torch.exp(self.W_ID(nn.functional.tanh(hid_features[:, i, :] + slot_features[:, j, :])))
                              for j in range(len_sents)]) for i in range(len_sents)])
            sum_for_softmax = torch.sum(sum_for_softmax, 3)
            sum_for_softmax = torch.sum(sum_for_softmax, 2)
            atts = torch.stack([sum_for_softmax[i][i] / sum(sum_for_softmax[i, :]) for i in range(len_sents)])
            r = torch.zeros(c_inte.shape)
            for i in range(len_sents):
                r += atts[i] * r_slot[:, i, :]
            r_inte = r + c_inte

        intent_output = nn.functional.softmax(self.W_inte_ans(torch.cat((r_inte, h[:, len_sents - 1, :]), 1)))
        return intent_output
