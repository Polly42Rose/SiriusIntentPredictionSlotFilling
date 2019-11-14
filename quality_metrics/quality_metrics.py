import torch

def intent_accuracy(labels, preds):
    preds = torch.argmax(preds, dim=1)
    list_len = labels.shape[0]
    return len([i for i in range(list_len) if labels[i] == preds[i]])/labels.shape[0]
