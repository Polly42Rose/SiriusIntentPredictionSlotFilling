import torch
import numpy as np


def intent_accuracy(labels, preds):
    preds = torch.argmax(preds, dim=1)
    list_len = labels.shape[0]
    return len([i for i in range(list_len) if labels[i] == preds[i]]) / labels.shape[0]


def slot_f1(labels, preds, num_slot):
    preds = torch.argmax(preds, dim=2)
    list_len = labels.shape[0]
    precision = 0
    recall = 0
    for i, label in enumerate(labels):
        pred = preds[i]
        true_pos = [len([j for j in range(len(label)) if label[j] == pred[j] and label[j] == k]) for k in range(num_slot)]
        false_pos = [len([j for j in range(len(label)) if pred[j] != k and pred[j] == k]) for k in range(num_slot)]
        false_neg = [len([j for j in range(len(label)) if pred[j] == k and label[j] != k]) for k in range(num_slot)]
        precision += np.mean([true_pos[i] / (true_pos[i] + false_pos[i]) if true_pos[i] + false_pos[i] > 0 else 0 
                              for i in range(num_slot)])
        recall += np.mean([true_pos[i] / (true_pos[i] + false_neg[i]) if true_pos[i] + false_neg[i] > 0 else 0
                           for i in range(num_slot)])
    precision = precision / list_len
    recall = recall / list_len
    recall = np.mean(np.array(recall))
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def sentence_accuracy(i_labels, i_preds, s_labels, s_preds):
    ok = 0
    i_preds = torch.argmax(i_preds, dim=1)
    s_preds = torch.argmax(s_preds, dim=2)
    for i, i_label in enumerate(i_labels):
        i_pred = i_preds[i]
        s_label = s_labels[i]
        s_pred = s_preds[i]
        if i_pred != i_label or len([j for j in range(len(s_label)) if s_label[j] == s_pred[j]]) != len(s_label):
            continue
        ok += 1
    return ok / len(i_preds)
