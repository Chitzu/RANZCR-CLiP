import numpy as np
import torch
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import precision_score, recall_score, f1_score


class StatsManager:

    def __init__(self, config):
        self.config = config

    #implement
    # def get_stats(self, predictions, labels):
    #     return None

    ## Accuracy
    def get_stats(self, predictions, labels):
        accuracy = MulticlassAccuracy(num_classes=10)
        pred = torch.tensor(np.array(predictions))
        lab = torch.tensor(np.array(labels))
        acc = accuracy(pred, lab)

        return acc


def calculate_metrics(true_labels, predicted_labels):
    true_labels = true_labels.tolist()
    macro_precision = precision_score(true_labels, predicted_labels, average='macro')

    macro_recall = recall_score(true_labels, predicted_labels, average='macro')

    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

    micro_precision = precision_score(true_labels, predicted_labels, average='micro')

    micro_recall = recall_score(true_labels, predicted_labels, average='micro')

    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')

    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1