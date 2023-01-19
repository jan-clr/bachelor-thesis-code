import torch
import torch.nn as nn


class CrossEntropyConsLoss(object):
    def __init__(self, weight=None, hard_labels=True):
        self.weight = weight
        self.hard_labels = hard_labels
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)

    def __call__(self, pred_stu: torch.Tensor, pred_tch: torch.Tensor):
        target = None
        if self.hard_labels:
            target = torch.argmax(pred_tch, dim=1)
        else:
            target = nn.functional.softmax(pred_tch, dim=1)

        return self.loss_fn(pred_stu, target)


def cross_entropy_cons_loss(pred_stu: torch.Tensor, pred_tch: torch.Tensor, hard_labels=True):
        """
        Takes the softmax of the teacher prediction and returns the CrossEntropyLoss between the teacher and student predictions. Gradient only applied to student.
        :param pred_stu: student prediction logits
        :param pred_tch: teacher prediction logits
        :param hard_labels: use argmax on teacher predictions
        :return:
        """
        loss_fn = nn.CrossEntropyLoss()
        target = None
        if hard_labels:
            target = torch.argmax(pred_tch, dim=1)
        else:
            target = nn.functional.softmax(pred_tch, dim=1)

        return loss_fn(pred_stu, target)
