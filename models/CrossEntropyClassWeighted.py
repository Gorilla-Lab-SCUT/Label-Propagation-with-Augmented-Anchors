import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

class CrossEntropyClassWeighted(_Loss):

    def __init__(self, size_average=True, ignore_index=-100, reduce=None, nClass=10):
        super(CrossEntropyClassWeighted, self).__init__(size_average)
        self.nClass = nClass
        self.ignore_index = ignore_index


    def forward(self, input, target, weight=None, reduce=None):
        return F.cross_entropy(input, target, weight, ignore_index=self.ignore_index, reduce=reduce)