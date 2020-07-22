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


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class DClassifierForSource(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True, nClass=10):
        super(DClassifierForSource, self).__init__(weight, size_average)
        self.nClass = nClass

    def forward(self, input):
        # _assert_no_grad(target)
        batch_size = input.size(0)

        prob = F.softmax(input, dim=1)

        ####################################### tang's original loss
        # if (prob.data[:, -1] == 1).sum() != 0:
        #     soft_weight = torch.FloatTensor(batch_size, self.nClass+1).fill_(1)
        #     temp = torch.FloatTensor(batch_size).fill_(1)
        #     temp[prob.data.cpu()[:, -1]==1] = 0.9999
        #     soft_weight[:, -1] = temp
        #     soft_weight_var = Variable(soft_weight).cuda()
        #     #loss = (1 - prob * soft_weight_var).log().mul(weight_var).sum(1).mean()
        #     loss = (1 - prob[:, -1] * soft_weight_var[:, -1]).log().mean()
        # else:
        # #    loss = (1 - prob).log().mul(weight_var).sum(1).mean()
        #     loss = (1 - prob[:, -1]).log().mean()

        ###################################### modified domain confusion loss
        if (prob.data[:, :self.nClass].sum(1) == 0).sum() != 0:  ########### in case of log(0)
            soft_weight = torch.FloatTensor(batch_size).fill_(0)
            soft_weight[prob[:, :self.nClass].sum(1).data.cpu() == 0] = 1e-6
            soft_weight_var = Variable(soft_weight).cuda()
            loss = -((prob[:, :self.nClass].sum(1) + soft_weight_var).log().mean())
            # loss = (-(1 - prob[:, -1] * soft_weight_var[:, -1]).log().mean() - (prob[:, -1] + soft_weight_zero_var[:, -1]).log().mean())/2
        else:
            loss = -(prob[:, :self.nClass].sum(1).log().mean())
        return loss