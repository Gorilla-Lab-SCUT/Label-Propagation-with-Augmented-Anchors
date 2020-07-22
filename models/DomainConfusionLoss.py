import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb
import time

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

def _process_zero_value(tensor):
    if (tensor == 0).sum() != 0:
        eps = torch.FloatTensor(tensor.size()).fill_(0)
        eps[tensor.data.cpu() == 0] = 1e-6
        eps = Variable(eps).cuda()
        tensor = tensor +eps
    return tensor

class ConsistTarget(_WeightedLoss):

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True, nClass = 10):
        super(ConsistTarget, self).__init__(weight, size_average)
        self.nClass = nClass

    def forward(self, input, type='type1', dis='cross_entropy'):
        ############################################# type #######################
        ## type 1: all p^{st}
        ## type 2: all P^s and P^t
        ## type 3: mix p^{st}, p^s and p^t, same with tang
        ############################################# dis ########################
        ## cross_entropy: same with tang, em + consistent ??
        ## kl           : consistent only ??
        ## l1           : consistent only ??
        ######################################################################
        batch_size = input.size(0)
        prob = F.softmax(input, dim=1)
        prob_s = F.softmax(input[:, :self.nClass], dim=1)
        prob_t = F.softmax(input[:, self.nClass:], dim=1)

        if type == 'type1':
            prob_s_used = prob[:, :self.nClass]
            prob_t_used = prob[:, self.nClass:]
            if dis == 'l1':
                loss = torch.norm(prob_s_used - prob_t_used, p=1)
            elif dis == 'cross_entropy':
                prob_s_used = _process_zero_value(prob_s_used)
                prob_t_used = _process_zero_value(prob_t_used)
                loss = - (prob_s_used.log().mul(prob_t_used)).sum(1).mean() - (prob_t_used.log().mul(prob_s_used)).sum(1).mean()
                loss = loss * 0.5
            elif dis == 'kl':
                prob_s_used = _process_zero_value(prob_s_used)
                prob_t_used = _process_zero_value(prob_t_used)
                loss = - (prob_s_used.log().mul(prob_t_used)).sum(1).mean() - (prob_t_used.log().mul(prob_s_used)).sum(1).mean() + (prob_s_used.log().mul(prob_s_used)).sum(1).mean() + (prob_t_used.log().mul(prob_t_used)).sum(1).mean()
                loss = loss * 0.5
        elif type == 'type2':
            prob_s_used = prob_s
            prob_t_used = prob_t
            if dis == 'l1':
                loss = torch.norm(prob_s_used - prob_t_used, p=1)
            elif dis == 'cross_entropy':
                prob_s_used = _process_zero_value(prob_s_used)
                prob_t_used = _process_zero_value(prob_t_used)
                loss = - (prob_s_used.log().mul(prob_t_used)).sum(1).mean() - (prob_t_used.log().mul(prob_s_used)).sum(1).mean()
                loss = loss * 0.5
            elif dis == 'kl':
                prob_s_used = _process_zero_value(prob_s_used)
                prob_t_used = _process_zero_value(prob_t_used)
                loss = - (prob_s_used.log().mul(prob_t_used)).sum(1).mean() - (prob_t_used.log().mul(prob_s_used)).sum(1).mean() + (prob_s_used.log().mul(prob_s_used)).sum(1).mean() + (prob_t_used.log().mul(prob_t_used)).sum(1).mean()
                loss = loss * 0.5
        elif type == 'type3':  ########### tang's method
            prob_s_used = prob[:, :self.nClass]
            prob_t_used = prob[:, self.nClass:]
            if dis == 'l1':
                raise NotImplementedError
            elif dis == 'cross_entropy':
                prob_s_used = _process_zero_value(prob_s_used)
                prob_t_used = _process_zero_value(prob_t_used)
                prob_s = _process_zero_value(prob_s)
                prob_t = _process_zero_value(prob_t)
                loss = - (prob_s_used.log().mul(prob_t)).sum(1).mean() - (prob_t_used.log().mul(prob_s)).sum(1).mean()
                loss = loss * 0.5
            else:
                raise NotImplementedError

        return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, kernel_type='gau'):

    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    if kernel_type == 'gau':
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)    #### distance
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]   #### similarity
        return sum(kernel_val) / len(kernel_val)
    elif kernel_type == 'linear':   #### not converge
        L2_distance = torch.mm(total, torch.transpose(total, 0, 1))
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]   #### similarity
        return sum(kernel_val) / len(kernel_val)

    elif kernel_type == 'cosine':
        total = F.normalize(total, dim=1, p=2)
        return torch.mm(total, torch.transpose(total, 0, 1))
    else:
        raise NotImplementedError

#########################################################################################################
###################################  ONLY one kernel is adopted ######################################
##########################################################################################################

def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, source_label=None, target_label=None, intra_only=True, number_cate=31, kernel_type='gau'):
    begin = time.time()
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,     ##################### instance level similarity
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma, kernel_type=kernel_type)
    if intra_only:
        print(time.time() - begin)
        S_value = torch.zeros(number_cate).cuda()
        S_count = torch.zeros(number_cate).cuda()
        T_value = torch.zeros(number_cate).cuda()
        T_count = torch.zeros(number_cate).cuda()
        ST_value = torch.zeros(number_cate, number_cate).cuda()
        ST_count = torch.zeros(number_cate, number_cate).cuda()
        print(time.time() - begin)
        ######################################### for source data
        for i_index in range(batch_size):
            for j_index in range(batch_size):  ########### what's the influence of self-similarity.
                if source_label[i_index] == source_label[j_index]:
                    S_value[source_label[i_index]] += kernels[i_index, j_index]
                    S_count[source_label[i_index]] += 1
                if target_label[i_index] == target_label[j_index]:
                    T_value[target_label[i_index]] += kernels[i_index + batch_size, j_index + batch_size]
                    T_count[target_label[i_index]] += 1

        ######################################### for source and target data
        for i_index in range(batch_size):
            for j_index in range(batch_size):
                ST_value[source_label[i_index], target_label[j_index]] += kernels[i_index, j_index + batch_size]
                ST_count[source_label[i_index], target_label[j_index]] += 1

        S_count[S_count == 0] = 1
        S_value = S_value / S_count
        T_count[T_count ==0] = 1
        T_value = T_value / T_count
        ST_count[ST_count == 0] = 1
        ST_value = ST_value / ST_count
        print(time.time() - begin)
        # ######################################### for target data
        # for i_index in range(batch_size):
        #     for j_index in range(batch_size):
        #         T_value[target_label[i_index], target_label[j_index]] += kernels[i_index + batch_size, j_index + batch_size]
        #         T_count[target_label[i_index], target_label[j_index]] += 1
        # T_count[T_count ==0] = 1
        # T_value = T_value / T_count
        # print(time.time() - begin)
        #
        # ######################################### for source and target data
        # for i_index in range(batch_size):
        #     for j_index in range(batch_size):
        #         ST_value[source_label[i_index], target_label[j_index]] += kernels[i_index, j_index + batch_size]
        #         ST_count[source_label[i_index], target_label[j_index]] += 1
        # ST_count[ST_count == 0] = 1
        # ST_value = ST_value / ST_count
        # print(time.time() - begin)

        loss_intra = (S_value + T_value - ST_value.diag() * 2)
        loss_intra = loss_intra[loss_intra != 0].mean()
        return loss_intra
        # loss_intra = (value * eye_index)[torch.gt(value * eye_index, 0)].mean()  ###### whether need to consider the bad condition, e.g., value[i,i] < 0
        # loss_inter = (-value * (1-eye_index))[torch.gt(-value * (1-eye_index), 0)].mean()   ##### many elements don't satisfy the constrain


    else:
        ST_value = torch.zeros(number_cate, number_cate).cuda()
        ST_count = torch.zeros(number_cate, number_cate).cuda()
        for i_index in range(batch_size):
            for j_index in range(batch_size):
                ST_value[source_label[i_index], target_label[j_index]] += kernels[i_index, j_index + batch_size]
                ST_count[source_label[i_index], target_label[j_index]] += 1
        ST_count[ST_count == 0] = 1
        ST_value = ST_value / ST_count
        print(time.time() - begin)
        eye_index = torch.eye(number_cate).cuda()
        ST_value_intra = ((ST_value * eye_index)[(ST_value * eye_index) != 0]).mean()
        ST_value_inter = ((ST_value * (1-eye_index))[(ST_value * (1-eye_index)) != 0]).mean()
        return -ST_value_intra + ST_value_inter



        # if intra_only:
        #     return loss_intra
        # else:
        #     S_value_nozero = S_value[S_value != 0].mean()
        #     T_value_nozero = T_value[T_value != 0].mean()
        #     eye_index = torch.eye(number_cate).cuda()
        #     ST_value_nozero = (ST_value * (1 - eye_index))[(ST_value * (1 - eye_index)) != 0].mean()
        #     loss_inter = S_value_nozero + T_value_nozero - ST_value_nozero * 2
        #     return (loss_intra - loss_inter)


# def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, source_label=None, target_label=None, intra_only=False, number_cate=31):
#     batch_size = int(source.size()[0])
#     kernels = guassian_kernel(source, target,
#         kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
#
#     S_value = torch.zeros(number_cate, number_cate)
#     S_count = torch.zeros(number_cate, number_cate)
#     T_value = torch.zeros(number_cate, number_cate)
#     T_count = torch.zeros(number_cate, number_cate)
#     ST_value = torch.zeros(number_cate, number_cate)
#     ST_count = torch.zeros(number_cate, number_cate)
#     ######################################### for source data
#     for i_index in range(batch_size):
#         for j_index in range(i_index+1):
#             S_value[source_label[i_index], source_label[j_index]] = kernels[i_index, j_index]
#             S_count[source_label[i_index], source_label[j_index]] += 1
#     S_count[S_count ==0] = 1
#     S_value = S_value / S_count
#     ######################################### for target data
#     for i_index in range(batch_size):
#         for j_index in range(i_index+1):
#             T_value[target_label[i_index], target_label[j_index]] = kernels[i_index+batch_size, j_index+batch_size]
#             T_count[target_label[i_index], target_label[j_index]] += 1
#     T_count[T_count ==0] = 1
#     T_value = T_value / T_count
#     ######################################### for source and target data
#     for i_index in range(batch_size):
#         for j_index in range(batch_size):
#             ST_value[source_label[i_index], target_label[j_index]] = kernels[i_index, j_index+batch_size]
#             ST_count[source_label[i_index], target_label[j_index]] += 1
#     ST_count[ST_count ==0] = 1
#     ST_value = ST_value / ST_count
#
#     value = S_value + T_value - ST_value * 2
#     eye_index = torch.eye(number_cate)
#
#     loss_intra = (value * eye_index).sum() / number_cate
#     loss_inter = (value * (1-eye_index)).sum() / (number_cate * (number_cate-1))
#
#     if intra_only:
#         return loss_intra
#     else:
#         return loss_inter + loss_intra

