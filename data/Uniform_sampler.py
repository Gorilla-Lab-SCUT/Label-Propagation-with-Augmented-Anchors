import torch
import random
import ipdb

class Sampler(object):
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class UniformBatchSampler(Sampler):
    def __init__(self, per_category, category_index_list, imgs, select_category):

        self.per_category = per_category
        self.category_index_list = category_index_list
        self.imgs = imgs
        self.batch_size = per_category * len(select_category)
        self.select_category = select_category
        self.batch_num = 10000  ### this kind of loop not end  ###len(self.imgs) // self.batch_size
    def __iter__(self):
        for bat in range(self.batch_num):
            batch = []
            for i in range(len(self.select_category)):   ##################### category_index_list[i] set to empty when this category is dropped
                batch = batch + random.sample(self.category_index_list[self.select_category[i]], self.per_category)
            random.shuffle(batch)
            yield batch

        # for idx in self.sampler:
        #     batch.append(idx)    ########### just the index of the image
        #     if len(batch) == self.batch_size:
        #         yield batch
        #         batch = []
        # if len(batch) > 0 and not self.drop_last:
        #     yield batch

    def __len__(self):
        return self.batch_num
        # else:
        #     return (len(self.sampler) + self.batch_size - 1) // self.batch_size
