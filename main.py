##############################################################################
#
#
##############################################################################
import json
import os
import shutil
import time
import math
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from data.prepare_data import generate_dataloader  # Prepare the data and dataloader
from models.alexnet import alexnet  # The model construction
from models.resnet import resnet  # The model construction
from models.resnet_dsbn import resnet as resnet_dsbn  # The model construction
from opts import opts  # The options for the project

from trainer import download_feature_and_pca_clustering
from trainer import download_feature_and_pca_label_prob
import ipdb

best_prec1 = 0





def main():
    global args, best_prec1
    args = opts()

    if args.arch.find('resnet') != -1:
        model = resnet(args)
    else:
        raise ValueError('Unavailable model architecture!!!')
    # define-multi GPU
    model = torch.nn.DataParallel(model).cuda()
    print(model)



    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('\n-------------------------------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------')
    log.close()


    cudnn.benchmark = True
    # process the data and prepare the dataloaders.
    # source_train_loader, source_val_loader, target_train_dataset, val_loader, source_val_loader_cluster, val_loader_cluster = generate_dataloader(args)
    source_train_loader_ce, source_train_dataset, target_train_loader_ce, target_train_dataset, source_val_loader, target_val_loader = generate_dataloader(args)
    if args.pseudo_type == 'cluster':  ### the AO of CAN
        clusering_labels_for_path = download_feature_and_pca_clustering(0, source_val_loader,
                                                                        target_val_loader, model, args)
    elif args.pseudo_type == 'lp':
        clusering_labels_for_path = download_feature_and_pca_label_prob(0, source_val_loader,
                                                                        target_val_loader, model,
                                                                        args)
    else:
        raise NotImplementedError



if __name__ == '__main__':
    main()





