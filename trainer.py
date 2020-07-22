import time
import torch
import os
import math
# import faiss
import numpy as np
from numpy import linalg as LA
from collections import Counter
import copy
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
import ipdb
import torch.nn.functional as F
import random
from models.DomainConfusionLoss import MMD
import time
import scipy

from pynndescent import NNDescent, PyNNDescentTransformer

def download_feature_and_pca_label_prob(epoch, train_loader, val_loader, model, args, moving_feature_centeriod_t=False):
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write('Constructing graph with %s, and the solver is: %s' % (args.dis_gra, args.LPSolver))
    log.write('\n')
    ########    lam = 2 / (1 + math.exp(-1 * 10 * iteration / args.epochs)) - 1
    cos_threshold = (args.cos_threshold) * (1 - epoch / args.epochs)

    model.eval()
    #### step 1: prepare source and target features and labels
    source_feature_list = []
    source_labels = []
    source_feature_list_category = []
    for i in range(args.num_classes):
        source_feature_list_category.append([])  ######### each for one categoty

    for i, (input, target, img_path) in enumerate(train_loader):
        print('soruce center calculation', i)
        input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            if args.dsbn:
                feature_source, _ = model(input_var, ST='S')

            else:
                feature_source, _ = model(input_var)
        feature_source = feature_source.cpu()
        batchsize = feature_source.size(0)
        for j in range(batchsize):
            img_label = target[j]
            label_temp = torch.zeros(1, args.num_classes)
            label_temp[0][img_label] = 1
            source_feature_list.append(feature_source[j].view(1, feature_source.size(1)))
            source_labels.append(label_temp)
            source_feature_list_category[img_label].append(feature_source[j].view(1, feature_source.size(1)))


    target_feature_list = []
    GT_labels = []
    image_paths = []

    for i, (input, target, img_path) in enumerate(val_loader):
        print('target feature calculation', i)
        input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            if args.dsbn:
                feature_target, _ = model(input_var, ST='T')
            else:
                feature_target, _ = model(input_var)
        batchsize = feature_target.size(0)
        feature_target = feature_target.cpu()
        for j in range(batchsize):
            GT_labels.append(target[j].item())
            image_paths.append(img_path[j])
            target_feature_list.append(feature_target[j].view(1, feature_target.size(1)))


    source_feature, source_labels, NumS, target_feature = FeaturePreprocess(source_feature_list, source_feature_list_category, source_labels, target_feature_list, args, moving_feature_centeriod_t)
    NumT = len(target_feature_list)
    all_feature = torch.cat((source_feature, target_feature), dim=0)  ### (Ns + Nt) * d
    target_label_initial = torch.zeros(NumT, args.num_classes)  ## the initial state make no influence
    all_label = torch.cat((source_labels, target_label_initial), dim=0)  ### (Ns + Nt) * c
    for lpiteration in range(args.LPIterNum):
        NumST = NumS + NumT
        ############ step 2: calculate the graph, N = Ns + Nt
        if args.dis_gra == 'l2':  ## NOT USED. similarity graph with L2 based distance.
            all_f1 = torch.unsqueeze(all_feature, 1)  ## N * 1 * d
            all_f2 = torch.unsqueeze(all_feature, 0)  ## 1 * N * d
            weight = ((all_f1 - all_f2)**2).mean(2) ## N * N * d -> N*N
            weight = torch.exp(-weight/ (2 * args.cor))  ############# here the \sigma is set to 1 as default !!!!!!!!!!!!!!!!!!!
            if args.TopkGraph:
                values, indexes = torch.topk(weight, args.graphk)
                ################################# W + W^T， This is better than the 补全one.
                weight[weight < values[:, -1].view(-1, 1)] = 0
                weight = weight + torch.t(weight)

        elif args.dis_gra == 'mul':   #### only mul:-> vector mul; mul + l2 process-> cos dis
            ## It is still fast, since the matrix multiply is implemented in parallel.
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            log.write('start calculate graph with brute-force implementation:')
            log.write(time.asctime(time.localtime(time.time())))
            log.write('\n')
            weight = torch.matmul(all_feature, all_feature.transpose(0, 1))  ## N * N
            weight[weight < 0] = 0
            weight = weight ** args.graph_gama
            if args.TopkGraph:
                values, indexes = torch.topk(weight, args.graphk)
                weight[weight < values[:, -1].view(-1, 1)] = 0
            log.write('brute-force implementation end:')
            log.write(time.asctime(time.localtime(time.time())))
            log.write('\n')
            log.close()
            weight = weight + torch.t(weight)
        elif args.dis_gra == 'nndescent':  ## A fast way to construct the k-nearest graph.
            # ipdb.set_trace()
            nn_data = all_feature.numpy()
            weight = torch.zeros(all_feature.size(0), all_feature.size(0))
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            log.write('nn descent start:')
            print('nn descent start\n')
            log.write(time.asctime(time.localtime(time.time())))
            log.write('\n')

            knn_indices, knn_value = NNDescent(nn_data, "cosine", {}, args.graphk, random_state=np.random)._neighbor_graph
            weight.scatter_(1, torch.from_numpy(knn_indices), 1 - torch.from_numpy(knn_value).float())
            log.write('nndescent end:')
            print('nndescent end')
            log.write(time.asctime(time.localtime(time.time())))
            log.write('\n')
            log.close()
            weight = weight + torch.t(weight)


        weight.diagonal(0).fill_(0)  ### zero the diagonal

        ######################## Step3: label propagation, F = (I - \alpha S)^{-1} Y
        if args.LPSolver == 'CloseF':        ##### the closed-form solver
            D = weight.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
            D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, NumST)
            D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(NumST, 1)
            S = D1*weight*D2  ############ same with D3 = torch.diag(D_sqrt_inv)  S = torch.matmul(torch.matmul(D3, weight), D3)
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            log.write('closed form solution start:')
            log.write(time.asctime(time.localtime(time.time())))
            log.write('\n')
            PredST = torch.matmul(torch.inverse(torch.eye(NumST) - args.AlphaGraph * S + 1e-8), all_label)
            log.write('closed form solution end:')
            log.write(time.asctime(time.localtime(time.time())))
            log.write('\n')
            log.close()
        elif args.LPSolver == 'CG':  ### the conjugate gradient solver, faster.
            D = weight.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
            D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, NumST)
            D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(NumST, 1)
            S = D1*weight*D2  ############ same with D3 = torch.diag(D_sqrt_inv)  S = torch.matmul(torch.matmul(D3, weight), D3)
            ################## A * X =  all_label
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            count = time.time()
            log.write('cg solution start:')
            log.write('time: %3f' % (time.time() - count))
            log.write('\n')
            ################################## nn solver, not the

            A = torch.eye(NumST) - args.AlphaGraph * S + 1e-8
            PredST = cg_solver(A, all_label)
            log.write('cg solution end:')
            log.write('time: %3f' % (time.time() - count))
            log.write('\n')
            log.close()

        else:
            raise NotImplementedError

        PredT = PredST[NumS:, :]
        #  all_label[NumS:, :] = PredT  ### keep the unlabeled data with no prior
        if lpiteration == 0:
            if args.noise_flag:   #### ablation study, replace pseudo labels with defined noise level.
                LabelT = torch.Tensor(GT_labels).long().clone()
                num_t = LabelT.size(0)
                wrong_num = int(args.noise_level * num_t)
                all_list = list(range(num_t))
                selecte_list = random.sample(all_list, wrong_num)
                for wrong_index in selecte_list:
                    all_class_list = list(range(args.num_classes))
                    ## remove the accurate one
                    all_class_list.remove(GT_labels[wrong_index])
                    random_label = random.choice(all_class_list)
                    LabelT[wrong_index] = random_label
                LabelT = LabelT.view(-1, 1)
            else:
                _, LabelT = torch.topk(PredT, 1)
        else:
            _, LabelT = torch.topk(PredT, 1)
        PredT = PredT / PredT.sum(1).view(-1, 1)    ### normalize
        PredT_nozeso = process_zero_value(PredT, nozero=True)
        PredT_zero = process_zero_value(PredT, nozero=False)
        ####### prediction confidence of the label propagation algorithm
        Instance_confidence = 1 + (PredT_zero * PredT_nozeso.log()).sum(1) / math.log(args.num_classes)   ### 1 - H(Z)/log(args.num_classes) #  torch.Tensor(size T number)
        #if args.entropy_weight:
        #    print(Instance_confidence)
        acc_cluster_label = torch.sum(torch.Tensor(GT_labels).long() == LabelT[:, 0]).item() / NumT
        if args.category_mean:
            acc_count = torch.zeros(args.num_classes)
            all_count = torch.zeros(args.num_classes)
            for i in range(NumT):
                all_count[GT_labels[i]] += 1
                if LabelT[i].item() == GT_labels[i]:
                    acc_count[GT_labels[i]] += 1
            acc_for_each_class1 = acc_count / all_count

        log = open(os.path.join(args.log, 'log.txt'), 'a')
        if args.category_mean:
            log.write("\nAcc for each class1: ")
            for i in range(12):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class1[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class1[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class1[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class1[i]))
            log.write("Avg. over all classes: %3f" % acc_for_each_class1.mean())
            print("Avg. over all classes: %3f" % acc_for_each_class1.mean())
        else:
            log.write("\n   iteration: %d,   Cluster Label Acc: %3f" % (lpiteration, acc_cluster_label))
            print("      Cluster Label Acc: %3f" % (acc_cluster_label))
        log.write("  cos threshold: %3f" % (cos_threshold))
        log.close()

        ############################ add the predicted target center to the known points or replace the source known points with target centers
        ########## calculate the target pseudo center  target_feature
        target_feature_list_pseudo = []
        target_confidence_list = []
        for i in range(args.num_classes):
            target_feature_list_pseudo.append([])  ######### each for one categoty
            target_confidence_list.append([])
        for i in range(NumT):
            psuedo_label = LabelT[i]
            target_feature_list_pseudo[psuedo_label].append(target_feature[i].view(1, target_feature.size(1)))
            target_confidence_list[psuedo_label].append(Instance_confidence[i].view(1, -1))
        target_center_labels = torch.eye(args.num_classes)
        target_center_feature = torch.zeros(args.num_classes, args.fea_dim)
        # ipdb.set_trace()
        ################### for predicted category with 0 samples
        noempty_list = []

        for i in range(args.num_classes):
            if len(target_feature_list_pseudo[i]) == 0:
                continue
            noempty_list.append(i)
            if args.entropy_weight:
                weight_for_this_category = torch.cat(target_confidence_list[i])
                weight_for_this_category = weight_for_this_category / weight_for_this_category.sum()
                target_center_feature[i] = (torch.cat(target_feature_list_pseudo[i], dim=0) * weight_for_this_category).sum(0)
            else:
                target_center_feature[i] = (torch.cat(target_feature_list_pseudo[i], dim=0).mean(0))

        target_center_labels = target_center_labels[noempty_list]
        target_center_feature = target_center_feature[noempty_list]

        if len(noempty_list) != args.num_classes:
            print('noempty', noempty_list)
        if args.l2_process:
            target_center_feature = F.normalize(target_center_feature, dim=1, p=2)

        all_feature = torch.cat((target_center_feature, all_feature), dim=0)
        all_label = torch.cat((target_center_labels, all_label), dim=0)
        NumS = NumS + len(noempty_list)

    #### the A2LP algorithm is over now.
    corresponding_labels = []
    for i in range(LabelT.size(0)):
        corresponding_labels.append(LabelT[i][0].item())

    if args.filter_low:
        selected_samples = []
        for i in range(len(target_feature_list)):  ##### remove the samples with low confidence.
            if Instance_confidence[i] > cos_threshold:   ################## if the confidence if high, select it.
                selected_samples.append(i)
        clustering_label_for_path = {image_paths[i]: [corresponding_labels[i], Instance_confidence[i]] for i in selected_samples}
    else:
        clustering_label_for_path = {image_paths[i]: [corresponding_labels[i], Instance_confidence[i]] for i in range(len(corresponding_labels))}


    return clustering_label_for_path

def FeaturePreprocess(source_feature_list, source_feature_list_category, source_labels, target_feature_list, args, moving_feature_centeriod_t=False):

    if args.S4LP == 'all':   ## The full A2LP
        if type(moving_feature_centeriod_t) != bool:   ###########
            target_center_labels = torch.eye(args.num_classes)
            NumS = len(source_feature_list) + args.num_classes
            source_labels = torch.cat(source_labels, dim=0)
            source_labels = torch.cat((source_labels, target_center_labels), dim=0)
            target_feature = torch.cat(target_feature_list, dim=0)
            source_feature = torch.cat(source_feature_list, dim=0)
            source_feature = torch.cat((source_feature, moving_feature_centeriod_t.cpu()), dim=0)
            if args.l2_process:         ########################## l2 progress of target data
                target_feature = F.normalize(target_feature, dim=1, p=2)
                source_feature = F.normalize(source_feature, dim=1, p=2)
        else:
            NumS = len(source_feature_list)
            source_labels = torch.cat(source_labels, dim=0)
            target_feature = torch.cat(target_feature_list, dim=0)
            source_feature = torch.cat(source_feature_list, dim=0)
            if args.l2_process:         ########################## l2 progress of target data
                target_feature = F.normalize(target_feature, dim=1, p=2)
                source_feature = F.normalize(source_feature, dim=1, p=2)

    elif args.S4LP == 'cluster':  ## NOT reported variant ### adopt several representative points for each category in LP
        target_feature = torch.cat(target_feature_list, dim=0)
        if args.l2_process:
            target_feature = F.normalize(target_feature, dim=1, p=2)
        ############ generate args.NC4LP cluster for each source category and select satisfied center for LP.
        SelectedCenters = []
        SelectedLabels = []
        for i in range(args.num_classes):
            num_instance_one_cate = len(source_feature_list_category[i])
            source_feature_one_cate = torch.cat(source_feature_list_category[i], dim=0)
            if args.l2_process:
                source_feature_one_cate = F.normalize(source_feature_one_cate, dim=1, p=2)
            if args.spherical_kmeans:
                kmeans = SphericalKMeans(n_clusters=args.NC4LP, random_state=0, max_iter=args.niter).fit(source_feature_one_cate)
            else:
                kmeans = KMeans(n_clusters=args.NC4LP, random_state=0, max_iter=args.niter).fit(source_feature_one_cate)
            Ind = kmeans.labels_
            Ind = torch.from_numpy(Ind)
            Centers = kmeans.cluster_centers_
            Centers = torch.from_numpy(Centers).float()
            CountTensor = torch.zeros(args.NC4LP)
            for j in range(num_instance_one_cate):
                CountTensor[Ind[j]] += 1
            threshold = int(num_instance_one_cate/args.NC4LP)
            if threshold < 3:
                threshold = 3
            one_hot_label = torch.zeros(1, args.num_classes)
            one_hot_label[0][i] = 1
            for j in range(args.NC4LP):
                if CountTensor[j] > threshold:  ## this center is selected
                    SelectedCenters.append(Centers[j].view(1, Centers.size(1)))
                    SelectedLabels.append(one_hot_label)
        NumS = len(SelectedCenters)
        source_feature = torch.cat(SelectedCenters, dim=0)
        source_labels = torch.cat(SelectedLabels, dim=0)

    elif args.S4LP == 'center':  ## A2LP Variant in paper ## only adopt the category center for each category in LP
        NumS = args.num_classes
        source_labels = torch.eye(args.num_classes)
        target_feature = torch.cat(target_feature_list, dim=0)
        source_feature = torch.zeros(args.num_classes, args.fea_dim)
        for i in range(args.num_classes):
            source_feature[i] = torch.cat(source_feature_list_category[i], dim=0).mean(0)
        if args.l2_process:
            target_feature = F.normalize(target_feature, dim=1, p=2)
            source_feature = F.normalize(source_feature, dim=1, p=2)

    else:
        raise NotImplementedError

    return source_feature, source_labels, NumS, target_feature

def process_zero_value(tensor, nozero=True):
    if (tensor <= 0).sum() != 0:
        if nozero:
            tensor[tensor <= 0] = 1e-8
        else:
            tensor[tensor <= 0] = 0
    return tensor

def download_feature_and_pca_clustering(epoch, train_loader, val_loader, model, args):
    model.eval()
    image_paths = []
    GT_labels = []
    source_feature_list = []
    for i in range(args.num_classes):
        source_feature_list.append([])  ######### each for one categoty
    for i, (input, target, img_path) in enumerate(train_loader):
        print('soruce center calculation', i)
        input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            if args.dsbn:
                feature_source, _ = model(input_var, ST='S')
            else:
                feature_source, _ = model(input_var)
        feature_source = feature_source.cpu()
        batchsize = feature_source.size(0)
        for j in range(batchsize):
            img_label = target[j]
            source_feature_list[img_label].append(feature_source[j].view(1, feature_source.size(1)))

    target_feature_list = []
    for i, (input, target, img_path) in enumerate(val_loader):
        print('target feature calculation', i)
        input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            if args.dsbn:
                feature_target, _ = model(input_var, ST='T')
            else:
                feature_target, _ = model(input_var)
        batchsize = feature_target.size(0)
        feature_target = feature_target.cpu()
        for j in range(batchsize):
            GT_labels.append(target[j].item())
            image_paths.append(img_path[j])
            target_feature_list.append(feature_target[j].view(1, feature_target.size(1)))
    pca = False
    if pca:  ## features are PCA-reduced to 256 dimensions, whitened and l2-normalized.
        feature_matrix = torch.cat(target_feature_list, dim=0).numpy()
        category_index = []
        for i in range(args.num_classes):
            category_index.append(len(source_feature_list[i]))
            source_feature_list[i] = torch.cat(source_feature_list[i], dim=0)
        feature_matrix_source = torch.cat(source_feature_list, dim=0).numpy()
        print('apply PCA to the cnn features to reduce dimensions')
        pcamatrix_source = faiss.PCAMatrix(2048, args.pca_dim, eigen_power=args.eigen_power)  ## eigen_power = -2 indicates full whitening, = 0 (default) indicates no whitening
        pcamatrix_source.train(feature_matrix_source)
        assert pcamatrix_source.is_trained
        feature_matrix_source = pcamatrix_source.apply_py(feature_matrix_source)  ## the features after pca and whitening
        if args.l2_process:
            print('apply l2 normalization to the cnn features')
            norm_value_source = LA.norm(feature_matrix_source, axis=1)
            feature_matrix_source = feature_matrix_source / norm_value_source[:, None]
        source_center_list = []
        start_index = 0
        for i in range(args.num_classes):
            source_center_list.append(feature_matrix_source[start_index: start_index+category_index[i]].mean(0))
            start_index = start_index + category_index[i]
        source_feature_array = np.array(source_center_list)
        #####################################################################################################################
        print('apply PCA to the cnn features to reduce dimensions')
        pcamatrix = faiss.PCAMatrix(2048, args.pca_dim, eigen_power=args.eigen_power)  ## eigen_power = -2 indicates full whitening, = 0 (default) indicates no whitening
        pcamatrix.train(feature_matrix)
        assert pcamatrix.is_trained
        feature_matrix = pcamatrix.apply_py(feature_matrix)  ## the features after pca and whitening
        if args.l2_process:
            print('apply l2 normalization to the cnn features')
            norm_value = LA.norm(feature_matrix, axis=1)
            feature_matrix = feature_matrix / norm_value[:, None]
        if args.spherical_kmeans:
            kmeans = SphericalKMeans(n_clusters=args.num_classes, random_state=0, init=source_feature_array, max_iter=args.niter).fit(feature_matrix)
        else:
            kmeans = KMeans(n_clusters=args.num_classes, random_state=0, init=source_feature_array, max_iter=args.niter).fit(feature_matrix)
    else:
        ########################################### calculte source category center directly
        if args.l2_process:         ########################## l2 progress of source center
            feature_matrix = torch.cat(target_feature_list, dim=0)
            feature_matrix = F.normalize(feature_matrix, dim=1, p=2)
            feature_matrix = feature_matrix.numpy()
        else:
            feature_matrix = torch.cat(target_feature_list, dim=0).numpy()

        for i in range(args.num_classes):
            source_feature_list[i] = torch.cat(source_feature_list[i], dim=0)  ########## K * [num * dim]
            if args.l2_process:     ########################## l2 progress of target instance
                # source_feature_list[i] = F.normalize(source_feature_list[i], dim=1, p=2)  ################ comment this is very important.
                source_feature_list[i] = F.normalize(source_feature_list[i].mean(0), dim=0, p=2)
            else:
                source_feature_list[i] = source_feature_list[i].mean(0)
            source_feature_list[i] = source_feature_list[i].numpy()
        source_feature_array = np.array(source_feature_list)
        print('use the original cnn features to play cluster')
        if args.spherical_kmeans:
            kmeans = SphericalKMeans(n_clusters=args.num_classes, random_state=0, init=source_feature_array, max_iter=args.niter).fit(
                feature_matrix)
        else:
            kmeans = KMeans(n_clusters=args.num_classes, random_state=0, init=source_feature_array, max_iter=args.niter).fit(feature_matrix)

    Ind = kmeans.labels_
    print(Ind)
    print(GT_labels)
    gt_label_array = np.array(GT_labels)
    acc_count = torch.zeros(args.num_classes)
    all_count = torch.zeros(args.num_classes)
    for i in range(len(gt_label_array)):
        all_count[gt_label_array[i]] += 1
        if gt_label_array[i] == Ind[i]:
            acc_count[gt_label_array[i]] += 1

    acc_for_each_class1 = acc_count / all_count
    acc_cluster_label = sum(gt_label_array == Ind) / gt_label_array.shape[0]
    corresponding_labels = []
    for i in range(len(Ind)):
        corresponding_labels.append(Ind[i])

    if args.filter_low:
        ccenters = torch.from_numpy(kmeans.cluster_centers_)
        processed_features = torch.from_numpy(feature_matrix)
        selected_samples = []
        for i in range(len(target_feature_list)):  ##### remove the samples with low confidence.
            dis = 0.5 * (1 - (processed_features[i] * ccenters[corresponding_labels[i]]).sum())
            if dis < args.cos_threshold:
                selected_samples.append(i)

        clustering_label_for_path = {image_paths[i]: corresponding_labels[i] for i in selected_samples}
    else:
        clustering_label_for_path = {image_paths[i]: corresponding_labels[i] for i in range(len(corresponding_labels))}
    # NMI_value = NMI_calculation(GT_labels, corresponding_labels)
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    if args.category_mean:
        log.write("\nAcc for each class1: ")
        for i in range(12):
            if i == 0:
                log.write("%dst: %3f" % (i + 1, acc_for_each_class1[i]))
            elif i == 1:
                log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class1[i]))
            elif i == 2:
                log.write(", %drd: %3f" % (i + 1, acc_for_each_class1[i]))
            else:
                log.write(", %dth: %3f" % (i + 1, acc_for_each_class1[i]))
        log.write("Avg. over all classes: %3f" % acc_for_each_class1.mean())
    log.write("   Avg. over all sample: %3f" % acc_cluster_label)
    log.close()

    return clustering_label_for_path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_for_each_class(output, target, total_vector, correct_vector):
    """Computes the precision for each class"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[target[i]] += 1
        correct_vector[torch.LongTensor([target[i]])] += correct[i]

    return total_vector, correct_vector


def cg_solver(A, B, X0=None, rtol=1e-3, maxiter=None):
    n, m = B.shape
    if X0 is None:
        X0 = B
    if maxiter is None:
        maxiter = 2 * min(n, m)
    X_k = X0
    R_k = B - A.matmul(X_k)
    P_k = R_k
    stopping_matrix = torch.max(rtol * torch.abs(B), 1e-3 * torch.ones_like(B))
    for k in range(1, maxiter+1):
        fenzi = R_k.transpose(0,1).matmul(R_k).diag()
        fenmu = P_k.transpose(0,1).matmul(A).matmul(P_k).diag()
        #fenmu[fenmu == 0] = 1e-8
        alpha_k = fenzi / fenmu
        X_kp1 = X_k + alpha_k * P_k
        R_kp1 = R_k - (A.matmul(alpha_k * P_k))
        residual_norm = torch.abs(A.matmul(X_kp1) - B)
        if (residual_norm <= stopping_matrix).all():
            break
        #fenzi[fenzi ==0] = 1e-8
        beta_k = (R_kp1.transpose(0, 1).matmul(R_kp1) / (fenzi)).diag()
        P_kp1 = R_kp1 + beta_k * P_k

        P_k = P_kp1
        X_k = X_kp1
        R_k = R_kp1
    return X_kp1


