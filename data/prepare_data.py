import os
import shutil
import torch
import scipy.io as scio
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.folder_new import ImageFolder_new
from data.Uniform_folder import ImageFolder_uniform
from data.Uniform_sampler import UniformBatchSampler
import numpy as np
import cv2

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    # weight_per_class[-1] = weight_per_class[-1]  ########### adjust the cate-weight for unknown category.
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def _random_affine_augmentation(x):
    M = np.float32([[1 + np.random.normal(0.0, 0.1), np.random.normal(0.0, 0.1), 0],
        [np.random.normal(0.0, 0.1), 1 + np.random.normal(0.0, 0.1), 0]])
    rows, cols = x.shape[1:3]
    dst = cv2.warpAffine(np.transpose(x.numpy(), [1, 2, 0]), M, (cols,rows))
    dst = np.transpose(dst, [2, 0, 1])
    return torch.from_numpy(dst)


def _gaussian_blur(x, sigma=0.1):
    ksize = int(sigma + 0.5) * 8 + 1
    dst = cv2.GaussianBlur(x.numpy(), (ksize, ksize), sigma)
    return torch.from_numpy(dst)



def generate_dataloader(args):
    # Data loading code
    traindir_source = os.path.join(args.data_path_source, args.src)
    traindir_target = os.path.join(args.data_path_source_t, args.src_t)
    valdir = os.path.join(args.data_path_target, args.tar)
    if not os.path.isdir(traindir_source):
        # split_train_test_images(args.data_path)
        raise ValueError('Null path of source train data!!!')

    # normalize_s = transforms.Normalize(mean=[0.9094, 0.9077, 0.9047],
    #                                  std=[0.1977, 0.2013, 0.2081])

    # normalize_s = transforms.Normalize(mean=[0.459, 0.459, 0.459],    ############# should be all the same ...
    #                                    std=[0.226, 0.226, 0.226])

    normalize_s = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    normalize_gray = transforms.Normalize(mean=[0.459, 0.459, 0.459],    ############# should be all the same ...
                                       std=[0.226, 0.226, 0.226])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #################### random sampled source and target dataset for cross-entropy training
    if args.img_process_s == 'ours':
        source_train_dataset = ImageFolder_uniform(
            traindir_source,
            transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_s,
            ])
        )

    elif args.img_process_s == 'longs':
        ####################### below is long's preprocess
        source_train_dataset = ImageFolder_uniform(
            traindir_source,
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_s,
            ])
        )

    elif args.img_process_s == 'simple':
        source_train_dataset = ImageFolder_uniform(
            traindir_source,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize_s,
            ])
        )

    source_train_loader_ce = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.workers, pin_memory=True
    )

    # uniformbatchsampler = UniformBatchSampler(args.per_category, source_train_dataset.category_index_list,
    #                                           source_train_dataset.imgs)   ##### select images in the iteration process
    # source_train_loader_cas = torch.utils.data.DataLoader(
    #     source_train_dataset, num_workers=args.workers, pin_memory=True, batch_sampler=uniformbatchsampler
    # )


    if args.img_process_t == 'ours':
        target_train_dataset_ce = ImageFolder_new(
            traindir_target,
            transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_s,
            ]),
            transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: _random_affine_augmentation(x)),
                transforms.Lambda(lambda x: _gaussian_blur(x)),
                normalize_s,
            ])
        )

    elif args.img_process_t == 'longs':
        ####################### long's preprocess
        target_train_dataset_ce = ImageFolder_new(
            traindir_target,
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_s,
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: _random_affine_augmentation(x)),
                transforms.Lambda(lambda x: _gaussian_blur(x)),
                normalize_s,
            ])
        )

    elif args.img_process_t == 'simple':
        target_train_dataset_ce = ImageFolder_new(
            traindir_target,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize_s,
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: _random_affine_augmentation(x)),
                transforms.Lambda(lambda x: _gaussian_blur(x)),
                normalize_s,
            ])
        )

    target_train_loader_ce = torch.utils.data.DataLoader(
        target_train_dataset_ce, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.workers, pin_memory=True
    )

    if args.img_process_s == 'ours':
        target_train_dataset = ImageFolder_uniform(
            traindir_target,
            transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_s,
            ])
        )

    elif args.img_process_s == 'longs':
        ####################### long's preprocess
        target_train_dataset = ImageFolder_uniform(
            traindir_target,
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_s,
            ])
        )

    elif args.img_process_s == 'simple':
        target_train_dataset = ImageFolder_uniform(
            traindir_target,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize_s,
            ])
        )



    if args.img_process_s == 'ours':
        source_val_dataset = ImageFolder_uniform(
            traindir_source,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize_s,
            ])
        )
    elif args.img_process_s == 'longs':
        source_val_dataset = ImageFolder_uniform(
            traindir_source,
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize_s,
            ])
        )
    elif args.img_process_s == 'simple':
        source_val_dataset = ImageFolder_uniform(
            traindir_source,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize_s,
            ])
        )


    source_val_loader = torch.utils.data.DataLoader(
        source_val_dataset, batch_size=500, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )


    if args.img_process_t == 'ours':
        target_test_dataset = ImageFolder_uniform(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.img_process_t == 'longs':
        target_test_dataset = ImageFolder_uniform(valdir, transforms.Compose([
            transforms.Resize(256, 256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.img_process_t == 'simple':
        target_test_dataset = ImageFolder_uniform(valdir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    target_val_loader = torch.utils.data.DataLoader(
        target_test_dataset,
        batch_size=500, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )


    return source_train_loader_ce, source_train_dataset, target_train_loader_ce, target_train_dataset, source_val_loader, target_val_loader

