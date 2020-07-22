#!/bin/bash 186

########################################################
## The original algorithm, constructing the graph with brute-force implementation, and solve the LP with closed-form solver. (low speed)
###########################################################

#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/Office31/  --src amazon  --tar dslr --pretrained --epochs 4000  --num_classes 31 --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/off31 --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add

#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/Office31/  --src webcam  --tar amazon --pretrained --epochs 4000  --num_classes 31 --print_freq 1 --test_freq 23 --per_category 4 \
#            --cluster_freq 230  --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/off31 --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6   --filter_low   --cos_threshold 0.5 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/Office31/  --src dslr  --tar amazon --pretrained --epochs 4000  --num_classes 31 --print_freq 1 --test_freq 23 --per_category 4 \
#            --cluster_freq 230  --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/off31 --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.5 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/Office31/  --src amazon  --tar webcam --pretrained --epochs 4000  --num_classes 31 --print_freq 1 --test_freq 23 --per_category 4 \
           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.4 --pseudo_type lp --spherical_kmeans \
            --entropy_weight  --schedule rev --pretrained   --log test/off31 --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
            --graph_gama 1  --LPIterNum 15 --filter_low   --cos_threshold 0.6 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP center --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add


#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src p --tar c --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 15  --filter_low   --cos_threshold 0.7 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add

#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src p --tar i --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src i --tar c --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src i --tar p --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src c --tar p --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add

#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src c --tar i --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add




########################################################
## The original algorithm, constructing the graph with nndescent, and solve the LP with CG solver. (fast)
###########################################################

###########################################################
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/Office31/  --src amazon  --tar dslr --pretrained --epochs 4000  --num_classes 31 --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/off31 --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CG --LPType lgc --niter 50 --LPIterationType add
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/Office31/  --src webcam  --tar amazon --pretrained --epochs 4000  --num_classes 31 --print_freq 1 --test_freq 23 --per_category 4 \
#            --cluster_freq 230  --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/off31 --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6   --filter_low   --cos_threshold 0.5 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CG --LPType lgc --niter 50 --LPIterationType add

#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/Office31/  --src dslr  --tar amazon --pretrained --epochs 4000  --num_classes 31 --print_freq 1 --test_freq 23 --per_category 4 \
#            --cluster_freq 230  --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/off31 --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.5 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CG --LPType lgc --niter 50 --LPIterationType add

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/Office31/  --src amazon  --tar webcam --pretrained --epochs 4000  --num_classes 31 --print_freq 1 --test_freq 23 --per_category 4 \
           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.4 --pseudo_type lp --spherical_kmeans \
            --entropy_weight  --schedule rev --pretrained   --log test/off31 --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
            --graph_gama 1  --LPIterNum 15 --filter_low   --cos_threshold 0.6 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP center --LPSolver CG --LPType lgc --niter 50 --LPIterationType add



CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src p --tar c --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
            --graph_gama 1  --LPIterNum 15  --filter_low   --cos_threshold 0.7 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CG --LPType lgc --niter 50 --LPIterationType add

#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src p --tar i --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CG --LPType lgc --niter 50 --LPIterationType add
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src i --tar c --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CG --LPType lgc --niter 50 --LPIterationType add
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src i --tar p --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CG --LPType lgc --niter 50 --LPIterationType add
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src c --tar p --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CG --LPType lgc --niter 50 --LPIterationType add
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/image_CLEF/  --src c --tar i --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
#           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
#            --entropy_weight  --schedule rev --pretrained   --log test/clef --type type1 --dis cross_entropy --img_process_t ours --img_process_s ours \
#            --graph_gama 1  --LPIterNum 6  --filter_low   --cos_threshold 0.7 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.5  --graphk 20 --S4LP all --LPSolver CG --LPType lgc --niter 50 --LPIterationType add


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/visDA/  --src train --tar validation --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
            --entropy_weight  --schedule rev --pretrained   --log test/visda --type type1 --dis cross_entropy --img_process_t simple --img_process_s simple \
            --graph_gama 1  --LPIterNum 8  --filter_low   --cos_threshold 0.7 --dis_gra nndescent --l2_process  --TopkGraph --AlphaGraph 0.75  --graphk 100 --S4LP center --LPSolver CG --LPType lgc --niter 50 --LPIterationType add


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --data_path_source /data1/domain_adaptation/visDA/  --src train --tar validation --pretrained --epochs 400  --num_classes 12  --print_freq 1 --test_freq 23 --per_category 4 \
           --arch resnet50 --lr 0.01 --gamma 0.75 --weight_decay 1e-4 --workers 4   --lrw 0.3 --pseudo_type lp --spherical_kmeans \
            --entropy_weight  --schedule rev --pretrained   --log test/visda --type type1 --dis cross_entropy --img_process_t simple --img_process_s simple \
            --graph_gama 1  --LPIterNum 8  --filter_low   --cos_threshold 0.7 --dis_gra mul --l2_process  --TopkGraph --AlphaGraph 0.75  --graphk 100 --S4LP center --LPSolver CloseF --LPType lgc --niter 50 --LPIterationType add
