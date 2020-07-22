import argparse


def opts():
    parser = argparse.ArgumentParser(description='Train alexnet on the cub200 dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path_source', type=str, default='',
                        help='Root of train data set of the source domain')
    parser.add_argument('--data_path_source_t', type=str, default='',
                        help='Root of train data set of the target domain')
    parser.add_argument('--data_path_target', type=str, default='',
                        help='Root of the test data set')
    parser.add_argument('--src', type=str, default='amazon',
                        help='choose between amazon | dslr | webcam')
    parser.add_argument('--src_t', type=str, default='webcam',
                        help='choose between amazon | dslr | webcam')
    parser.add_argument('--tar', type=str, default='webcam',
                        help='choose between amazon | dslr | webcam')
    parser.add_argument('--num_classes', type=int, default=31,
                        help='number of classes of data used to fine-tune the pre-trained model')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of the source data.')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01, help='The Learning Rate.')
    parser.add_argument('--lrw', type=float, default=1.0, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=str, default='rev', help='rev | constant')
    parser.add_argument('--gamma', type=float, default=0.75, help='2.25 (visda) and 0.75 (others).')
    # checkpoints
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', type=str, default='', help='Checkpoints path to resume(default none)')
    parser.add_argument('--pretrained_checkpoint', type=str, default='', help='Pretrained checkpoint to resume (default none)')
    parser.add_argument('--test_only', '-t', action='store_true', help='Test only flag')
    #### graph
    parser.add_argument('--dis_gra', type=str, default='l2', help='dis for graph')
    parser.add_argument('--cor', type=float, default=1.0, help='cor in the computation of l2 distance')
    parser.add_argument('--TopkGraph', action='store_true', help='full graph 2 topk graph')
    parser.add_argument('--graphk', type=int, default=10, help='KNN grapg')
    parser.add_argument('--AlphaGraph', type=float, default=0.5, help='level for propagation.')


    parser.add_argument('--noise_level', type=float, default=0.1, help='cor in the computation of l2 distance')
    parser.add_argument('--noise_flag', action='store_true', help='full graph 2 topk graph')
    # Architecture
    parser.add_argument('--arch', type=str, default='resnet101', help='Model name')
    parser.add_argument('--img_process_t', type=str, default='simple', help='Model name')
    parser.add_argument('--img_process_s', type=str, default='simple', help='Model name')
    parser.add_argument('--flag', type=str, default='original', help='flag for different settings')
    parser.add_argument('--type', type=str, default='type1', help='type1 | type2 | type3')
    parser.add_argument('--dis', type=str, default='cross_entropy', help='cross_entropy | kl | l1')
    parser.add_argument('--pretrained', action='store_true', help='whether using pretrained model')
    parser.add_argument('--per_category', type=int, default=4, help='number of domains')
    parser.add_argument('--fea_dim', type=int, default=2048, help='feature dim')
    parser.add_argument('--uniform_type_s', type=str, default='soft', help='hard | soft | none')
    parser.add_argument('--uniform_type_t', type=str, default='soft', help='hard | soft | none')
    parser.add_argument('--dsbn', action='store_true', help='whether use domain specific bn')
    parser.add_argument('--fixbn', action='store_true', help='whether fix the ImageNet pretrained BN layer')
    parser.add_argument('--OurMec', action='store_true', help='whether use our cross entropy style MEC | original mec')
    parser.add_argument('--OurPseudo', action='store_true', help='whether use cluster label for cross entropy directly | tangs')
    parser.add_argument('--category_mean', action='store_true', help='Only True for visda, acc calculated over categories')
    parser.add_argument('--clufrq_dec', action='store_true', help='whether decrease the cluster freq.')
    parser.add_argument('--threed', action='store_true', help='ori + aug + grey | ori + grey.')
    parser.add_argument('--only_lrw', action='store_true', help='lrw weight | lamda')
    parser.add_argument('--niter', type=int, default=500, help='iteration of clustering')

    parser.add_argument('--pseudo_type', type=str, default='cluster', help='cluster (spherical_kmeans cluster) or lp (label propagation)')
    parser.add_argument('--l2_process', action='store_true', help='')
    parser.add_argument('--spherical_kmeans', action='store_true', help='')
    parser.add_argument('--entropy_weight', action='store_true', help='whether adopt the prediction entropy of LP prediction as weight')


    parser.add_argument('--S4LP', type=str, default='all', help='all | cluster | center')
    parser.add_argument('--LPSolver', type=str, default='Itera', help='Itera | CloseF')
    parser.add_argument('--LPType', type=str, default='lgc', help='lgc | hmn | parw | omni')
    parser.add_argument('--alpha', type=float, default=0.99, help='hyper-parameter.')
    parser.add_argument('--lamb', type=float, default=1.0, help='hyper-parameter')
    parser.add_argument('--NC4LP', type=int, default=3, help='number of clusters for each category in clustering')
    parser.add_argument('--LPIterNum', type=int, default=15, help='number of clusters for each category in clustering')
    parser.add_argument('--LPIterationType', type=str, default='add', help='replace | add')

    parser.add_argument('--min_num_cate', type=int, default=3, help='lowest number of image in each class')
    parser.add_argument('--filter_low', action='store_true', help='filter the samples with low prediction confidence')
    parser.add_argument('--cos_threshold', type=float, default=0.05, help='hyper-parameter.')
    parser.add_argument('--weight_type', type=str, default='cas_ins', help='replace | add')
    parser.add_argument('--graph_gama', type=int, default=1, help='for graph construction, follow manifold-based search')
    parser.add_argument('--dis_margin', type=float, default=1.0, help='hyper-parameter.')
    parser.add_argument('--moving_weight', type=float, default=0.7, help='hyper-parameter.')


    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints', help='Log folder')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--test_freq', default=10, type=int,
                        help='test frequency (default: 1)')
    parser.add_argument('--cluster_freq', default=1, type=int,
                        help='clustering frequency (default: 1)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--score_frep', default=300, type=int,
                        metavar='N', help='print frequency (default: 300, not download score)')
    args = parser.parse_args()

    args.data_path_source_t = args.data_path_source
    args.data_path_target = args.data_path_source
    args.src_t = args.tar


    args.log = args.log + '_' + args.src + '2' + args.tar + '_' + args.arch + '_' + args.flag + '_' + args.type + '_' + \
               args.dis + '_' + args.uniform_type_s + '_'  + args.pseudo_type + str(args.lrw) + '_' + str(args.cos_threshold) + args.dis_gra


    return args
