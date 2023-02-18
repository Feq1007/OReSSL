import argparse
from river.datasets import synth


def get_options(parser=argparse.ArgumentParser()):
    # dataset and imbalanced type
    parser.add_argument('--dataset', type=str, default='hyperplane', help='Dataset')
    parser.add_argument('--datatype', type=str, default='artificial', help='The dataset is synthetic or not')

    # Method options
    parser.add_argument('--init_size', type=int, default=800, help='Number of initialization samples')
    parser.add_argument('--eval_size', type=int, default=20000, help='Number of samples in evaluation data')
    parser.add_argument('--label_ratio', type=float, default=20, help='percentage of labeled data')
    parser.add_argument('--imb_type', type=str, default='long', help='Long tailed or step imbalanced')
    parser.add_argument('--imb_ratio', type=int, default=10, help='Imbalance ratio')

    parser.add_argument('--val_iteration', type=int, default=200, help='Frequency for the evaluation')
    parser.add_argument('--eval_times', type=int, default=4, help='eval times to save the model with highest accuracy')

    # Optimization options
    parser.add_argument('--train_eval_ratio', type=float, default=0.8, help='weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size, default=64')
    parser.add_argument('--workers', type=int, default=2,
                        help='number of data loading workers, you had better put it 4 times of your gpu')

    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for, default=10')

    parser.add_argument('--lr', type=float, default=0.1, help='select the learning rate, default=1e-3')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=118, help="random seed")
    parser.add_argument('--adjust_epoch', type=int, default=20, help='the number of epoch to adjust the learning rate')
    parser.add_argument('--rate', type=float, default=0.1, help='adjust the learning rate')

    # Hyperparameters for MixMatch
    parser.add_argument('--mix_alpha', default=0.75, type=float)
    parser.add_argument('--lambda_u', default=0.5, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--w_ent', default=1.5, type=float)
    parser.add_argument('--align', type=bool, default=True, help='Distribution alignment term')
    parser.add_argument('--align_length', type=int, default=10, help='Distribution alignment history length')

    # Device options
    parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Path to load a previous trained model if not empty (default empty)')

    # online learning setting
    parser.add_argument('--lmbda', type=float, default=1e-4, help='lambda: to control the radius of micro-clsuter')
    parser.add_argument('--minRE', type=float, default=0.8, help='a threshold of the reliability')
    parser.add_argument('--maxMC', type=int, default=700, help='maximum number of labeled micro-cluster')
    parser.add_argument('--maxUMC', type=int, default=200, help='maximum number of unlabeled micro-cluster')
    parser.add_argument('--k', type=int, default=3, help='k of classifier knn')
    parser.add_argument('--init_k_per_class', type=int, default=30, help='k for initial micro-clusters')
    parser.add_argument('--propagate', type=bool, default=False, help='do propagate or not')
    parser.add_argument('--imblearn', type=bool, default=True, help="apply the online imbalance learning")

    opt = parser.parse_args()

    return opt

def get_features_classes(dataset):
    # return (class_features, class_num, layers)
    if dataset == 'Shuttle':
        return 9, 7, [[9, 16], [16, 8], [8, 7], [8, 8], [8, 7]]
    if dataset == '4CRE-V1':
        return 2, 4, [[2, 10], [10, 2], [2, 4], [2, 2], [2, 4]]
    if dataset == '5CVT':
        return 2, 5, [[2,4,8],[8, 4],[4,5],[4,4],[4,5]]
    if dataset == 'airlines':
        return 7, 2, [[7, 16], [16, 8],[8, 2], [8, 4], [4, 2]]
    if dataset == 'spam':
        return 500, 2, [[500, 64, 32],[32, 16], [16, 2],[16,8],[8,2]]
    if dataset == 'gas':
        return 129, 6, [[129, 128, 64], [64, 32], [32, 6], [32, 32], [32, 6]]
    if dataset == 'covtypeNorm':
        return 55, 7, [[55, 128], [128, 64, 32], [32, 7], [32, 32], [32,7]]


def get_proper_parameters(dataname):
    if dataname == "elecNormNew":
        return {
            'knn': 20,
            'epsilon': 5e-4,
            're_threshold': 0.8
        }
    elif dataname == 'amazon-employee':
        return {
            'knn': 20,
            'epsilon': 1e-3,
            're_threshold': 0.9
        }
    else:
        return {
            'knn': 20,
            'epsilon': 1e-4,
            're_threshold': 0.8
        }

def get_artificial_dataset(dataname):
    if dataname == "hyperplane":
        return synth.Hyperplane()

def get_imb_distribution(dataname):
    if dataname == 'hyperplane':
        return {0:.9, 1:.1}


if __name__ == '__main__':
    opt = get_options()

