import argparse


def get_options(parser=argparse.ArgumentParser()):
    # Method options
    parser.add_argument('--dataset', type=str, default='4CRE-V1', help='Dataset')
    parser.add_argument('--init_size', type=int, default=1000, help='Number of samples in the maximal class')
    parser.add_argument('--label_ratio', type=float, default=20, help='percentage of labeled data')
    parser.add_argument('--imb_ratio', type=int, default=20, help='Imbalance ratio')
    parser.add_argument('--imb_type', type=str, default='long', help='Long tailed or step imbalanced')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = get_options()
