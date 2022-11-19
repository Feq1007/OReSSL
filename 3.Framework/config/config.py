import argparse  
  
def get_options(parser=argparse.ArgumentParser()):  
    # Optimization options
    parser.add_argument('--workers', type=int, default=0,  
                        help='number of data loading workers, you had better put it '  
                              '4 times of your gpu')  
  
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size, default=64')  
  
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for, default=10')  
  
    parser.add_argument('--lr', type=float, default=3e-5, help='select the learning rate, default=1e-3')  
  
    parser.add_argument('--seed', type=int, default=118, help="random seed")  
  
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')

    parser.add_argument('--train_eval_ratio', type=float, default=0.8, help='weight decay for optimizer')
    
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')

    parser.add_argument('--rate', type=float, default=0.1, help='start epoch')

    # Device options
    parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')

    # Method options
    parser.add_argument('--init_size', type=int, default=1000, help='Number of samples in the maximal class')
    parser.add_argument('--label_ratio', type=float, default=20, help='percentage of labeled data')
    parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio')
    parser.add_argument('--val-iteration', type=int, default=500, help='Frequency for the evaluation')
    
    #dataset and imbalanced type
    parser.add_argument('--dataset', type=str, default='4CRE-V1', help='Dataset')
    parser.add_argument('--imb_type', type=str, default='long', help='Long tailed or step imbalanced')
    
    parser.add_argument('--checkpoint_path',type=str,default='',  
                        help='Path to load a previous trained model if not empty (default empty)')  

    # online learning setting
    parser.add_argument('--lmbda', type=float, default=1e-4, help='lambda: to control the radius of micro-clsuter')
    parser.add_argument('--minRE', type=float, default=0.8, help='a threshold of the reliability')
    parser.add_argument('--maxMC', type=int, default=700, help='maximum number of labeled micro-cluster')
    parser.add_argument('--maxUMC', type=int, default=200, help='maximum number of unlabeled micro-cluster')
    parser.add_argument('--k', type=int, default=3, help='k of classifier knn')
    parser.add_argument('--init_k_per_class', type=int, default=30, help='k for initial micro-clusters')
    parser.add_argument('--propagate', type=bool, default=False, help='do propagate or not')
  
    opt = parser.parse_args()
  
    return opt  
  
if __name__ == '__main__':  
    opt = get_options()