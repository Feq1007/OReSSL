import argparse  
  
def get_options(parser=argparse.ArgumentParser()):  
    parser.add_argument('--lambda', type=float, default=1e-4,  
                        help='hyper parameter to control the radius of micro cluster')  

    opt = parser.parse_args()  
    return opt  
  
if __name__ == '__main__':  
    opt = get_options()