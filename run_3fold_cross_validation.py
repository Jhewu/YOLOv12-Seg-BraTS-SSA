import subprocess 
import argparse

def run_kfold_cross_validation(): 
    for i in range(K): 
        print(K)

if __name__ == "__main__": 
    # -------------------------------------------------------------
    des="""
    Run K-fold cross validation for YOLO Ultralytics by calling run_yolo.py
    K times
    """
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", "--param_dirs", nargs="+", type=str,help='directories of parameters.py containing YOLO Ultralytics hyperparameters for each fold. \t[3_fold_dataset/parameters_1.py, 3_fold_dataset/parameters_2.py, 3_fold_dataset/parameters_3.py]')
    parser.add_argument("-k", "--k", type=int,help='K parameter in K-Fold Cross Validation. Default is 3\t[3]')
    args = parser.parse_args()

    if args.k is not None:
        K = args.k
    else: K = 3
    if args.param_dirs is not None:
        PARAM_DIRS = args.param_dirs    
    else: PARAM_DIRS = ["3_fold_dataset/parameters_1.py", "3_fold_dataset/parameters_2.py", "3_fold_dataset/parameters_3.py"]

    print(args.param_dirs)  

    # run_kfold_cross_validation(K)
