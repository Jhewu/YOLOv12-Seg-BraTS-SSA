from typing import List
import subprocess 
import argparse
import time
import os

def reformat_parameter_dirs(param_dirs: List[str]) -> List[str]:
    """
    Args: 
        param_dirs (List[str]): List of parameter directory paths
    Returns: 
        param_dirs (List[str]): List of reformatted parameter directory paths in the format "3_fold_run.parameters_0"
    """
    return [dir.replace("/", ".").rstrip(".py") for dir in param_dirs]
    
def run_kfold_cross_validation(): 
    """
    Runs run_yolo.py K times with different parameter directories for K-Fold Cross Validation.

    Args: 
        PARAM_DIRS (GLOBAL List[str]): List of parameter directory paths in the format "3_fold_run.parameters_0"
        K (GLOBAL int): K parameter in K-Fold Cross Validation    
    """

    command = ["python3", "run_yolo.py", "--train_or_evaluate", "1", "--detection_or_segmentation", "1"]

    # Check if directories exist before running the command
    for i in range(K): 
        parameter_dir = PARAM_DIRS[i]
        if os.path.exists(parameter_dir): 
            print("Directory exists: ", parameter_dir)
        else: 
            print("Directory does not exist: ", parameter_dir)
            return

    for i in range(K): 
        print(f"Running fold {i+1}/{K}...")

        # Expand the command to include the parameter directory
        parameter_dir = reformat_parameter_dirs([PARAM_DIRS[i]])[0]
        command.extend(["--param_dir", f"{parameter_dir}"])
        result = subprocess.run(command, capture_output=True, text=True)

        # Print the result
        print("Output: ", result.stdout) # Output from the script
        print("\nError: ", result.stderr)  # Error from the script

        # Buffer time
        time.sleep(5)
        print(f"\nFinished fold {i+1}/{K}...")
        print("Waiting for 5 seconds before starting the next fold...\n")
        
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
    else: PARAM_DIRS = ["3_fold_run/parameters_0.py", "3_fold_run/parameters_1.py", "3_fold_run/parameters_2.py"]

    run_kfold_cross_validation()
    print(f"\nFinished K-Fold Cross Validation! Check Directory for {K} directories of trained models and results.")
