import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor

def create_dir(folder_name: str) -> None:
    """
    Creates given directory if it does not exist
    Args: 
        folder_name (str): directory to create
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name) 

def reconstruct_masks(data_path: str, split: str, root_dest_dir: str) -> None: 
    """
    Using a pretrained YOLOv12-Seg model, it performs inference on the dataset and then saves the masks. The resulting masks will be IMAGE_SIZE x IMAGE_SIZE mask (e.g., 160x160), which might or might not align with the original dataset. We will resize the labels when evaluating the model with evaluate_ensemble.py
    Args: 
        data_path (str): directory for the dataset
        split (str): the split to reconstruct from
        root_dest_dir (str): destination directory
    FUTURE TODO: Optimize with batch inference and threadpoolexecutor for I/O tasks
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Obtain image paths
    images = sorted([os.path.join(data_path, "images", split, i) for i in os.listdir(os.path.join(data_path, "images", split))])

    # Configure YOLO inference
    args = dict(conf=CONFIDENCE, save=False, device="cuda", imgsz=IMAGE_SIZE, batch=1, verbose=False)  
    predictor = CustomSegmentationPredictor(overrides=args)
    predictor.setup_model(MODEL_PATH)

    # Create output directory
    dest_dir = os.path.join(root_dest_dir, split) ; create_dir(dest_dir)

    for idx, image_path in enumerate(tqdm(images[:])):
        results = predictor(image_path)
        for result in results:                      # <- iterate for each result (in this case, only one because batch is 1)
            if result.masks:                        # <- if there is a mask prediction
                cumulative = np.zeros(IMAGE_SIZE)   # <- initialize the mask accumulator
                masks = result.masks                # <- list of masks (there can be more than one mask in masks)
                for mask in masks:                  # <- for each detected object in the masks
                    mask_binary = (mask.data.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
                    cumulative = np.maximum(mask_binary, cumulative) # <- convert to numpy and apply obtain the maximum of each pixels  
                                                                                            #    basically, accumulate all the 1's from the masks                
                cumulative = (cumulative * 255).astype(np.uint8)
                cv2.imwrite(os.path.join( dest_dir, os.path.basename(result.path) ), cumulative)

if __name__ == "__main__": 
    DATA_PATH = "data/stacked_segmentation"
    SPLIT = "val"
    MODEL_PATH = "yolo_checkpoint/weights/best.pt"
    DEST_DIR = f"reconstructed_{SPLIT}/labels"
    
    IMAGE_SIZE = 160
    CONFIDENCE = 0.7
    
    reconstruct_masks(DATA_PATH, SPLIT, DEST_DIR)
