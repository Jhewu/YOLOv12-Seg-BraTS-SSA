import os
import cv2
import torch 
import numpy as np
from tqdm import tqdm

from monai.metrics import DiceMetric, HausdorffDistanceMetric

def evaluate_ensemble(pred_dir: str, label_dir: str, image_size: int) -> None:
    """
    Calculate the mean Dice score (with monai) with the two given directories
    Args: 
        pred_dir (str): directory containing the predict images (reconstructed with reconstruct_masks.py)
        label_dir (str): directory containing the original masks (not modified)
        image_size (int): use to create empty "label" if no predictions
    """

    # Get label paths
    label_paths = sorted([os.path.join(label_dir, i) for i in os.listdir(label_dir)])

    # total_dice = 0
    num_samples = len(label_paths)

    # Declare monai metrics (correctly)
    metric = DiceMetric(
            include_background = False, # exclude background when reporting Dice (standard practice)
            reduction="mean_batch",     
            get_not_nans = False, 
            ignore_empty = False, 
            num_classes = None,         # infers from data (will be 1 channel)
            return_with_label = False
        )

    hd95 = HausdorffDistanceMetric(
            include_background = False, 
            percentile=95, 
            reduction = "none",
            get_not_nans = True, 
        )
         
    metric.reset() # (not needed, but best practice)
    hd95.reset()
    total_TP = total_FP = total_FN = 0
    for i, label_path in enumerate(tqdm(label_paths)):
        label_name = os.path.basename(label_path)
        pred_path = os.path.join(pred_dir, label_name)

        pred = None
        # Load prediction
        if os.path.exists(pred_path):
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert to tensor and NORMALIZE to [0, 1] range
            pred = torch.from_numpy(pred).float() / 255.0
            
            # Add channel and batch dimension
            pred = pred.unsqueeze(0).unsqueeze(0)
            
        else:
            # Create empty mask if prediction doesn't exist
            pred = torch.zeros(1, 1, image_size, image_size) # Use correct size

        # Load label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (image_size, image_size), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor and NORMALIZE to [0, 1] range
        label = torch.from_numpy(label).float() / 255.0
        
        # Add channel and batch dimension
        label = label.unsqueeze(0).unsqueeze(0)
        
        # Update metrics (sigmoid -> binarization -> metric)
        pred_binary = (pred > 0.5).float()
        label_binary = (label > 0.5).float() # technically, it's not needed as they are already binarized when saving
        metric(pred_binary, label_binary)

        # Calculate HD95
        mask = label
        pred_hot_encoded = torch.cat([1 - pred_binary, pred_binary], dim=1)
        mask_hot_encoded = torch.cat([1 - mask, mask], dim=1)                    
        hd95(pred_hot_encoded, mask_hot_encoded)

         # Calculate Precision and Recall
        TP = (pred_binary * mask).sum().float()
        FP = (pred_binary * (1 - mask)).sum().float()
        FN = ((1 - pred_binary) * mask).sum().float()
        total_TP += TP.item()
        total_FP += FP.item()
        total_FN += FN.item()

    mean_dice = metric.aggregate().item()
    val_precision_metric = total_TP / (total_TP + total_FP + 1e-6)
    val_recall_metric = total_TP / (total_TP + total_FN + 1e-6)
    hd95_raw_results, hd95_not_nans_count = hd95.aggregate()
    is_valid = hd95_not_nans_count.bool()
    successful_hd95_values = hd95_raw_results[is_valid]
    val_hd95_metric = torch.mean(successful_hd95_values).item()
    
    print(f"\nThe mean dice score is {mean_dice}")
    print(f"\nThe mean hd95 score is {val_hd95_metric}")
    print(f"\nThe mean precision score is {val_precision_metric}")
    print(f"\nThe mean recall score is {val_recall_metric}")

if __name__ == "__main__":
    SPLIT = "test"
    PRED_PATH = f"reconstructed_{SPLIT}/labels"
    LABEL_PATH = "data/stacked_segmentation/masks"
    IMAGE_SIZE = 160
    
    evaluate_ensemble(os.path.join(PRED_PATH, SPLIT), os.path.join(LABEL_PATH, SPLIT), IMAGE_SIZE)
