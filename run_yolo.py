from custom_yolo_trainer.custom_trainer import CustomSegmentationTrainer, CustomDetectionTrainer
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from parameters import *

import time
import os

# os.environ["OMP_NUM_THREADS"] = "4"

def get_current_time() -> str: 
    """
    Returns: 
        (str): time in YmdHMS format
    """
    current_time = time.localtime()
    return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

def create_dir(folder_name: str) -> None:
    """
    Creates the given directory if it does not exist
    Args: 
        folder_name (str): directory to create
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  

def train_yolo() -> None: 
    """
    Trains YOLO model. All hyperparameters are configured in parameters.py
    """
    print("\nStarting Training...")

    print(f"\nThis is dataset {f"./data/{DATASET}.yaml"}\n")

    args = dict(
                # General Hyperparameters
                model=f"{MODEL}.yaml", 
                data=DATASET, 
                epochs=EPOCH, 
                pretrained=PRETRAINED, 
                imgsz=IMAGE_SIZE, 
                single_cls=SINGLE_CLS, 
                close_mosaic=CLOSE_MOSAIC, 
                fraction=FRACTION,
                freeze=None,  
                lr0=INITIAL_LR, 
                lrf=FINAL_LR, 
                warmup_epochs=WARMUP_EPOCH, 
                cls=CLS, 
                box=BOX, 
                dfl=DFL, 
                seed=SEED, 
                batch=BATCH,
                amp=MIX_PRECISION, 
                multi_scale=MULTI_SCALE, 
                cos_lr=COS_LR,
                plots=PLOT,
                profile=PROFILE,
                project=f"{MODE}_{MODEL}_{get_current_time()}",
                name=f"{MODEL}_{DATASET}", 
                
                # Data Augmentation Hyperparameters
                hsv_h=HSV_H, 
                hsv_s=HSV_S, 
                hsv_v=HSV_V, 
                degrees=DEGREES,
                translate=TRANSLATE,
                scale=SCALE,
                flipud=FLIPUD, 
                fliplr=FLIPLR, 
                mosaic=MOSAIC, 
                shear=SHEAR, 
                perspective=PERSPECTIVE, 
                mixup=MIXUP, 
                cutmix=CUTMIX)
    
    if LOAD_AND_TRAIN: 
        print("\nLoading and Training...")
        args["model"] = BEST_MODEL_DIR_TRAIN
        args["resume"] = RESUME
    
    # trainer = CustomSegmentationTrainer(overrides=args)
    trainer = CustomDetectionTrainer(overrides=args)
    trainer.train()


#     p_args = dict(model="yolo12n-seg_converged/weights/best.pt",
#             data=f"data/data.yaml", 
#             device="cpu", 
#             verbose=True,
#             imgsz=160, 
#             save=False)
#             
#     YOLO_predictor = CustomSegmentationPredictor(overrides=p_args)
#     YOLO_predictor.setup_model(p_args["model"])
# 
#     import cv2
#     from torchvision import transforms
# 
#     x = cv2.imread("BraTS-SSA-00046-00071-t1c.png", cv2.IMREAD_UNCHANGED)
#     x = transforms.ToTensor()(x)
#     x = transforms.Resize(size=(160, 160))(x)
#     x = x.to("cpu")  
#     x = x.unsqueeze(0)
#     # x = x.repeat(155, 1, 1, 1)
# 
#     # YOLO_predictor(x)
#     import time 
#     start = time.perf_counter()
#     YOLO_predictor.model.model(x)
#     end = time.perf_counter()
#     print((end - start)*1000)

    # print(f"\nEnsuring the Model's input layer was changed: {trainer.setup_model()}")
    # print(f"\nFinish training, please check your directory for folder named 'train-....")

if __name__ == "__main__":
    train_yolo()
