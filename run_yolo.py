# Local
from custom_yolo_trainer.custom_trainer import CustomSegmentationTrainer, CustomDetectionTrainer
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from parameters import *

# Internal
import time
import os

# External
import torch
from torch.profiler import profile, ProfilerActivity, record_function


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

def test_ultralytics_yolo(warm_ups: bool = True,
              batch: int = 128,
              iterations: int = 3) -> None:
    # Set environment variables to restrict other libraries to 1 thread
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Restrict PyTorch's intra-op parallelism to 1 thread
    torch.set_num_threads(1)

    # You can also check the current setting
    print(f"PyTorch using {torch.get_num_threads()} threads.")
    from ultralytics import YOLO
    dummy_data = torch.zeros(batch, 3, 160, 160)

    model = YOLO("yolo11s-seg.pt")
    model = model.model
    model.to("cpu")
    
    if warm_ups:
        for _ in range(2):
            model(dummy_data)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                for _ in range(iterations):
                    model(dummy_data)

        # Get the key averages
        key_avg = prof.key_averages()

        # Calculate total CPU time from all operations
        total_cpu_time = sum([item.self_cpu_time_total for item in key_avg])
        avg_time_per_iteration = total_cpu_time / iterations

        print(f"--- Results averaged over {iterations} iterations ---")
        print(f"Total CPU time: {total_cpu_time / 1e6:.2f}")
        print(f"Average time per iteration: {
              avg_time_per_iteration / 1e6:.2f}")
        print("\nPer-operation breakdown:")
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=10
        ))
        

def test_yolo(warm_ups: bool = True,
              batch: int = 128,
              iterations: int = 3) -> None:
    # Set environment variables to restrict other libraries to 1 thread
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Restrict PyTorch's intra-op parallelism to 1 thread
    torch.set_num_threads(1)

    # You can also check the current setting
    print(f"PyTorch using {torch.get_num_threads()} threads.")

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
        args["model"] = BEST_MODEL_DIR
        args["resume"] = RESUME

    YOLO_predictor = CustomSegmentationPredictor(overrides=args)
    YOLO_predictor.setup_model(args["model"])
    model = YOLO_predictor.model.model

    model.to("cpu")
    model.eval()


    if warm_ups:
        for _ in range(2):
            model(dummy_data)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                for _ in range(iterations):
                    model(dummy_data)

        # Get the key averages
        key_avg = prof.key_averages()

        # Calculate total CPU time from all operations
        total_cpu_time = sum([item.self_cpu_time_total for item in key_avg])
        avg_time_per_iteration = total_cpu_time / iterations

        print(f"--- Results averaged over {iterations} iterations ---")
        print(f"Total CPU time: {total_cpu_time / 1e6:.2f}")
        print(f"Average time per iteration: {
              avg_time_per_iteration / 1e6:.2f}")
        print("\nPer-operation breakdown:")
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=10
        ))


def train_yolo() -> None:
    """
    Trains YOLO model. All hyperparameters are configured in parameters.py
    """

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
        args["model"] = BEST_MODEL_DIR
        args["resume"] = RESUME

    trainer = CustomSegmentationTrainer(overrides=args)
    # trainer = CustomDetectionTrainer(overrides=args)
    trainer.train()


if __name__ == "__main__":
    train_yolo()
    # test_yolo()
    # test_ultralytics_yolo()
