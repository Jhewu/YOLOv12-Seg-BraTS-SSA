# YOLOv12-Seg-BraTS-SSA
Brain Tumor Segmentation with YOLOv12-Seg (n) on BraTS SSA 

# Directory Description
- archive/ : archive of probably useless files and directories
- 2026/ : 2026 archive
- custom_yolo_predictor : custom YOLO Ultralytics for 4-channels predict
- custom_yolo_trainer : custom YOLO Ultralytics for 4-channels training
- data : standard training data in YOLO Ultralytics format (non k-fold)
- 3_fold_dataset: training dataset for k-fold YOLO detect and segment training
- 3_fold_detect_run : directory where parameters_*.py and data_*.yaml are stored for 3-fold cross validation training in one go (detect)
- 3_fold_segment_run : directory where parameters_*.py and data_*.yaml are stored for 3-fold cross validation training in one go (segment)

# File Description
- evaluate_ensemble.py : previously created to evaluate YOLO-UNet essemble
- evaluate_model.py : created to evaluate YOLOv12-seg model in DSC 
- reconstruct_yolo_masks: primarily used to generate masks from the validation dataset and used in conjunction with evaluate_model.py to evaluate the YOLOv12 DSC
- run_yolo : runs YOLOv12 inference or training
- parameters.py : separate parameter configuration file. Used in conjunction with run_yolo.py
- ...*pt : pretrained checkpoints from YOLO Ultralytics



