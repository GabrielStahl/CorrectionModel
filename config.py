import os
import torch

# Configuration
environment = os.environ.get('ENVIRONMENT', 'local')  # Default to 'local' if the environment variable is not set

# CHOOSE 
data_subset = "train_set" # CHOOSE FROM: train_set, val_set, test_set
UMap = "softmax" # CHOOSE FROM: modality_ensemble, softmax, dropout, deep_ensemble, test_time_augmentation
ignore_background = True

# CHOOSE class weights for weighted DiceLoss calculation
class_weights = torch.tensor([0.2, 1.0, 1.5, 1.5]) 

# all 1
#class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])

# Model configuration
in_channels = 3 # 3 input channels (original MRI, predicted segmentation, uncertainty map)
out_channels = 4 # 4 classes (0-3): 0 = Background, 1 = outer tumor region, 2 = enhancing tumor, 3 = tumor core
dropout = 0.3

# Data configuration
crop_size = (150, 180, 155)

# Training configuration
if environment == 'local':
    batch_size = 1
    learning_rate = 0.0001
    epochs = 30
elif environment == 'cluster':
    batch_size = 1
    learning_rate = 0.0001
    epochs = 100

if environment == 'local':
    root_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/" 
    train_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/train_data/"
    val_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/val_data/"
    test_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/test_data/"
    model_save_path = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/CorrectionModel/Checkpoints/"
    ensemble_path = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/BasicSeg3D/checkpoints/modality_ensemble/"
    output_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/BasicSeg3D/Predicted_Segmentations/"
    print('Environment is: local')
elif environment == 'cluster':
    root_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/"
    train_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/TRAIN_SET/"
    val_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/VAL_SET/"
    test_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/TEST_SET/"
    model_save_path = '/cluster/project2/UCSF_PDGM_dataset/CorrectionModel/Checkpoints/'
    ensemble_path = "/cluster/project2/UCSF_PDGM_dataset/BasicSeg/Checkpoints/modality_ensemble/"
    output_dir = f"/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/predictions_{data_subset}/"
    print('Environment is: cluster')