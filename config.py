import torch.nn as nn
import torch.optim as optim
import os

# Data configuration

environment = os.environ.get('ENVIRONMENT', 'local')  # Default to 'local' if the environment variable is not set

if environment == 'local':
    data_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/train_data/" #evaluate_data/ or train_data/
    test_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/DATA/test_data/"
    model_save_path = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/CorrectionModel/Checkpoints/"
    output_dir = "/Users/Gabriel/MRes_Medical_Imaging/RESEARCH_PROJECT/CorrectionModel/Predicted_Corrections/"
    print('Environment is: local')
elif environment == 'cluster':
    data_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/"
    test_dir = "/cluster/project2/UCSF_PDGM_dataset/UCSF-PDGM-v3/TEST_SET/"
    model_save_path = '/cluster/project2/UCSF_PDGM_dataset/CorrectionModel/Checkpoints/'
    output_dir = "/cluster/project2/UCSF_PDGM_dataset/CorrectionModel/Predicted_Corrections/"
    print('Environment is: cluster')


# Model configuration
in_channels = 3 # 3 input channels (original MRI, predicted segmentation, uncertainty map)
out_channels = 5 # 5 classes (0-4): 0 = no change needed, 1 = change to outer tumor region, 2 = change to enhancing tumor, 3 = change to tumor core
dropout = 0.3

# Data configuration
crop_size = (150, 180, 155)

# Training configuration
if environment == 'local':
    batch_size = 1
    learning_rate = 0.01
    epochs = 2
elif environment == 'cluster':
    batch_size = 1
    learning_rate = 0.01
    epochs = 80