import os
import numpy as np
import nibabel as nib
import torch
from model import CorrectionUNet
from low_memory_model import UltraLightCorrectionUNet
import config
import torch.nn as nn
from data_loader import CorrectionDataset
from tqdm import tqdm
from torch.utils.data import DataLoader

class CorrectionInference:
    def __init__(self, model):
        self.model = model
        self.crop_size = config.crop_size
        self.original_shape = (240, 240, 155)

    def postprocess_output(self, output):
        # Convert output tensor to numpy array
        output_numpy = output.detach().cpu().numpy()
        
        # Remove batch dimension
        output_numpy = output_numpy[0, :, :, :]
        
        # Map the class indices to the original intensity values
        class_to_intensity = {
            0: 0,  # Background
            1: 2,  # Outer tumor region
            2: 4,  # Enhancing tumor
            3: 1,  # Tumor core
        }
        map_func = np.vectorize(lambda x: class_to_intensity[x])
        correction_mask = map_func(output_numpy).astype(np.uint8)
        
        # Pad the correction mask to the original shape
        padded_mask = self.pad_to_original_shape(correction_mask, dtype=np.uint8)
        
        return padded_mask
    
    def pad_to_original_shape(self, mask, dtype=np.uint8):
        depth, height, width = self.original_shape
        crop_depth, crop_height, crop_width = self.crop_size

        pad_depth = (depth - crop_depth) // 2
        pad_height = (height - crop_height) // 2
        pad_width = (width - crop_width) // 2

        padded_mask = np.zeros(self.original_shape, dtype=dtype)
        padded_mask[pad_depth:pad_depth+crop_depth, 
                    pad_height:pad_height+crop_height, 
                    pad_width:pad_width+crop_width] = mask

        return padded_mask

    def perform_inference(self, data_loader, device):
        self.model.eval()
        correction_masks = []
        patient_numbers = []

        with torch.no_grad():
            for input_tensor, _, patient_number in tqdm(data_loader): # ignore target image
                input_tensor = input_tensor.to(device)
                output = self.model(input_tensor)
                
                # Apply softmax to obtain class probabilities
                output = nn.functional.softmax(output, dim=1)
                
                # Apply argmax to obtain the class indices
                output = torch.argmax(output, dim=1)
                
                # Postprocess the output
                correction_mask = self.postprocess_output(output)
                correction_masks.append(correction_mask)

                patient_numbers.append(patient_number[0])  # Extract from batch
        
        return correction_masks, patient_numbers

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    environment = config.environment

    # Create the model
    if environment == "local":
        model = UltraLightCorrectionUNet(in_channels=config.in_channels, out_channels=config.out_channels)
    else:
        model = UltraLightCorrectionUNet(in_channels=config.in_channels, out_channels=config.out_channels) # after having solved memory issues: use CorrectionUNet
        
    model.to(device)

    # Load the trained model weights
    weights = "T1c_bias_modality_ensemble_correction_model_epoch_5.pth" # best epoch
    model_save_path = os.path.join(config.model_save_path, weights)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded trained model weights from: {model_save_path}")
    else:
        print(f"Trained model weights not found at: {model_save_path}")
        return

    # Set the model to evaluation mode
    model.eval()

    # Set data directory
    if config.data_subset == "train_set":
        directory = config.train_dir
    elif config.data_subset == "val_set":
        directory = config.val_dir
    elif config.data_subset == "test_set":
        directory = config.test_dir

    inference_folders = [f for f in os.listdir(directory) if f.startswith("UCSF-PDGM")]

    print(f"Getting patients from directory: {directory}")
    print(f"Performing inference on: {len(inference_folders)} patients")

    # Create an instance of the CorrectionInference class
    inference = CorrectionInference(model)
    
    dataset = CorrectionDataset(config.data_subset, test_mode=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Ensure the output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # Perform inference
    correction_masks, patient_numbers = inference.perform_inference(data_loader, device)
    
    # Save the correction masks
    for correction_mask, patient_number in zip(correction_masks, patient_numbers):
        output_path = os.path.join(config.root_dir, f"corrections_{config.data_subset}",f"correction_UCSF-PDGM-{patient_number}.nii.gz")
        correction_nifti = nib.Nifti1Image(correction_mask, affine=np.eye(4))
        nib.save(correction_nifti, output_path)
        print(f"Correction mask saved at: {output_path}")

if __name__ == "__main__":
    main()