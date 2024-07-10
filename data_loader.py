import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import config

class CorrectionDataset(Dataset):
    def __init__(self, data_dir, patient_folders, crop_size=config.crop_size, transform=None):
        self.data_dir = data_dir
        self.patient_folders = patient_folders
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, index):
        patient_folder = self.patient_folders[index]
        patient_number = patient_folder.split("_")[0].split("-")[-1]

        # Load original MRI, predicted segmentation, uncertainty map, and error mask
        mri_path = os.path.join(self.data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_T2_bias.nii.gz")
        pred_seg_path = os.path.join(self.data_dir, "predictions", f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
        uncertainty_path = os.path.join(self.data_dir, "predictions", f"uncertainty_UCSF-PDGM-{patient_number}.nii.gz")
        error_mask_path = os.path.join(self.data_dir, "error_masks", f"UCSF-PDGM-{patient_number}_error_mask.nii.gz")

        mri_image = self._load_nifti_image(mri_path)
        pred_seg_image = self._load_nifti_image(pred_seg_path)
        uncertainty_image = self._load_nifti_image(uncertainty_path)
        error_mask = self._load_nifti_image(error_mask_path)

        # Center crop all images
        mri_image = self._center_crop(mri_image, self.crop_size)
        pred_seg_image = self._center_crop(pred_seg_image, self.crop_size)
        uncertainty_image = self._center_crop(uncertainty_image, self.crop_size)
        error_mask = self._center_crop(error_mask, self.crop_size)

        # Normalize MRI image
        mri_image = mri_image / np.max(mri_image)

        # Combine inputs
        input_image = np.stack([mri_image, pred_seg_image, uncertainty_image], axis=0)

        # Convert input and error mask to float32
        input_image = input_image.astype(np.float32)
        error_mask = error_mask.astype(np.float32)

        # Map intensity values to class indices for error mask
        intensity_to_class = {
            0: 0,  # No change needed
            2: 1,  # Change to outer tumor region
            4: 2,  # Change to enhancing tumor
            1: 3   # Change to tumor core
        }
        map_func = np.vectorize(lambda x: intensity_to_class[x])
        error_mask = map_func(error_mask)

        # Add channel dimension to error mask
        error_mask = np.expand_dims(error_mask, axis=0)

        # Convert to PyTorch tensors
        input_image = torch.from_numpy(input_image)
        error_mask = torch.from_numpy(error_mask).long()

        # Apply transform if provided
        if self.transform:
            input_image = self.transform(input_image)

        return input_image, error_mask

    def _load_nifti_image(self, path):
        return nib.load(path).get_fdata()

    def _center_crop(self, image, crop_size):
        depth, height, width = image.shape
        crop_depth, crop_height, crop_width = crop_size

        start_depth = (depth - crop_depth) // 2
        start_height = (height - crop_height) // 2
        start_width = (width - crop_width) // 2

        cropped_image = image[start_depth:start_depth+crop_depth,
                              start_height:start_height+crop_height,
                              start_width:start_width+crop_width]
        return cropped_image

    @staticmethod
    def split_data(data_dir, train_ratio=0.5, val_ratio=0.5, test_ratio=0.0, seed=None):
        random.seed(seed)
        all_patient_folders = [folder for folder in os.listdir(data_dir) 
                               if folder.startswith("UCSF-PDGM-") and "FU" not in folder and "541" not in folder]
        random.shuffle(all_patient_folders)
        num_patients = len(all_patient_folders)
        train_size = int(num_patients * train_ratio)
        val_size = int(num_patients * val_ratio)
        train_folders = all_patient_folders[:train_size]
        val_folders = all_patient_folders[train_size:train_size + val_size]
        test_folders = all_patient_folders[train_size + val_size:]
        return train_folders, val_folders, test_folders

def visualize_example(data_dir):
    train_folders, _, _ = CorrectionDataset.split_data(data_dir)
    dataset = CorrectionDataset(data_dir, train_folders)
    input_image, error_mask = dataset[0]

    # Convert PyTorch tensors back to numpy arrays for visualization
    input_image = input_image.numpy()
    error_mask = error_mask.numpy()

    # Visualize middle slices of each input channel and the error mask
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(input_image[0, :, input_image.shape[2]//2, :], cmap='gray')
    axes[0, 0].set_title("Original MRI")
    axes[0, 1].imshow(input_image[1, :, input_image.shape[2]//2, :], cmap='jet')
    axes[0, 1].set_title("Predicted Segmentation")
    axes[1, 0].imshow(input_image[2, :, input_image.shape[2]//2, :], cmap='viridis')
    axes[1, 0].set_title("Uncertainty Map")
    axes[1, 1].imshow(error_mask[0, :, error_mask.shape[2]//2, :], cmap='jet')
    axes[1, 1].set_title("Error Mask")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = config.data_dir
    visualize_example(data_dir)