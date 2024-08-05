import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import config

class CorrectionDataset(Dataset):
    def __init__(self, data_subset, modality = "T1c_bias", UMap = "modality_ensemble", crop_size=config.crop_size, transform=None, test_mode=False):
        
        self.test_mode = test_mode # if true, don't try to load target images (error masks)
        
        self.data_subset = data_subset

        if data_subset == "train_set":
            self.data_dir = config.train_dir
        elif data_subset == "val_set":
            self.data_dir = config.val_dir
        elif data_subset == "test_set":
            self.data_dir = config.test_dir
        
        self.patient_folders = [folder for folder in os.listdir(self.data_dir) if folder.startswith("UCSF-PDGM-") and "FU" not in folder]
        self.crop_size = crop_size
        self.transform = transform
        self.modality = modality
        self.UMap = UMap

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, index):
        patient_folder = self.patient_folders[index]
        patient_number = patient_folder.split("_")[0].split("-")[-1]

        # Load original MRI, predicted segmentation, uncertainty map, and error mask
        mri_path = os.path.join(self.data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_{self.modality}.nii.gz")
        pred_seg_path = os.path.join(config.root_dir, f"predictions_{self.data_subset}", self.UMap, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
        uncertainty_path = os.path.join(config.root_dir, f"predictions_{self.data_subset}", self.UMap, f"{self.UMap}_UMap_UCSF-PDGM-{patient_number}.nii.gz")

        mri_image = self._load_nifti_image(mri_path)
        pred_seg_image = self._load_nifti_image(pred_seg_path)
        uncertainty_image = self._load_nifti_image(uncertainty_path)

        # print unique values in the predicted segmentation image
        print(f"Unique values in the predicted segmentation image: {np.unique(pred_seg_image)}")

        # Center crop all images
        mri_image = self._center_crop(mri_image, self.crop_size)
        pred_seg_image = self._center_crop(pred_seg_image, self.crop_size)
        uncertainty_image = self._center_crop(uncertainty_image, self.crop_size)

        # Map intensity values to class indices for predicted segmentation
        intensity_to_class = {
            0: 0,  # Background
            2: 1,  # Outer tumor region
            4: 2,  # Enhancing tumor
            1: 3   # Tumor core
        }
        map_func = np.vectorize(lambda x: intensity_to_class[x])
        pred_seg_image = map_func(pred_seg_image)

        # Normalize MRI image
        mri_image = mri_image / np.max(mri_image)

        # Combine inputs
        input_image = np.stack([mri_image, pred_seg_image, uncertainty_image], axis=0)

        # Convert input and error mask to float32
        input_image = input_image.astype(np.float32)

        # Convert to PyTorch tensors
        input_image = torch.from_numpy(input_image)

        # Apply transform if provided
        if self.transform:
            input_image = self.transform(input_image)

        if self.test_mode:
            return input_image, patient_number
        else:
            # Load and process error mask as before
            error_mask_path = os.path.join(config.root_dir, f"error_masks_{self.data_subset}", f"UCSF-PDGM-{patient_number}_error_mask.nii.gz")
            error_mask = self._load_nifti_image(error_mask_path)
            error_mask = self._center_crop(error_mask, self.crop_size)
            error_mask = error_mask.astype(np.float32)

            # Map intensity values to class indices for error mask
            intensity_to_class = {
                0: 0,  # Should be background
                2: 1,  # Should be outer tumor region
                4: 2,  # Should be enhancing tumor
                1: 3,  # Should be tumor core
                3: 4   # No change needed
            }
            map_func = np.vectorize(lambda x: intensity_to_class[x])
            error_mask = map_func(error_mask)
            error_mask = np.expand_dims(error_mask, axis=0)
            error_mask = torch.from_numpy(error_mask).long()

            return input_image, error_mask, patient_number

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