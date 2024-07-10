import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import config

def load_nifti(file_path):
    return nib.load(file_path).get_fdata()

def save_nifti(data, file_path):
    nib.save(nib.Nifti1Image(data, np.eye(4)), file_path)

def generate_error_mask(pred_seg, true_seg):
    error_mask = np.zeros_like(true_seg)
    error_mask[pred_seg != true_seg] = true_seg[pred_seg != true_seg]
    return error_mask

def process_patient(patient_folder, pred_seg_dir, data_dir, output_dir):
    patient_number = patient_folder.split("_")[0].split("-")[-1]
    
    # Load predicted and true segmentations
    pred_seg_path = os.path.join(pred_seg_dir, f"segmentation_UCSF-PDGM-{patient_number}.nii.gz")
    true_seg_path = os.path.join(data_dir, patient_folder, f"UCSF-PDGM-{patient_number}_tumor_segmentation.nii.gz")
    
    pred_seg = load_nifti(pred_seg_path)
    true_seg = load_nifti(true_seg_path)
    
    # Generate error mask
    error_mask = generate_error_mask(pred_seg, true_seg)
    
    # Save error mask
    output_path = os.path.join(output_dir, f"UCSF-PDGM-{patient_number}_error_mask.nii.gz")
    save_nifti(error_mask, output_path)

def main():
    data_dir = config.data_dir
    pred_seg_dir = os.path.join(data_dir, "predictions")
    output_dir = os.path.join(data_dir, "error_masks")
    os.makedirs(output_dir, exist_ok=True)

    patient_folders = [folder for folder in os.listdir(data_dir) 
                       if folder.startswith("UCSF-PDGM-") and "FU" not in folder and "541" not in folder]

    for patient_folder in tqdm(patient_folders, desc="Generating error masks"):
        process_patient(patient_folder, pred_seg_dir, data_dir, output_dir)

    print(f"Error masks generated and saved in {output_dir}")

if __name__ == "__main__":
    main()