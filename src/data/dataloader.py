import os
import json
import subprocess
import nibabel as nib
import tensorflow as tf
from skimage import exposure
from math import ceil
import matplotlib.pyplot as plt
import random
from skimage import img_as_float

class DataLoader:
    def __init__(self, data_path, view, data_id, split_ratio=None, crop=False, batch_size=None, split_json_path=None):
        # print("Done")
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.all_subjects = None
        self.subjects_lists = []
        self.labels = {'train': 0, 'test': 1, 'validation': 2}
        # self.size = [0, 0, 0]
        self.batch_size = batch_size
        self.slices_number = None
        self.view = view
        self.crop = crop
        self.data_id = data_id
        self.split_json_path = split_json_path
        
    def list_subjects(self):
        subjects = os.listdir(self.data_path)

        if self.data_id == 'Motion' or self.data_id == 'Motion_Simulated': 
            subjects = [item for item in subjects if item.startswith('sub')]
        elif self.data_id == 'BraTS':
            subjects = [item for item in subjects if item.startswith('BraTS2021')]
            
        self.all_subjects = subjects

    def get_nifti_path(self, subject, number_of_motion=None, suffix=None):
        if self.data_id == 'Motion':
            ref_path_stand = f'{self.data_path}/{subject}/anat/{subject}_acq-standard_T1w.nii'
            ref_path_motion = f'{self.data_path}/{subject}/anat/{subject}_acq-headmotion{number_of_motion}_T1w.nii'
        
        elif self.data_id == 'Motion_Simulated':
            ref_path_stand = f'/kaggle/input/mmmai-regist-data/MR-ART-Regist/{subject}/anat/{subject}_acq-standard_T1w.nii'
            ref_path_motion = f'{self.data_path}/{subject}/anat/{subject}_acq-{suffix}.nii'
            
        elif self.data_id == 'BraTS':
            ref_path_stand = f'{self.data_path}/{subject}/{subject}_t1_No_Motion.nii'
            ref_path_motion = f'{self.data_path}/{subject}/{subject}_t1_Motion{number_of_motion}.nii'
        
        return [ref_path_stand, ref_path_motion]

    def get_paired_volumes(self, path):
        if os.path.exists(path[0]) and os.path.exists(path[1]):
            free_data = nib.load(path[0]).get_fdata()
            motion_data = nib.load(path[1]).get_fdata()
            
            if self.crop:
                if self.view == 'Sagittal':
                    free_data = exposure.rescale_intensity(free_data[37:-37], out_range=(0, 1.0))
                    motion_data = exposure.rescale_intensity(motion_data[37:-37], out_range=(0, 1.0))
                elif self.view == 'Axial':
                    free_data = exposure.rescale_intensity(free_data[:, :, 90:-27], out_range=(0, 1.0))
                    motion_data = exposure.rescale_intensity(motion_data[:, :, 90:-27], out_range=(0, 1.0))
                elif self.view == 'Coronal':
                    free_data = exposure.rescale_intensity(free_data[:,30:-49,:], out_range=(0, 1.0))
                    motion_data = exposure.rescale_intensity(motion_data[:,30:-49,:], out_range=(0, 1.0))
            else:
                free_data = exposure.rescale_intensity(free_data, out_range=(0, 1.0))
                motion_data = exposure.rescale_intensity(motion_data, out_range=(0, 1.0))  
                
            return tf.convert_to_tensor(free_data), tf.convert_to_tensor(motion_data)
        else:
            return None, None
            
    def pad_volume(self, volume):
        # Pad along the depth (first dimension)
        current_shape = tf.shape(volume)
        target_depth = 256
        pad_total = target_depth - current_shape[0]
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        padded_volume = tf.pad(volume, paddings=[[pad_before, pad_after], [0, 0], [0, 0]], mode='CONSTANT', constant_values=0)
        
        # If the data is from BraTS, pad along the z-dimension (third dimension)
        if self.data_id == 'BraTS':
            current_shape = tf.shape(padded_volume)
            pad_total = target_depth - current_shape[2]
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            barts_padded_volume = tf.pad(padded_volume, paddings=[[0, 0], [0, 0], [pad_before, pad_after]], mode='CONSTANT', constant_values=0)

            return barts_padded_volume
    
        return padded_volume


    def split_data(self):
        self.list_subjects()
        if self.split_json_path:
            # Load subjects from JSON file
            with open(self.split_json_path, 'r') as f:
                split_dict = json.load(f)
            
            # Extract and filter subjects
            train_subjects = [s for s in split_dict.get('train', []) if s in self.all_subjects]
            test_subjects = [s for s in split_dict.get('test', []) if s in self.all_subjects]
            validation_subjects = [s for s in split_dict.get('validation', []) if s in self.all_subjects]
            
            self.subjects_lists = [train_subjects, test_subjects, validation_subjects]

            if self.data_id == 'Motion_Simulated': 
                # Calculate dataset sizes
                self.size = [
                    len(train_subjects) * 4 * 256,
                    len(test_subjects) * 4 * 256,
                    len(validation_subjects) * 4 * 256
                ]
            else:
                # Calculate dataset sizes
                self.size = [
                    len(train_subjects) * 2 * 256,
                    len(test_subjects) * 2 * 256,
                    len(validation_subjects) * 2 * 256
                ]

        else:
            # Original ratio-based splitting
            if ceil(sum(self.split_ratio)) == 1 and len(self.split_ratio) <= 3:
                self.split_ratio.insert(0, 0)
                cumulative_sum = [sum(self.split_ratio[:i + 1]) for i in range(len(self.split_ratio))]
                number_of_subjects = len(self.all_subjects)

                for i in range(1, len(self.split_ratio)):
                    self.subjects_lists.append(
                        self.all_subjects[int(round(cumulative_sum[i - 1] * number_of_subjects)):int(
                            round(cumulative_sum[i] * number_of_subjects))])
            else:
                print("The Summation of ratios is not equal to 1")
   
       
    def generator(self, mode, enable_SAP=True):
        subjects = self.subjects_lists[self.labels[mode]]

        def data_gen():
            for subject in subjects:
                if self.data_id == 'Motion_Simulated':
                    for suffix in ["rot0to15nnods5_T1w", "rot0to15nnods10_T1w", 
                           "pitch15dur2p5nnods5_T1w", "pitch15dur2p5nnods10_T1w"]:

                        pathes = self.get_nifti_path(subject, suffix=suffix)
                        free_unpadded, motion_unpadded = self.get_paired_volumes(pathes)

                        if (free_unpadded is not None) and (motion_unpadded is not None):
                            free = self.pad_volume(free_unpadded)
                            motion = self.pad_volume(motion_unpadded)
                        else:
                            print("[WARNING] Skipped subject due to missing volume(s).")
                            print(f"  Free path: {pathes[0]}")
                            print(f"  Motion path: {pathes[1]}")
                            continue
    
                        if (free is not None) and (motion is not None):
                            if self.view == 'Sagittal':
                                self.slices_number = motion.shape[0]
                            elif self.view == 'Axial':
                                self.slices_number = motion.shape[2]
                            elif self.view == 'Coronal':
                                self.slices_number = motion.shape[1]
    
                            for slice_id in range(0, self.slices_number):
                                start_idx = slice_id + 1
                                end_idx = (slice_id + 1) + 1
                                if (end_idx < self.slices_number - 1):
                                    if self.view == 'Sagittal':
                                        free_slice = free[start_idx:end_idx]
                                        motion_slice = motion[start_idx:end_idx]
                                        motion_before_slice = motion[start_idx-1:end_idx-1]
                                        motion_after_slice = motion[start_idx+1:end_idx+1]
                                        free_slice = tf.transpose(free_slice, perm=[1, 2, 0])
                                        motion_slice = tf.transpose(motion_slice, perm=[1, 2, 0])
                                        motion_before_slice = tf.transpose(motion_before_slice, perm=[1, 2, 0])
                                        motion_after_slice = tf.transpose(motion_after_slice, perm=[1, 2, 0])
    
                                    elif self.view == 'Axial':
                                        free_slice = free[:, :, start_idx:end_idx]
                                        motion_slice = motion[:, :, start_idx:end_idx]
                                        motion_before_slice = motion[:, :, start_idx-1:end_idx-1]
                                        motion_after_slice = motion[:, :, start_idx+1:end_idx+1]
                                        free_slice = tf.transpose(free_slice, perm=[0, 1, 2])
                                        motion_slice = tf.transpose(motion_slice, perm=[0, 1, 2])
                                        motion_before_slice = tf.transpose(motion_before_slice, perm=[0, 1, 2])
                                        motion_after_slice = tf.transpose(motion_after_slice, perm=[0, 1, 2])
    
                                    elif self.view == 'Coronal':
                                        free_slice = free[:, start_idx:end_idx, :]
                                        motion_slice = motion[:, start_idx:end_idx, :]
                                        motion_before_slice = motion[:, start_idx-1:end_idx-1, :]
                                        motion_after_slice = motion[:, start_idx+1:end_idx+1, :]
                                        free_slice = tf.transpose(free_slice, perm=[0, 2, 1])
                                        motion_slice = tf.transpose(motion_slice, perm=[0, 2, 1])
                                        motion_before_slice = tf.transpose(motion_before_slice, perm=[0, 2, 1])
                                        motion_after_slice = tf.transpose(motion_after_slice, perm=[0, 2, 1])
    

                                    if enable_SAP:
                                        yield (
                                            (motion_before_slice, motion_slice, motion_after_slice),
                                            free_slice
                                        )
                                    else:
                                        yield (
                                            (motion_slice),
                                            free_slice
                                        )
                else:
                    for i in range(1,3):
                        pathes = self.get_nifti_path(subject, number_of_motion=str(i))
                        free_unpadded, motion_unpadded = self.get_paired_volumes(pathes)
                        
                        if (free_unpadded is not None) and (motion_unpadded is not None):
                            free = self.pad_volume(free_unpadded)
                            motion = self.pad_volume(motion_unpadded)
                        else:
                            print("[WARNING] Skipped subject due to missing volume(s).")
                            print(f"  Free path: {pathes[0]}")
                            print(f"  Motion path: {pathes[1]}")
                            continue
        
                        if (free is not None) and (motion is not None):
                            if self.view == 'Sagittal':
                                self.slices_number = motion.shape[0]
                            elif self.view == 'Axial':
                                self.slices_number = motion.shape[2]
                            elif self.view == 'Coronal':
                                self.slices_number = motion.shape[1]
    
                            for slice_id in range(0, self.slices_number):
                                start_idx = slice_id + 1
                                end_idx = (slice_id + 1) + 1
                                if (end_idx < self.slices_number - 1):
                                    if self.view == 'Sagittal':
                                        free_slice = free[start_idx:end_idx]
                                        motion_slice = motion[start_idx:end_idx]
                                        motion_before_slice = motion[start_idx-1:end_idx-1]
                                        motion_after_slice = motion[start_idx+1:end_idx+1]
                                        free_slice = tf.transpose(free_slice, perm=[1, 2, 0])
                                        motion_slice = tf.transpose(motion_slice, perm=[1, 2, 0])
                                        motion_before_slice = tf.transpose(motion_before_slice, perm=[1, 2, 0])
                                        motion_after_slice = tf.transpose(motion_after_slice, perm=[1, 2, 0])
    
                                    elif self.view == 'Axial':
                                        free_slice = free[:, :, start_idx:end_idx]
                                        motion_slice = motion[:, :, start_idx:end_idx]
                                        motion_before_slice = motion[:, :, start_idx-1:end_idx-1]
                                        motion_after_slice = motion[:, :, start_idx+1:end_idx+1]
                                        free_slice = tf.transpose(free_slice, perm=[0, 1, 2])
                                        motion_slice = tf.transpose(motion_slice, perm=[0, 1, 2])
                                        motion_before_slice = tf.transpose(motion_before_slice, perm=[0, 1, 2])
                                        motion_after_slice = tf.transpose(motion_after_slice, perm=[0, 1, 2])
    
                                    elif self.view == 'Coronal':
                                        free_slice = free[:, start_idx:end_idx, :]
                                        motion_slice = motion[:, start_idx:end_idx, :]
                                        motion_before_slice = motion[:, start_idx-1:end_idx-1, :]
                                        motion_after_slice = motion[:, start_idx+1:end_idx+1, :]
                                        free_slice = tf.transpose(free_slice, perm=[0, 2, 1])
                                        motion_slice = tf.transpose(motion_slice, perm=[0, 2, 1])
                                        motion_before_slice = tf.transpose(motion_before_slice, perm=[0, 2, 1])
                                        motion_after_slice = tf.transpose(motion_after_slice, perm=[0, 2, 1])
    
                                
                                    if tf.math.count_nonzero(free_slice) == 0:
                                        continue
                                    
                                    if enable_SAP:
                                        yield (
                                            (motion_before_slice, motion_slice, motion_after_slice),
                                            free_slice
                                        )
                                    else:
                                        yield (
                                            (motion_slice),
                                            free_slice
                                        )

        input_signature = (
            (tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
             tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
             tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)),
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(data_gen, output_signature=input_signature)
        dataset = dataset.batch(self.batch_size)

        return dataset