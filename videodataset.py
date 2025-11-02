import os
import glob
import torch
import numpy as np
import pandas as pd
import cv2
import decord
from torch.utils.data import Dataset
from transformers import VideoMAEFeatureExtractor
from typing import Dict, List

# Set decord to use PyTorch tensors
decord.bridge.set_bridge('torch')

class VideoActionDataset(Dataset):
    """
    A robust PyTorch Dataset class that can load video data from two sources:
    1. Direct .mp4 video files (loaded with decord).
    2. Folders of .jpg frames (loaded with cv2).
    """
    
    def __init__(self, 
                 data_dir: str,
                 csv_file: str, 
                 feature_extractor: VideoMAEFeatureExtractor, 
                 num_frames_to_sample: int = 16):
        
        print(f"Initializing dataset from CSV: {csv_file}")
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.num_frames_to_sample = num_frames_to_sample
        
        try:
            self.labels_df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: Label file not found at {csv_file}")
            raise
            
        self.all_labels = sorted(self.labels_df['category_id'].unique())
        self.num_classes = len(self.all_labels)
        
        print(f"Found {len(self.labels_df)} video clips in {self.num_classes} classes.")

    def __len__(self) -> int:
        return len(self.labels_df)

    def _load_from_mp4(self, video_path: str) -> List[np.ndarray]:
        """Loads and samples frames from a .mp4 file."""
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num=self.num_frames_to_sample, dtype=int)
        frames = vr.get_batch(indices) # This is a torch.Tensor
        # Convert to list of NumPy arrays for the feature extractor
        frames_list = [frame.numpy() for frame in frames]
        return frames_list

    def _load_from_frames_folder(self, folder_path: str) -> List[np.ndarray]:
        """Loads and samples frames from a folder of .jpg files."""
        frame_paths = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg")),
            key=lambda x: int(os.path.basename(x).split('.')[0]) # Sort by frame number
        )
        total_frames = len(frame_paths)
        if total_frames == 0:
            raise FileNotFoundError(f"No .jpg frames found in {folder_path}")

        indices = np.linspace(0, total_frames - 1, num=self.num_frames_to_sample, dtype=int)
        
        frames_list = []
        for frame_idx in indices:
            frame_path = frame_paths[frame_idx]
            frame = cv2.imread(frame_path)
            if frame is None:
                raise IOError(f"Could not read frame {frame_path}")
            # Convert from BGR (OpenCV) to RGB (Hugging Face)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame)
        return frames_list

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.labels_df.iloc[idx]
        relative_path = row['clip_path']
        label = row['category_id']
        
        full_path = os.path.join(self.data_dir, relative_path)
        
        try:
            # Check if the path is a directory or a file
            if os.path.isdir(full_path):
                frames_list = self._load_from_frames_folder(full_path)
            elif os.path.isfile(full_path):
                frames_list = self._load_from_mp4(full_path)
            else:
                raise FileNotFoundError(f"Path not found: {full_path}")

            # Preprocess with the feature extractor
            inputs = self.feature_extractor(frames_list, return_tensors="pt")
            
            # --- TYPO REMOVED ---
            
            # Squeeze(0) removes the batch dim, giving (C, T, H, W)
            squeezed_tensor = inputs.pixel_values.squeeze(0) 
            
            return {
                "pixel_values": squeezed_tensor, 
                "labels": torch.tensor(label, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Warning: Failed to load clip {full_path} at index {idx}. Error: {e}")
            return self[0] # Fallback to the first item