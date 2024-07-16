import os
import numpy as np
import torch
import torchaudio
from typing import Tuple
from termcolor import cprint
from torchvision import transforms
from scipy.signal import resample, butter, filtfilt
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from PIL import Image

def preprocess_meg(meg_data, original_sf=200, target_sf=120):  
    # リサンプリング
    meg_data = torch.tensor(meg_data, dtype=torch.float32)
    resampled_data = torchaudio.transforms.Resample(orig_freq=original_sf, new_freq=target_sf)(meg_data)

    # バンドパスフィルタ
    nyquist = 0.5 * target_sf
    low = 0.1 / nyquist
    high = 40 / nyquist
    b, a = butter(1, [low, high], btype="band")
    filtered_data = filtfilt(b, a, resampled_data, axis=-1)

    # ベースライン補正
    baseline = np.mean(filtered_data[:, :int(0.5 * target_sf)], axis=-1, keepdims=True)
    corrected_data = filtered_data - baseline

    # 正規化とクリッピング
    mean = np.mean(corrected_data, axis=-1, keepdims=True)
    std = np.std(corrected_data, axis=-1, keepdims=True)
    normalized_data = (corrected_data - mean) / std
    scaled_data = np.clip(normalized_data, -20, 20)

    return torch.tensor(scaled_data, dtype=torch.float32)

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).float()
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i].numpy()
        X = preprocess_meg(X) 
        X = torch.tensor(X, dtype=torch.float32)
        
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return 1  # Ensure the channel dimension is set to 1
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
class ImageDataset(Dataset):
    def __init__(self, image_paths_file: str, base_dir: str, transform=None):
        with open(image_paths_file, 'r') as f:
            self.image_paths = f.read().splitlines()
        self.base_dir = base_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.base_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image
