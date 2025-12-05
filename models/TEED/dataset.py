import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import h5py
from scipy.signal import butter, sosfiltfilt

DATASET_NAMES = [
    'SEISMIC', 'BIPED', 'BIPED-B2', 'BIPED-B3', 'BIPED-B5', 'BIPED-B6',
    'BSDS', 'BRIND', 'ICEDA', 'BSDS300', 'CID', 'DCD', 'MDBD',
    'PASCAL', 'NYUD', 'BIPBRI', 'UDED', 'DMRIR', 'CLASSIC'
]

BIPED_MEAN = [103.939, 116.779, 123.68, 137.86]
DEFAULT_MEAN = [104.007, 116.669, 122.679, 137.86]


# ============ SEISMIC UTILITIES ============
def bandpass(data, fs, lowcut=2, highcut=10, order=4):
    """Apply bandpass filter to seismic data."""
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=1)

def load_seismic_h5(h5_path):
    """Load and preprocess seismic H5 file.
    Returns:
        data: seismic data array (channels, time)
        dt: time sample interval (seconds)
        dx: spatial sample interval (meters)
    """
    with h5py.File(h5_path, "r") as fp:
        ds = fp["data"]
        data = ds[...]  # (channels, time)
        dt = float(ds.attrs.get("dt_s", 0.005))
        dx = float(ds.attrs.get("dx_m", 1.0))
    
    return data, dt, dx

def preprocess_seismic(data, dt):
    clip_val = np.percentile(np.abs(data), 95)
    data_clip = np.clip(data, -clip_val, clip_val)

    fs = 1.0 / dt
    data_filt = bandpass(data_clip, fs, lowcut=2, highcut=10)

    # Normalize to [0,1]
    img = (data_filt - data_filt.min()) / (data_filt.max() - data_filt.min() + 1e-8)

    # Expand to 4 channels for compatibility
    img = np.repeat(img[..., None], 4, axis=2)

    # Convert to uint8 0-255
    img = (img * 255).astype(np.uint8)
    return img

class DatasetConfig:
    """ Centralized dataset config for easier management"""

    DEFAULTS = {
        'img_height': 512,
        'img_width': 512,
        'yita': 0.5,
        'mean': DEFAULT_MEAN,
        'test_list': None,
        'train_list': None,
    }
    
    CONFIGS = {
        'SEISMIC': {
            'img_height': 1024,
            'img_width': 1024,
            'mean': DEFAULT_MEAN,
            'yita': 0.5,
        },
        'BIPED': {
            'img_height': 720,
            'img_width': 1280,
            'mean': BIPED_MEAN,
            'yita': 0.5,
        },
        'BIPED-B2': {
            'img_height': 720,
            'img_width': 1280,
            'mean': BIPED_MEAN,
            'yita': 0.5,
        },
        'BIPED-B3': {
            'img_height': 720,
            'img_width': 1280,
            'mean': BIPED_MEAN,
            'yita': 0.5,
        },
        'BIPED-B5': {
            'img_height': 720,
            'img_width': 1280,
            'mean': BIPED_MEAN,
            'yita': 0.5,
        },
        'BIPED-B6': {
            'img_height': 720,
            'img_width': 1280,
            'mean': BIPED_MEAN,
            'yita': 0.5,
        },
        'BSDS': {
            'img_height': 512,
            'img_width': 512,
            'mean': DEFAULT_MEAN,
            'yita': 0.5,
        },
        'BSDS300': {
            'img_height': 512,
            'img_width': 512,
            'mean': DEFAULT_MEAN,
            'yita': 0.5,
        },
        'BRIND': {
            'img_height': 512,
            'img_width': 512,
            'mean': DEFAULT_MEAN,
            'yita': 0.5,
        },
        'ICEDA': {
            'img_height': 1024,
            'img_width': 1408,
            'mean': DEFAULT_MEAN,
            'yita': 0.5,
        },
        'CID': {
            'img_height': 512,
            'img_width': 512,
            'mean': DEFAULT_MEAN,
            'yita': 0.3,
        },
        'PASCAL': {
            'img_height': 416,
            'img_width': 512,
            'mean': DEFAULT_MEAN,
            'yita': 0.3,
        },
        'NYUD': {
            'img_height': 448,
            'img_width': 560,
            'mean': DEFAULT_MEAN,
            'yita': 0.5,
        },
        'MDBD': {
            'img_height': 720,
            'img_width': 1280,
            'mean': DEFAULT_MEAN,
            'yita': 0.3,
        },
        'DCD': {
            'img_height': 352,
            'img_width': 480,
            'mean': DEFAULT_MEAN,
            'yita': 0.2,
        },
        'CLASSIC': {
            'img_height': 512,
            'img_width': 512,
            'mean': DEFAULT_MEAN,
            'yita': 0.5,
        },
        'UDED': {
            'img_height': 512,
            'img_width': 512,
            'mean': DEFAULT_MEAN,
            'yita': 0.5,
        },
    }

    @classmethod
    def get(cls, dataset_name: str) -> Dict:
        """Get configuration for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset configuration
        """
        if dataset_name not in DATASET_NAMES:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Choose from {DATASET_NAMES}")
        
        config = cls.DEFAULTS.copy()
        if dataset_name in cls.CONFIGS:
            config.update(cls.CONFIGS[dataset_name])
        return config


def dataset_info(dataset_name: str, is_linux: bool = True) -> Dict:
    """Get dataset information and paths."""
    config = DatasetConfig.get(dataset_name)
    
    config['data_dir'] = os.environ.get(f'{dataset_name}_DATA_DIR', 'data/{dataset_name}')
    config['test_list'] = os.environ.get(f'{dataset_name}_TEST_LIST', 'test_pair.lst')
    config['train_list'] = os.environ.get(f'{dataset_name}_TRAIN_LIST', 'train_pair.lst')
    
    return config


class TestDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 test_data: str,
                 img_height:int,
                 img_width: int,
                 test_list: Optional[str] = None,
                 arg=None):
        
        if test_data not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset: {test_data}")

        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.args = arg
        self.up_scale = arg.up_scale if arg else False
        self.mean_bgr = arg.mean_test
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

    def get_mean(self, arg) -> List[float]:
        """ Extract BGR mean vals"""
        if arg and hasattr(arg, 'mean_test'):
            mean - arg.mean_test
            return mean[:3] if len(mean) > 3 else mean 
        return DEFAULT_MEAN[:3]

    def _build_index(self):
        sample_indices = []

        if self.test_data.upper() == 'SEISMIC':
            if not self.test_list:
                raise ValueError("test_list not provided for SEISMIC dataset")
            
            list_path = self._resolve_path(self.test_list)
            with open(list_path, "r") as f:
                h5_files = [line.strip() for line in f if line.strip()]
            return [self._resolve_path(f, relative_to_data_root=True) for f in h5_files]

        if self.test_data == "CLASSIC":
            images_path = self.data_root if not self.test_list else self._resolve_path(self.test_list)
            return [images_path, None]

        # For other datasets
        if not self.test_list:
            raise ValueError(f"test_list required for {self.test_data} dataset")

        list_path = self._resolve_path(self.test_list)
        with open(list_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid line in {list_path}: {line}")
            
            img_path = self._resolve_path(parts[0], relative_to_data_root=True)
            label_path = self._resolve_path(parts[1], relative_to_data_root=True)
            sample_indices.append((img_path, label_path))

        return sample_indices
    
    def _resolve_path(self, path: str, relative_to_data_root: bool = False) -> str:
        """Resolve a path, handling both absolute and relative paths."""
        if os.path.isabs(path):
            return path
        
        if relative_to_data_root:
            return os.path.join(self.data_root, path)
        return path

    def __len__(self):
            if self.test_data.upper() == 'SEISMIC':
                return len(self.data_index)
            elif self.test_data == 'CLASSIC':
                return len(self.data_index[0]) if isinstance(self.data_index[0], list) else 1
            else:
                return len(self.data_index)

    def __getitem__(self, idx: int) -> Dict:
        if self.test_data.upper() == 'SEISMIC':
            return self._get_seismic_item(idx)
        elif self.test_data == 'CLASSIC':
            return self._get_classic_item(idx)
        else:
            return self._get_standard_item(idx)
        
    def _get_seismic_item(self, idx: int) -> Dict:
        """Load seismic data item."""
        img_path = self.data_index[idx]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # convert grayscale to 3-channel
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        
        img = img.astype(np.float32)
        img -= np.array(self.mean_bgr, dtype=np.float32)
        image = img.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        
        label = torch.zeros((1, image.shape[1], image.shape[2]))
        file_name = os.path.basename(img_path)
        
        return {
            'images': image,
            'labels': label,
            'file_names': file_name,
            'image_shape': image.shape
        }
        
    def _get_standard_item(self, idx: int) -> Dict:
        """Load standard dataset item (image + label pair)."""
        image_path, label_path = self.data_index[idx]
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if label_path and not os.path.exists(label_path):
            raise FileNotFoundError(f"Label not found: {label_path}")
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) if label_path else None
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        if label is None and label_path:
            raise ValueError(f"Failed to load label: {label_path}")
        
        im_shape = image.shape
        image, label = self.transform(image, label)
        file_name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        
        return {
            'images': image,
            'labels': label,
            'file_names': file_name,
            'image_shape': im_shape
        }

     
    def transform(self, img, gt):
        
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.squeeze().cpu().numpy()

        img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)
        gt = cv2.resize(gt, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)

        
        img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)
        if gt is not None:
            gt = cv2.resize(gt, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)

        if self.up_scale:
            img = cv2.resize(img, (0, 0), fx=1.3, fy=1.3)

        if img.shape[0] < 512 or img.shape[1] < 512:
            img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)

        if img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
            img_width = ((img.shape[1] // 8) + 1) * 8
            img_height = ((img.shape[0] // 8) + 1) * 8
            img = cv2.resize(img, (img_width, img_height))

        # Normalize and convert
        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        # Process label
        if self.test_data == "CLASSIC" or gt is None:
            gt = torch.zeros((1, img.shape[1], img.shape[2]))
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


# ============ TRAINING DATASETS ============
class BipedDataset(Dataset):
    """Training dataset for BIPED and similar edge detection datasets."""
    
    def __init__(self,
                 data_root: str,
                 img_height: int,
                 img_width: int,
                 train_mode: str = 'train',
                 crop_img: bool = False,
                 arg=None):
        """Initialize BIPED training dataset."""

        if train_mode not in ['train', 'test']:
            raise ValueError(f"Invalid train_mode: {train_mode}")

        self.data_root = data_root
        self.train_mode = train_mode
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = self._get_mean(arg)
        self.crop_img = crop_img
        self.arg = arg
        self.data_index = self._build_index()

    def _get_mean(self, arg) -> List[float]:
        """Extract BGR mean values."""
        if arg and hasattr(arg, 'mean_train'):
            mean = arg.mean_train
            return mean[:3] if len(mean) > 3 else mean
        return DEFAULT_MEAN[:3]

    def _build_index(self) -> List[Tuple[str, str]]:
        """Build training index from list file."""
        if not self.arg or not hasattr(self.arg, 'train_list'):
            raise ValueError("train_list argument required")

        list_path = self.arg.train_list
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Train list not found: {list_path}")

        sample_indices = []
        with open(list_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid line in {list_path}: {line}")
            
            img_path = parts[0]
            gt_path = parts[1]
            
            # Handle both absolute and relative paths
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.data_root, img_path)
            if not os.path.isabs(gt_path):
                gt_path = os.path.join(self.data_root, gt_path)
            
            sample_indices.append((img_path, gt_path))

        return sample_indices

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Dict:
        image_path, label_path = self.data_index[idx]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Label not found: {label_path}")

        image, label = self.transform(image, label)

        return {
            'images': image,
            'labels': label,
            'file_names': os.path.basename(image_path)
        }

    def transform(self, img: np.ndarray, gt: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform image and label for training with augmentation."""
        gt = np.array(gt, dtype=np.float32) / 255.
        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr

        h, w = gt.shape
        crop_size = self.img_height if self.img_height == self.img_width else self.img_height

        # Data augmentation: random cropping
        if w > 400 and h > 400:
            if np.random.random() > 0.4:
                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
                img = img[i:i + crop_size, j:j + crop_size]
                gt = gt[i:i + crop_size, j:j + crop_size]
            else:
                crop_size_alt = 300
                i = random.randint(0, h - crop_size_alt)
                j = random.randint(0, w - crop_size_alt)
                img = img[i:i + crop_size_alt, j:j + crop_size_alt]
                gt = gt[i:i + crop_size_alt, j:j + crop_size_alt]
                img = cv2.resize(img, (crop_size, crop_size))
                gt = cv2.resize(gt, (crop_size, crop_size))
        else:
            img = cv2.resize(img, (crop_size, crop_size))
            gt = cv2.resize(gt, (crop_size, crop_size))

        gt[gt > 0.1] += 0.2
        gt = np.clip(gt, 0., 1.)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()

        return img, gt

# ========================================

class SeismicTrainDataset(Dataset):
    """Training dataset for seismic H5 files."""
    
    def __init__(self, 
                 file_list: List[str],
                 img_height: int = 1024,
                 img_width: int = 1024,
                 arg=None):
        
        self.files = file_list
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = self._get_mean(arg)

    def _get_mean(self, arg) -> List[float]:
        """Extract BGR mean values."""
        if arg and hasattr(arg, 'mean_train'):
            mean = arg.mean_train
            return mean[:3] if len(mean) > 3 else mean
        return DEFAULT_MEAN[:3]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        h5_path = self.files[idx]
        
        # Load seismic data
        data, dt, dx = load_seismic_h5(h5_path)
        img = preprocess_seismic(data, dt)
        img = cv2.resize(img.T, (self.img_width, self.img_height))
        
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        img = img.astype(np.float32)
        img -= np.array(self.mean_bgr, dtype=np.float32)
        
        # convert to tensor [C, H, W]
        image = img.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        
        label = torch.zeros((1, self.img_height, self.img_width)) # Dummy label

        return {
            'images': image,
            'labels': label,
            'file_names': os.path.basename(h5_path)
        }

