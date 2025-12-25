import os
import glob
import h5py
import numpy as np
import cv2
from scipy.signal import butter, sosfiltfilt
import torch

# Paths
INPUT_DIR = "data/SEISMIC"
OUTPUT_DIR = "data/SEISMIC_IMG"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Processing parameters
clip_percent = 95.0
band = [2, 10]  # Hz
order = 4

# Target sizes
TARGET_WIDTH = 512
TARGET_HEIGHT = 128    # adjust as needed for your net

EXPORT_SCALE = 2       # how much larger you want the PNG for display, e.g., 2x

def image_normalization(im, max_val=255.0):
    """
    Normalize image to range [0, 1].
    
    Args:
        im: Input image (numpy array)
        max_val: Maximum value in the image (default 255 for uint8)
        
    Returns:
        Normalized image
    """
    im = im.astype(np.float32)
    return im / max_val

def save_image_batch_to_disk(preds, output_dir, file_names, img_shape=None, arg=None, is_inchannel=False):
    """
    Save a batch of predicted edge maps to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle list of predictions
    if isinstance(preds, list):
        preds = preds[0]
    
    # Convert to numpy
    if isinstance(preds, torch.Tensor):
        preds = torch.sigmoid(preds)
        preds = preds.cpu().detach().numpy()
    
    # Process each prediction
    for i, pred in enumerate(preds):
        # Get file name
        if isinstance(file_names, (list, tuple)):
            fname = file_names[i] if i < len(file_names) else f"output_{i}.png"
        else:
            fname = file_names
        
        base_name = os.path.splitext(fname)[0] if isinstance(fname, str) else f"output_{i}"
        out_path = os.path.join(output_dir, f"{base_name}.png")
        
        # Handle shape: pred is [C, H, W] or [H, W]
        if len(pred.shape) == 3:
            pred = pred[0]  # Take first channel
        
        # Convert to uint8
        if pred.max() <= 1.0:
            pred = (pred > 0.5).astype(np.uint8) * 255
        else:
            pred = np.clip(pred, 0, 255).astype(np.uint8)
        
        # Resize if needed - img_shape is (C, H, W)
        if img_shape is not None:
            shape = img_shape if not isinstance(img_shape[0], (tuple, list)) else img_shape[i]
            if len(shape) == 3:
                h, w = int(shape[1]), int(shape[2])
            else:
                h, w = int(shape[0]), int(shape[1])
            
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Upscale for display
        h, w = pred.shape[:2]
        pred = cv2.resize(pred, (w * EXPORT_SCALE, h * EXPORT_SCALE), interpolation=cv2.INTER_NEAREST)

        # Save
        cv2.imwrite(out_path, pred)


def visualize_result(res_data, arg=None):
    """
    Visualize training results (original image, ground truth, and predictions).
    
    Args:
        res_data: List containing [image, gt, pred1, pred2, ...]
        arg: Arguments object (optional)
        
    Returns:
        Concatenated visualization image
    """
    # Extract components
    if len(res_data) < 2:
        raise ValueError("res_data must contain at least image and gt")
    
    image = res_data[0]
    gt = res_data[1]
    preds = res_data[2:] if len(res_data) > 2 else []
    
    # Convert to uint8 if needed
    def to_uint8(img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if len(img.shape) == 4:
            img = img[0]
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:
            img = img[0] if img.shape[0] == 1 else img.mean(axis=0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    
    image = to_uint8(image)
    gt = to_uint8(gt)
    preds = [to_uint8(p) for p in preds]
    
    # Resize all to same dimensions
    h, w = image.shape[:2]
    gt = cv2.resize(gt, (w, h))
    preds = [cv2.resize(p, (w, h)) for p in preds]
    
    # Concatenate horizontally
    all_imgs = [image, gt] + preds
    vis = np.concatenate(all_imgs, axis=1)
    
    return vis


def count_parameters(model):
    """
    Count the total number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)