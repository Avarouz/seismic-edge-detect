"""
TEED: Tiny Edge Detection Network
"""

from __future__ import print_function
import argparse
import os
import time, platform
from datetime import datetime
from pathlib import Path
from dataset import DATASET_NAMES, BipedDataset, TestDataset, SeismicTestDataset, dataset_info

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING']="0"

from dataset import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
from loss2 import *
from ted import TED # TEED architecture
from utils.img_processing import (image_normalization, save_image_batch_to_disk,
                   visualize_result, count_parameters)

IS_LINUX = platform.system() == "Linux"

class Config:
    ''' easier management for training / testing '''

    def __init__(self, args, train_inf, test_inf):
        self.args = args
        self.train_inf = train_inf
        self.test_inf = test_inf

        self.training_dir = Path(args.output_dir) / args.train_data
        self.training_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.training_dir / args.checkpoint_data

        # Loss weights for TEED
        self.l_weight0 = [1.1, 0.7, 1.1, 1.3]  # bdcn loss2-B4
        self.l_weight = [[0.05, 2.], [0.05, 2.], [0.01, 1.], [0.01, 3.]]  # cats loss
        
        self.device = torch.device('cuda' if torch.cuda.device_count() > 0 else 'cpu')
        
        self.tb_writer = None
        if args.tensorboard and not args.is_testing:
            self._setup_tensorboard()

    def _setup_tensorboard(self):
        """Initialize TensorBoard writer and log training settings."""
        self.tb_writer = SummaryWriter(log_dir=str(self.training_dir))
        
        training_notes = (
            f"{self.args.version_notes}\n"
            f"LR: {self.args.lr} | WD: {self.args.wd}\n"
            f"Image size: {self.args.img_width}x{self.args.img_height}\n"
            f"Adjust LR: {self.args.adjust_lr} | LRs: {self.args.lrs}\n"
            f"Loss functions: BDCNloss2 + CAST-loss2\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Trained on: {self.args.train_data}"
        )
        
        settings_path = self.training_dir / 'training_settings.txt'
        with open(settings_path, 'w') as f:
            f.write(training_notes)
        
        print("Training details:\n", training_notes)

    def print_device_info(self):
        """Print GPU and PyTorch information."""
        print(f"\nDevice: {self.device}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Train image mean: {self.args.mean_train}")
        print(f"Test image mean: {self.args.mean_test}\n")


def train_one_epoch(epoch, dataloader, model, criterions, optimizer, config):
    model.train()
    criterion1, criterion2 = criterions

    loss_avg =[]
    imgs_res_folder = Path(config.args.output_dir) / 'current_res'
    imgs_res_folder.mkdir(parents=True, exist_ok=True)

    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(config.device)  # BxCxHxW
        labels = sample_batched['labels'].to(config.device)  # BxHxW

        preds_list = model(images)
        loss1 = sum([criterion2(preds, labels,l_w) for preds, l_w in zip(preds_list[:-1],config.l_weight0)]) # bdcn_loss2 [1,2,3] TEED
        loss2 = criterion1(preds_list[-1], labels, config.l_weight[-1], config.device) # cats_loss [dfuse] TEED

        tLoss = loss2 + loss1 # TEED

        optimizer.zero_grad()
        tLoss.backward()
        optimizer.step()

        loss_avg.append(tLoss.item())
        

        if batch_id % config.args.show_log == 0:
            avg_batch_loss = np.array(loss_avg).mean()
            print(f"{time.ctime()} | Epoch: {epoch:3d} | Batch: {batch_id:4d}/{len(dataloader):4d} | "
                  f"Loss: {tLoss.item():.4f}")
            
        if batch_id % config.args.log_interval_vis == 0:
            _visualize_batch(images, labels, preds_list, tLoss, epoch, batch_id, 
                            dataloader, imgs_res_folder, config)
    
    return np.array(loss_avg).mean()


def _visualize_batch(images, labels, preds_list, loss, epoch, batch_id, dataloader, 
                     save_dir, config):
    """Visualize model predictions during training."""
    res_data = [images.cpu().numpy()[0]]
    res_data.append(labels.cpu().numpy()[0])
    
    for pred in preds_list:
        tmp = pred[0]
        tmp = torch.sigmoid(tmp).unsqueeze(dim=0)
        res_data.append(tmp.cpu().detach().numpy())
    
    vis_imgs = visualize_result(res_data, arg=config.args)
    vis_imgs = cv2.resize(vis_imgs, 
                          (int(vis_imgs.shape[1] * 0.8), int(vis_imgs.shape[0] * 0.8)))
    
    text = f"Epoch: {epoch} | Iter: {batch_id}/{len(dataloader)} | Loss: {loss.item():.4f}"
    cv2.putText(vis_imgs, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite(str(save_dir / 'results.png'), vis_imgs)
       

def validate_one_epoch(epoch, dataloader, model, config, test_resize=False):
    # XXX This is not really validation, but testing

    model.eval()
    
    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(config.device)
            file_names = sample_batched['file_names']
            
            preds = model(images, single_test=test_resize)
            
            output_dir = Path(config.args.output_dir) / config.args.train_data / str(epoch) / \
                        (config.args.test_data + '_res')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_image_batch_to_disk(preds[-1], str(output_dir), file_names,
                                    img_shape=None, arg=config.args)  # Don't resize


def test(checkpoint_path, dataloader, model, config, resize_input=False):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    model.eval()

    durations = []

    # just for the new dataset
    # os.makedirs(os.path.join(output_dir,"healthy"), exist_ok=True)
    # os.makedirs(os.path.join(output_dir,"infected"), exist_ok=True)

    with torch.no_grad():
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(config.device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']

            if config.device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            preds = model(images, single_test=resize_input)
            end = time.perf_counter()
            
            if config.device.type == 'cuda':
                torch.cuda.synchronize()
            
            durations.append(end - start)
            
            output_dir = Path(config.args.res_dir) / f"{config.args.train_data}2{config.args.test_data}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_image_batch_to_disk(preds, str(output_dir), file_names,
                                    image_shape, arg=config.args)
            
            torch.cuda.empty_cache()
    
    total_time = np.sum(durations)
    fps = len(dataloader) / total_time

    print("="*60)
    print(f"Testing finished on {config.args.test_data} dataset")
    print(f"FPS: {fps:.2f}")
    print(f"Total time: {total_time:.2f}s")
    print("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TEED: Tiny Edge Detection Network')
    
    # Mode flags
    parser.add_argument('--is_training', action='store_true', default=False,
                       help='Run in training mode.')
    parser.add_argument('--is_testing', action='store_true', default=False,
                       help='Run in testing mode.')
    
    parser.add_argument('--choose_test_data', type=int, default=None,
                       help='Index of test dataset (0-18). Only used if --test_data not specified.')
    
    # Dataset parameters
    parser.add_argument('--train_data', type=str, choices=DATASET_NAMES, default=DATASET_NAMES[0],
                       help='Training dataset name.')
    parser.add_argument('--test_data', type=str, choices=DATASET_NAMES, default=None,
                       help='Testing dataset name.')
    
    # Directory parameters
    parser.add_argument('--input_dir', type=str, default='data/train',
                       help='Path to training data directory.')
    parser.add_argument('--input_val_dir', type=str, default='data/test',
                       help='Path to validation/test data directory.')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Path to output checkpoint directory.')
    parser.add_argument('--res_dir', type=str, default='results',
                       help='Path to results directory.')
    
    # Data list parameters
    parser.add_argument('--train_list', type=str, default='train_list.txt',
                       help='Path to training list file.')
    parser.add_argument('--test_list', type=str, default='test_list.txt',
                       help='Path to test list file.')
    
    # Image parameters
    parser.add_argument('--img_width', type=int, default=300,
                       help='Image width for training.')
    parser.add_argument('--img_height', type=int, default=300,
                       help='Image height for training.')
    parser.add_argument('--test_img_width', type=int, default=320,
                       help='Image width for testing.')
    parser.add_argument('--test_img_height', type=int, default=320,
                       help='Image height for testing.')
    parser.add_argument('--crop_img', action='store_true', default=True,
                       help='Crop training images instead of resizing.')
    parser.add_argument('--up_scale', action='store_true', default=False,
                       help='Upscale test images by 1.5x.')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=8,
                       help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training.')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loading workers.')
    parser.add_argument('--lr', type=float, default=8e-4,
                       help='Initial learning rate.')
    parser.add_argument('--lrs', type=float, nargs='+', default=[8e-5],
                       help='Learning rates for adjust_lr epochs.')
    parser.add_argument('--wd', type=float, default=2e-4,
                       help='Weight decay.')
    parser.add_argument('--adjust_lr', type=int, nargs='+', default=[4],
                       help='Epochs at which to adjust learning rate.')
    
    # Checkpoint and resume
    parser.add_argument('--checkpoint_data', type=str, default='latest_model.pth',
                       help='Checkpoint filename to load.')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume training from checkpoint.')
    
    # Logging and device
    parser.add_argument('--tensorboard', action='store_true', default=True,
                       help='Use TensorBoard for logging.')
    parser.add_argument('--show_log', type=int, default=20,
                       help='Interval for printing logs.')
    parser.add_argument('--log_interval_vis', type=int, default=200,
                       help='Interval for visualizing predictions.')
    parser.add_argument('--use_gpu', type=int, default=0,
                       help='GPU device index.')
    
    # Misc
    parser.add_argument('--mean_train', type=float, default=104.00699,
                       help='Mean for training data normalization.')
    parser.add_argument('--mean_test', type=float, default=104.00699,
                       help='Mean for test data normalization.')
    parser.add_argument('--channel_swap', type=int, nargs='+', default=[2, 1, 0],
                       help='Channel order (for BGR/RGB conversion).')
    parser.add_argument('--predict_all', action='store_true', default=False,
                       help='Generate all TEED outputs.')
    parser.add_argument('--version_notes', type=str,
                       default='TEED training with BIPED dataset',
                       help='Notes about this training run.')
    
    args = parser.parse_args()
    
    # handle test_data: use explicit --test_data if provided, otherwise use --choose_test_data
    if args.test_data is None:
        if args.choose_test_data is not None:
            args.test_data = DATASET_NAMES[args.choose_test_data]
        else:
            args.test_data = args.train_data
    
    # Override with dataset-specific info if needed
    if not args.is_testing:
        train_inf = dataset_info(args.train_data, is_linux=IS_LINUX)
        args.input_dir = args.input_dir or train_inf['data_dir']
        args.train_list = args.train_list or train_inf['train_list']
    
    test_inf = dataset_info(args.test_data, is_linux=IS_LINUX)

    args.input_val_dir = args.input_val_dir or test_inf['data_dir']
    args.test_list = args.test_list or test_inf['test_list']
    args.test_img_width = args.test_img_width or test_inf['img_width']
    args.test_img_height = args.test_img_height or test_inf['img_height']
    args.mean_test = args.mean_test or test_inf['mean']
    
    # Set mode based on flags
    if args.is_training:
        args.is_testing = False
    elif args.is_testing:
        args.is_testing = True
    
    return args, dataset_info(args.train_data, is_linux=IS_LINUX)


def main(args, train_inf):

    test_inf = dataset_info(args.test_data, is_linux=IS_LINUX)
    config = Config(args, train_inf, test_inf)
    config.print_device_info()


    # Instantiate model and move it to the computing device
    model = TED().to(config.device)

    if args.is_testing:
        print("\n" + "="*60)
        print("RUNNING IN TESTING MODE")
        print("="*60 + "\n")
        
        if args.test_data.upper() == 'SEISMIC':
            dataset_val = SeismicTestDataset(args.input_val_dir, test_list=args.test_list,
                                            img_height=args.test_img_height,
                                            img_width=args.test_img_width, arg=args)
        else:
            dataset_val = TestDataset(args.input_val_dir, test_data=args.test_data,
                                    img_width=args.test_img_width,
                                    img_height=args.test_img_height,
                                    test_list=args.test_list, arg=args)
            
        dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,
                                   num_workers=args.workers)
        
        if_resize_img = args.test_data not in ['BIPED', 'CID', 'MDBD']
        test(config.checkpoint_path, dataloader_val, model, config, if_resize_img)
        
        # Log test results
        log_path = Path(args.res_dir) / 'test_log.txt'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] "
                   f"{args.train_data} -> {args.test_data} | Checkpoint: {args.checkpoint_data}\n")
        
        num_params = count_parameters(model)
        print('='*60)
        print(f'TEED Parameters: {num_params}')
        print('='*60)
        return

    # Training mode
    print("\n" + "="*60)
    print("RUNNING IN TRAINING MODE")
    print("="*60 + "\n")
    
    print(f"DEBUG: train_data={args.train_data}, test_data={args.test_data}")
    print(f"DEBUG: input_val_dir={args.input_val_dir}, test_list={args.test_list}")
    
    
    # Load datasets

    # Load datasets
    if args.train_data.upper() == 'SEISMIC':
        dataset_train = SeismicTrainDataset(args.input_dir, train_list=args.train_list,
                                        img_height=args.img_height,
                                        img_width=args.img_width, arg=args)
    else:
        dataset_train = BipedDataset(args.input_dir, img_width=args.img_width,
                                    img_height=args.img_height, train_mode='train', crop_img=args.crop_img, arg=args)
    

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.workers)
    print(f"Training dataset: {len(dataset_train)} samples")
    
    if args.test_data.upper() == 'SEISMIC':
        dataset_val = SeismicTestDataset(args.input_val_dir, test_list=args.test_list,
                                        img_height=args.test_img_height,
                                        img_width=args.test_img_width, arg=args)
    else:
        dataset_val = TestDataset(args.input_val_dir, test_data=args.test_data,
                                img_width=args.test_img_width,
                                img_height=args.test_img_height,
                                test_list=args.test_list, arg=args)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,
                               num_workers=args.workers)
    
    # Setup training
    criterion = [cats_loss, bdcn_loss2]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    num_params = count_parameters(model)
    print('='*60)
    print(f'TEED Parameters: {num_params}')
    print('='*60)
    
    if_resize_img = args.test_data not in ['BIPED', 'CID', 'MDBD']
    
    # Training loop
    seed = 1021
    k = 0
    for epoch in range(args.epochs):
        # Reset random seed periodically
        if epoch % 5 == 0:
            seed += 1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            print("Random seed reset")
        
        # Adjust learning rate
        if epoch in args.adjust_lr:
            new_lr = args.lrs[k]
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate adjusted to {new_lr}")
            k += 1
        
        # Train
        avg_loss = train_one_epoch(epoch, dataloader_train, model, criterion,
                                  optimizer, config)
        
        # Validate
        validate_one_epoch(epoch, dataloader_val, model, config, if_resize_img)
        
        # Save checkpoint
        checkpoint_path = Path(config.args.output_dir) / config.args.train_data / str(epoch)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        save_path = checkpoint_path / f'{epoch}_model.pth'
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, save_path)
        
        # Log to TensorBoard
        if config.tb_writer is not None:
            config.tb_writer.add_scalar('loss', avg_loss, epoch + 1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} completed | Loss: {avg_loss:.4f} | LR: {current_lr}")
    
    if config.tb_writer is not None:
        config.tb_writer.close()
    
    print("\nTraining completed!")


if __name__ == '__main__':
    args, train_info = parse_args()
    main(args, train_info)