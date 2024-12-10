import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchmetrics import JaccardIndex
from dinov2_semantic_model import Dinov2SegmentationModel
from torchvision.transforms import functional as TF
from utils import IoULoss
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
from nuimages import NuImages
import sys
sys.path.append("..")
from nuimagesScripts.nuimages_dataset import NuImagesDataset
#from nuimagesScripts.nuimages_utils.engine import train_one_epoch, evaluate
from nuimagesScripts.nuimages_utils.model_utils import  get_transform, collate_fn

input_size = (560, 1120)
weight_factor = 5


def train_one_epoch(model, loader, optimizer, criterion, device, metric, weight_factor=1.0):
    model.train()
    running_loss = 0.0
    total_iou = 0.0
    
    for images, masks in tqdm(loader, desc="Training"): 
        images =  torch.stack(list(image for image in images)).to(device)
        masks = [list(t.values())[1] for t in masks]
        # print(masks[0].shape,masks[1].shape,masks[2].shape)
        # print(images[0].shape,images[1].shape,images[2].shape)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss and apply weight factor
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Calculate mIoU
        preds = outputs.argmax(dim=1)  # Get predicted class for each pixel
        total_iou += metric(preds, masks).item()
    
    # Average loss and mIoU over the entire dataset
    avg_loss = running_loss / len(loader.dataset)
    avg_iou = total_iou / len(loader)
    
    return avg_loss, avg_iou


def validate(model, loader, criterion, metric, device):
    model.eval()
    running_loss = 0.0
    iou_score = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            # Compute loss
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Compute IoU
            iou_score += metric(preds, masks).item()

    return running_loss / len(loader.dataset), iou_score / len(loader)


if __name__ == '__main__':
    nuimages = NuImages(dataroot="../nuimages_data/sets/nuimages", version="v1.0-train", verbose=True, lazy=False)
    nuimages_val = NuImages(dataroot="../nuimages_data/sets/nuimages", version="v1.0-val", verbose=True, lazy=False)
    transforms = get_transform(train=True)
    transforms_val = get_transform(train=False)
    dataset = NuImagesDataset(nuimages, transforms=transforms)
    dataset_val = NuImagesDataset(nuimages_val, transforms=transforms_val)

    print(f"{len(dataset)} training samples and {len(dataset_val)} val samples.")

    num_classes = len(nuimages.category) + 1  # add one for background class

    # Hyperparameters
    batch_size = 4
    learning_rate = 5e-5
    num_epochs = 200

    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4,
                                 collate_fn=collate_fn)
    



    checkpoint_dir = "nuimages_result/checkpoints"  # Directory to save checkpoints
    figures_dir = "nuimages_result/figures"  # Directory to save figures
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)



    # Initialize the model
    model = Dinov2SegmentationModel(num_classes=num_classes).to(device)

    # Loss function and optimizer
    criterion = IoULoss(num_classes=20, ignore_index=19)  # Ignore unlabeled pixels in Cityscapes
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # Evaluation metric (mIoU)
    iou_metric = JaccardIndex(num_classes=num_classes, average='macro', ignore_index=19).to(device)

    # Training loop
    train_losses, val_losses, val_iou_scores = [], [], []
    for epoch in tqdm(range(1, num_epochs + 1)):  # Start from epoch 1
        train_loss, train_iou = train_one_epoch(model, dataloader_train, optimizer, criterion, device, iou_metric, weight_factor=weight_factor)
        val_loss, val_iou = validate(model, dataloader_val, criterion, iou_metric, device)

        #train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_iou_scores.append(val_iou)

        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}")

        # Save checkpoint and figures every 5 epochs
        if epoch % 10 == 0 or epoch == num_epochs:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_iou{val_iou}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_iou_scores': val_iou_scores,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

            # Save training/validation loss figure
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            loss_fig_path = os.path.join(figures_dir, f"loss_epoch_{epoch}.jpg")
            plt.savefig(loss_fig_path)  # Save plot to a file
            plt.close()  # Close the plot to free memory
            print(f"Loss figure saved at {loss_fig_path}")

            # Save validation mIoU figure
            plt.figure(figsize=(10, 5))
            plt.plot(val_iou_scores, label='Validation mIoU')
            plt.xlabel('Epoch')
            plt.ylabel('mIoU')
            plt.legend()
            plt.title('Validation mIoU')
            iou_fig_path = os.path.join(figures_dir, f"miou_epoch_{epoch}.jpg")
            plt.savefig(iou_fig_path)  # Save plot to a file
            plt.close()  # Close the plot to free memory
            print(f"mIoU figure saved at {iou_fig_path}")