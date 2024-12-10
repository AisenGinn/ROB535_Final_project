import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchmetrics import JaccardIndex
from dinov2_semantic_model import Dinov2SegmentationModel
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os

input_size = (560, 1120)
weight_factor = 5

def save_loader_sample(data_loader, output_dir, num_samples=3):
    """Save a few samples from the data loader."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    
    data_iter = iter(data_loader)
    for i in range(num_samples):
        images, masks = next(data_iter)  # Get a batch of images and masks
        # Select the first image and mask in the batch
        image = images[0]
        mask = masks[0]
        
        # Convert image tensor to PIL Image
        image = TF.to_pil_image(image)
        
        # Define file paths for saving
        image_path = os.path.join(output_dir, f"sample_{i}_image.png")
        mask_path = os.path.join(output_dir, f"sample_{i}_mask.png")
        
        # Save the image and mask
        image.save(image_path)
        mask_image = TF.to_pil_image(mask.byte())  # Convert mask to uint8
        mask_image.save(mask_path)
        
        print(f"Saved image: {image_path}")
        print(f"Saved mask: {mask_path}")

def mask_transform(mask):
    """
    Converts a PIL image mask to a PyTorch tensor using the trainId mapping.
    """
    """
    Converts a PIL image mask to a PyTorch tensor without remapping.
    """
    mask_array = np.array(mask, dtype=np.int64)  # Convert mask to NumPy array
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # Add channel dimension
    resized_mask = TF.resize(mask_tensor, size=input_size, interpolation=TF.InterpolationMode.NEAREST)
    return resized_mask.squeeze(0).long()  # Remove channel dimension and ensure it's long

def train_one_epoch(model, loader, optimizer, criterion, device, metric, weight_factor=1.0):
    model.train()
    running_loss = 0.0
    total_iou = 0.0
    
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

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
            valid_mask = (masks != -1)  # Create a mask to exclude -1
            preds = preds[valid_mask]
            masks = masks[valid_mask]
            iou_score += metric(preds, masks).item()

    return running_loss / len(loader.dataset), iou_score / len(loader)

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # Show full array
    torch.set_printoptions(profile="full")  # Disable summarization for tensors

    # Hyperparameters
    num_classes = 35  # Cityscapes has 19 classes, adds additional class for ignore objects
    batch_size = 4
    learning_rate = 5e-5
    num_epochs = 200
    checkpoint_dir = "cityscape_result/checkpoints"  # Directory to save checkpoints
    figures_dir = "cityscape_result/figures"  # Directory to save figures
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Data preprocessing and augmentation
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Cityscapes dataset
    root_dir = '/home/ubuntu/cityscapes_dataset'  # Update this to the correct path
    train_dataset = Cityscapes(
        root=root_dir,
        split='train',
        mode='fine',
        target_type='semantic',
        transform=transform,
        target_transform=mask_transform
    )

    val_dataset = Cityscapes(
        root=root_dir,
        split='val',
        mode='fine',
        target_type='semantic',
        transform=transform,
        target_transform=mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    
    # # Save transformed masks for inspection
    # os.makedirs("debug_masks", exist_ok=True)
    # for i in range(3):
    #     _, mask = train_dataset[i]  # Load a sample
    #     transformed_mask = mask_transform(mask)
    #     TF.to_pil_image(transformed_mask.byte()).save(f"debug_masks/mask_{i}.png")

    
    # # Save training samples
    # print("Saving samples from train_loader:")
    # save_loader_sample(train_loader, output_dir="./vis/train_samples", num_samples=3)

    # Save validation samples
    # print("Saving samples from val_loader:")
    # save_loader_sample(val_loader, output_dir="./vis/val_samples", num_samples=3)

    # Initialize the model
    model = Dinov2SegmentationModel(num_classes=num_classes).to(device)

    # Loss function and optimizer
    #criterion = IoULoss(num_classes=20, ignore_index=19)  # Ignore unlabeled pixels in Cityscapes
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define a StepLR scheduler to decay learning rate every 10 epochs
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Evaluation metric (mIoU)
    iou_metric = JaccardIndex(num_classes=num_classes, average='macro').to(device)

    # Training loop
    train_losses, val_losses, val_iou_scores = [], [], []
    for epoch in tqdm(range(1, num_epochs + 1)):  # Start from epoch 1
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device, iou_metric, weight_factor=weight_factor)
        val_loss, val_iou = validate(model, val_loader, criterion, iou_metric, device)

        #train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_iou_scores.append(val_iou)

        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}")
        
        # scheduler.step()

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