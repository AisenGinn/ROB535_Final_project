import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from dinov2_semantic_model import Dinov2SegmentationModel
import os
import numpy as np

# Input size and root directory for Cityscapes
input_size = (560, 1120)
root_dir = '/home/ubuntu/cityscapes_dataset'  # Update with the correct path

def mask_transform(mask):
    """
    Converts a PIL image mask to a PyTorch tensor.
    """
    mask_array = np.array(mask, dtype=np.int64)  # Convert mask to NumPy array
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # Add channel dimension
    resized_mask = TF.resize(mask_tensor, size=input_size, interpolation=TF.InterpolationMode.NEAREST)
    return resized_mask.squeeze(0).long()  # Remove channel dimension and ensure it's long

def test(model, loader, metric, device):
    """
    Evaluates the model on the test dataset.
    """
    model.eval()
    iou_score = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            # Compute IoU
            valid_mask = (masks != -1)  # Exclude ignored indices
            preds = preds[valid_mask]
            masks = masks[valid_mask]
            iou_score += metric(preds, masks).item()

    avg_iou = iou_score / len(loader)
    return avg_iou

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test dataset
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = Cityscapes(
        root=root_dir,
        split='test',
        mode='fine',
        target_type='semantic',
        transform=transform,
        target_transform=mask_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Load model and best checkpoint
    num_classes = 35
    model = Dinov2SegmentationModel(num_classes=num_classes).to(device)

    checkpoint_path = "/home/ubuntu/dinov2/cityscape_result/checkpoints/checkpoint_epoch_40_iou0.045723755806684495.pth"  # Replace with actual path
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Define evaluation metric
    iou_metric = JaccardIndex(num_classes=num_classes, average='macro').to(device)

    # Run the test evaluation
    test_iou = test(model, test_loader, iou_metric, device)
    print(f"Test mIoU: {test_iou:.4f}")
