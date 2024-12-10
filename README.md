# **Enhanced DINOv2**

This project builds upon the [DINOv2 repository](https://github.com/facebookresearch/dinov2) to extend its capabilities for tasks to **semantic segmentation** on datasets **Cityscapes** and **nuScenes**.

---

## **Features**
- **DINOv2 Integration**: Leverages pre-trained DINOv2 models.
- **Semantic Segmentation**: Extended for urban datasets.
- **Custom Metrics**: Includes mIoU evaluation.

---

## **Installation**
1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/your-username/ROB535_Final_project.git
   cd ROB535_Final_project
   pip install -r requirements.txt

2. (Optional) You can use pre-train segmentation head in [DINOv2 repository](https://github.com/facebookresearch/dinov2).

## **Usage**
### Dataset Setup
- **Cityscapes**:
  Download the dataset from [Cityscapes](https://www.cityscapes-dataset.com/) and change the path in cityscapes_train.py.
- **nuScenes**:
  Download the dataset from [nuScenes](https://www.nuscenes.org/) and change the path in nuimages_train.py.

### Training
To train the model on a specific dataset, use the following command (modify dataset_name to cityscapes/nuimages):
```bash
python dataset_name_train.py
```

---
## **Results**
The model's performance was evaluated on the Cityscapes and nuScenes test datasets, with results summarized below:

| Dataset      | mIoU (%) | Remarks            |
|--------------|----------|--------------------|
| Cityscapes   | 62.5     | Basic segmentation |
| nuScenes     | 55.5     | Limited training   |

### Qualitative Analysis
The qualitative results demonstrate that the model successfully identifies the edges of prominent objects such as lanes, vehicles, and buildings. However, the outputs are often noisy, with noticeable misclassifications within object boundaries. Future improvements can focus on enhancing feature representation and reducing noise for better segmentation accuracy.

---

## **Contributions**
This repository extends the original DINOv2 functionality by:
- Integrating **semantic segmentation tasks** for Cityscapes and nuScenes datasets.
- Adding evaluation metrics such as **mIoU** for performance measurement.
- Providing tools for visualization and qualitative analysis of segmentation outputs.

---

## **Acknowledgments**
- **Facebook Research** for the [DINOv2 model](https://github.com/facebookresearch/dinov2).
- **Cityscapes** and **nuScenes** teams for their datasets, essential for benchmarking segmentation tasks.
- The open-source community for their valuable contributions and tools.

---

## **License**
This project inherits the same license as the DINOv2 repository. Refer to the `LICENSE` file for detailed terms and conditions.


