# ResNet18 Classification Project

This project implements **ResNet18** for image classification on the **PASCAL dataset** and related subsets.  
It supports multiple input modalities (`RGB`, `RGB_RESIZED`, `RAW`, `PACKED`) and allows flexible training and evaluation.  

---

## 📂 Project Structure
```
ResNet18/
├── dataset/ # Main dataset (e.g., PASCAL VOC)
├── models/ # Saved models (.pth)
│ └── Format: Network_Modality_acc_timestamp.pth
├── results/ # Logs, test results, or visualizations
├── subset/ # Small subset for quick experiments
│ ├── train_small/
│ ├── test_small/
│ └── val_small/
├── utils/ # Utility scripts (dataset preprocessing, etc.)
├── .gitignore # Ignored files/folders
├── config_test.json # Hyperparameter config for testing
├── model.py # ResNet18 network definition
├── train4types(no valid).py # Training script (default augmentation)
├── train4types(no valid)BW norm.py # Training script (with BW norm for RAW)
├── test4types.py # Evaluation script
└── ResNet18.png # Network diagram
```


---

## 🚀 Usage

### 1. Dataset Preparation
- Place the **PASCAL dataset** in the `dataset/` directory.  
- For testing or debugging, you can use the small subset under `subset/`.

### 2. Training
Run the training script depending on the modality:

```bash
# Train on RGB/RAW/RGB_RESIZED/PACKED images
python train4types(no valid).py
```
### 3. Testing
Run:
```bash
python test4types.py
```
The test configuration (batch size, epochs, LR, etc.) is defined in config_test.json.
### 4. Model
model.py implements ResNet18, including:

CommonBlock: standard residual block (identity shortcut).

SpecialBlock: downsampling block with 1×1 convolution shortcut.

The classifier head includes dropout for regularization.
## ⚙️ Requirements

Python 3.8+

PyTorch >= 1.12

torchvision

numpy

Pillow

visdom (optional, for visualization of loss/accuracy)
## 📈 Example Results

| Modality | LR   | Batch | Epochs | Weight Decay | Improvement     | Accuracy |
|----------|------|-------|--------|--------------|-----------------|----------|
| RAW      | 3e-4 | 32    | 50     | 0.05         | None            | 0.6956   |
| RAW      | 3e-4 | 32    | 100    | 0.05         | None            | 0.7168   |
| RAW      | 3e-4 | 32    | 100    | 0.05         | 50% Flip        | 0.7405   |
| RAW      | 3e-4 | 32    | 100    | 0.05         | BW Balance Norm | 0.7188   |

## 📌 Notes

.gitignore excludes datasets, model weights (*.pth), and cache files.

Empty directories (like dataset/, models/, results/) are tracked using .gitkeep.

The repository is intended for experimentation with different input modalities and training strategies.

## 🔮 Future Work

Add more data augmentation for RAW (e.g., random crop, color jitter).

Explore different optimizers and schedulers (e.g., cosine annealing).

Extend support to detection tasks.

