# COVID-19 Chest X-Ray Classification Using PyTorch

A deep learning project to classify chest X-ray images into three categories: **Normal**, **Viral Pneumonia**, and **COVID-19** using ResNet18 architecture with PyTorch.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Results](#results)

---

## Project Overview

This project demonstrates a complete machine learning pipeline for medical image classification. The model uses transfer learning with a pre-trained ResNet18 architecture to classify chest X-ray images into three categories:

- **Normal**: Healthy chest X-rays
- **Viral**: Chest X-rays showing viral pneumonia
- **COVID-19**: Chest X-rays showing COVID-19 symptoms

The project leverages PyTorch's computational efficiency and includes data preprocessing, model training, validation, and visualization components.

---

## Project Structure

The notebook is divided into the following key parts:

### 1. **Imports and Setup** (Cell 1)
- Imports necessary libraries: PyTorch, torchvision, NumPy, PIL, and Matplotlib
- Sets random seeds for reproducibility
- Displays PyTorch version information

### 2. **Data Organization** (Cell 2)
- Organizes raw X-ray images into structured directories
- Renames folders to standardized class names: `normal`, `viral`, `covid`
- Creates a separate test directory with 30 images per class for evaluation
- Uses random sampling to split training and test data

### 3. **Custom Dataset Class** (Cell 3)
- Implements `ChestXRayDataset` class extending `torch.utils.data.Dataset`
- Loads images from class directories
- Handles image retrieval and random sampling
- Converts images to RGB format for compatibility
- Provides dataset length and item indexing

### 4. **Data Transforms** (Cell 4)
- **Training Transform**: Resizes images to 224×224, applies random horizontal flips, converts to tensors, and normalizes
- **Test Transform**: Resizes to 224×224, converts to tensors, and normalizes
- Uses ImageNet normalization standards (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 5. **Dataset Initialization** (Cells 5-6)
- Creates training dataset pointing to the main image directories
- Creates test dataset pointing to the test subdirectories
- Loads both datasets with the defined transformations

### 6. **DataLoader Setup** (Cell 7)
- Creates PyTorch DataLoaders for batching
- Batch size set to 6
- Enables shuffling for training data
- Provides dataset statistics (number of batches)

### 7. **Visualization** (Cells 8-10)
- Implements `show_images()` function to visualize X-ray samples
- Displays predicted vs actual labels
- Color-codes correct predictions (green) vs incorrect ones (red)
- Denormalizes images for proper visualization

### 8. **Model Architecture** (Cell 11)
- Uses pre-trained ResNet18 from torchvision
- Modifies final fully connected layer from 1000 to 3 outputs (one per class)
- Defines Cross-Entropy Loss function for multi-class classification
- Sets up Adam optimizer with learning rate 3e-5

### 9. **Training Loop** (Cell 15)
- Implements comprehensive training function with:
  - Epoch-based training
  - Validation at regular intervals (every 20 steps)
  - Loss calculation and backpropagation
  - Accuracy metrics computation
  - Early stopping when accuracy ≥ 95%
  - Training vs evaluation mode switching

### 10. **Execution and Prediction** (Cells 16-18)
- Runs training loop for specified number of epochs
- Visualizes predictions on test data before and after training
- Displays performance improvements over time

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip or conda package manager
- GPU recommended (optional, for faster training)

### Step 1: Clone the Repository
```bash
git clone https://github.com/EarnTHYPart/COVID-19-Chest-XRay.git
cd COVID-19-Chest-XRay
```

### Step 2: Create Virtual Environment (Recommended)

**Using venv:**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**Using conda:**
```bash
conda create -n covid-xray python=3.8
conda activate covid-xray
```

### Step 3: Install Dependencies
```bash
pip install torch torchvision matplotlib pillow numpy
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### Optional: GPU Support (CUDA)
For faster training on NVIDIA GPUs:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Dataset

### Download Dataset
The project uses the **COVID-19 Radiography Database** available on Kaggle.

**Download Link:** [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

### Dataset Structure
After downloading, extract the dataset into the project directory:

```
COVID-19-Chest-XRay/
├── COVID-19 Radiography Database/
│   ├── normal/          (NORMAL images)
│   ├── viral/           (Viral Pneumonia images)
│   ├── covid/           (COVID-19 images)
│   └── test/            (Test split)
│       ├── normal/
│       ├── viral/
│       └── covid/
├── Complete Notebook.ipynb
└── README.md
```

The notebook automatically reorganizes the downloaded data into the correct structure when first run.

---

## How to Run

### Running in Jupyter Notebook

1. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open the Notebook:**
   - Navigate to `Complete Notebook.ipynb`
   - Click to open

3. **Run Cells:**
   - Click on each cell and press `Shift+Enter` to execute
   - Or use `Kernel → Run All` to execute all cells
   - Cells run sequentially and depend on previous cells

### Running in JupyterLab

```bash
jupyter lab
```

### Running in VS Code with Jupyter Extension

1. Install the Jupyter extension in VS Code
2. Open `Complete Notebook.ipynb`
3. Click the Run button on each cell or use the Jupyter interface

### Training Configuration

To modify training parameters, edit these variables in the notebook:

```python
# Adjust batch size (Cell 7)
batch_size = 6  # Increase for faster training, decrease for lower memory usage

# Adjust learning rate (Cell 12)
optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)

# Adjust training epochs (Cell 16)
train(epochs=1)  # Increase for more training iterations
```

---

## Model Architecture

**ResNet18** is a 18-layer residual neural network:

- **Input:** 224×224 RGB images
- **Feature Extraction:** Multiple residual blocks with skip connections
- **Backbone:** Pre-trained on ImageNet
- **Output Layer:** 3 neurons (one per class)
- **Final Activation:** Softmax (applied by CrossEntropyLoss)

### Modified Architecture
```
ResNet18 (pretrained)
    └── Final FC Layer: 512 → 3 outputs
```

### Training Details
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=3e-5)
- **Batch Size:** 6
- **Validation Accuracy Target:** ≥ 95%

---

## Results

The model achieves:
- **High accuracy** in distinguishing between normal, viral pneumonia, and COVID-19 X-rays
- **Early stopping** at ≥95% accuracy for efficient training
- **Visualization** of predictions with color-coded correctness (green=correct, red=incorrect)

### Training Progress
- Validation loss and accuracy are evaluated every 20 training steps
- Sample predictions are visualized during training to monitor performance
- Training typically converges within 1 epoch due to transfer learning

---

## Usage Tips

- **First Run:** The first execution will reorganize the dataset. This is normal and only happens once.
- **GPU Memory:** If you run out of memory, reduce `batch_size` from 6 to 4 or 2
- **Training Time:** With GPU, training typically takes 10-30 minutes per epoch
- **Predictions:** Use `show_preds()` function anytime to visualize model predictions on test data

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Dataset not found | Ensure the `COVID-19 Radiography Database` folder is extracted in the project directory |
| Out of memory | Reduce batch size in Cell 7 |
| Slow training | Enable GPU support or reduce dataset size |
| Import errors | Run `pip install -r requirements.txt` |

---

## License

This project is provided as-is for educational and research purposes.

---

## Citation

If you use this project, please cite the original dataset:

**Rahman, T., Chowdhury, A., Khandakar, A.** (2021). COVID-19 Radiography Database. Mendeley Data.

---

## Contact & Support

For issues, questions, or contributions, please open an issue on the [GitHub repository](https://github.com/EarnTHYPart/COVID-19-Chest-XRay).

---

**Happy Learning!** 🚀

---

## Model Saving & Evaluation

New convenience cells have been added at the end of the notebook:

- **Evaluation Metrics:** Computes overall test accuracy, a confusion matrix, and per-class precision/recall. It will also render a confusion matrix plot for quick visual inspection.
- **Model Save/Load + TorchScript:** Saves the trained `resnet18` weights to `models/resnet18_covid_xray.pth`, demonstrates loading them into a fresh model, and optionally exports a TorchScript module to `models/resnet18_covid_xray_script.pt` for deployment.

### How to Use
- Run all training cells as usual.
- Execute the new "Evaluation" cell to see metrics and the confusion matrix.
- Execute the "Save and Load Model" cell to persist weights and produce a TorchScript artifact.

Artifacts will be written under the `models/` folder, which is created automatically if it doesn't exist.
