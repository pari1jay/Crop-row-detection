# 🌾 Crop Row Detection using U-Net

This project implements a deep learning-based **semantic segmentation** model to detect **crop rows** in agricultural field images. It uses a modified **U-Net architecture** trained on annotated crop row images to identify crop rows with semantic precision.

---

## ✅ Overview

Traditional image processing techniques struggle with occlusions, lighting variation, and background clutter in field imagery. This project leverages **supervised deep learning** with **U-Net** to semantically segment crop rows from RGB images.

---

## 📁 Dataset

- Total Images: 281
- Image Size: 320 × 240 (RGB)
- Label Format: Grayscale masks with pixel values {0, 255}
- Folders:
  - `images/` - RGB input images
  - `train_labels/` - Ground truth segmentation masks for training
  - `test_labels/` - Ground truth masks for testing (optional)
- Data Augmentation: Includes random flips, rotations, brightness, and channel shift

---

## 🧠 Model Architecture

- **Base Model**: U-Net (encoder-decoder)
- **Activation**: ReLU
- **Final Layer**: Sigmoid (for binary classification)
- **Loss Function**: Binary Cross Entropy (BCE)
- **Optimizer**: Adam
- **Skip Connections**: Used for preserving spatial resolution

You may optionally use a lightweight variant (SqueezeUNet) for faster training on limited hardware.

---

## 🏋️ Training

- Epochs: 25
- Batch Size: 4
- Validation Split: 10%
- Early Stopping: Enabled (patience = 3)
- Model Checkpoint: Saves best model as `model_for_nuclei.h5`
- TensorBoard: Logs training under `/tensorboard/`

---

## 📊 Evaluation

- **Metric**: Mean Intersection over Union (mIoU)
- **Thresholding**: Prediction probabilities are thresholded at 0.25
- **Post-processing**:
  - Binary mask generation
  - Clustering for crop row shape extraction
- **Encoding**:
  - Run-Length Encoding (RLE) for mask compression

---

## 🖼️ Results

- **mIoU Achieved**: 0.21
- Moderate performance due to:
  - Small dataset size
  - High variability in lighting/backgrounds
  - Possible label noise

Example predictions and masks can be visualized in the `results/` directory (optional).

---

## 🚀 How to Run

```bash
# Step 1: Clone this repository
git clone https://github.com/yourusername/crop-row-detection.git
cd crop-row-detection

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Preprocess and augment data
python augment_images.py

# Step 4: Train the model
python train_model.py

# Step 5: Evaluate on test data
python evaluate_model.py

# Optional: Visualize predictions
python visualize_predictions.py
