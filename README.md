# Boston Housing Price Prediction using Deep Learning and Ensemble Methods

## Overview
This project focuses on predicting housing prices using the Boston Housing dataset.
Starting from a basic Fully Connected Neural Network (FCNN) implemented in PyTorch,
the model is progressively improved through architectural changes, optimization strategies,
regularization, and finally combined with traditional machine learning models using ensemble learning.

The final hybrid ensemble model achieves ~92% accuracy, outperforming the baseline neural network.

---

## Experiment Aim
- Build an accurate housing price prediction model
- Analyze the impact of:
  - Network depth
  - Optimizers and learning rate scheduling
  - Regularization techniques
  - Ensemble learning (NN + Random Forest + Gradient Boosting)
- Achieve prediction accuracy beyond 90%

---

## Dataset
- **Boston Housing Dataset**
- 506 samples
- 13 input features (crime rate, number of rooms, tax rate, etc.)
- 1 target variable: median house price

Preprocessing:
- Random shuffle
- 80% training / 20% testing split
- Feature normalization applied to input features only

---

## Methodology

### Baseline Model
- Simple 2-layer FCNN
- Sigmoid activation
- MSE loss
- Accuracy: ~86%

### Improved Neural Network
- Deeper architecture
- ReLU activation
- Batch Normalization
- AdamW optimizer with weight decay
- Learning rate scheduler
- SmoothL1 loss

### Final Ensemble Model
- Neural Network (PyTorch)
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression as meta-model (stacking)

---

## Results

Accuracy 91.18%
Mean Absolute Error (MAE) 2.0251
Training Epochs 400
Final Loss 1.0596 

### Visualization
- Actual vs Predicted Prices
- Scatter plot (Predicted vs Actual)
- Residual pattern analysis

### Output
Please check on the "Result" folder for more detail outputs

---

## Analysis Highlights
- Simple neural networks underfit due to limited capacity and sigmoid activation
- Batch normalization and ReLU significantly stabilize training
- AdamW optimizer improves generalization
- Ensemble learning benefits from model diversity and error correction
- Normalizing input features only (X) yields better performance than normalizing both X and Y

---

## Repository Structure
Code/        - Training and ensemble code
Dataset/       - Dataset
Result/    - Output plots
