import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

torch.manual_seed(100)
np.random.seed(100)

# Load dataset "housing.data" (13 features and 1 target value)
path = r"C:\Users\wisnu\OneDrive\Desktop\Experiment1\housing.data"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Parsing data
X, Y = [], []
for line in lines:
    vals = [float(v) for v in line.strip().split() if v.strip() != ""]
    X.append(vals[:-1])  #First 13 values
    Y.append(vals[-1])  #Last value/ target value

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

# Shuffle, Split, and Normalize
idx = np.random.permutation(len(X))
X, Y = X[idx], Y[idx]

rate = 0.8  #Split data into 80% training
train_len = int(len(X) * rate)
trainX, testX = X[:train_len], X[train_len:]
trainY, testY = Y[:train_len], Y[train_len:]

#Normalize features (only X)
mean, std = trainX.mean(axis=0), trainX.std(axis=0)
std[std == 0] = 1
trainX = (trainX - mean) / std
testX = (testX - mean) / std

# Convert to tensors for PyTorch for NN training
trainX_t = torch.FloatTensor(trainX)
trainY_t = torch.FloatTensor(trainY)
testX_t = torch.FloatTensor(testX)
testY_t = torch.FloatTensor(testY)

#Create DataLoader for batching
batch_size = 16
train_set = TensorDataset(trainX_t, trainY_t)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

# Define Neural Network Model
class Model_EnsembleNN(nn.Module):
    def __init__(self):
        super(Model_EnsembleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 192),  #Input layer
            nn.BatchNorm1d(192),  #Normalize activation, stable
            nn.ReLU(),  #Activation for non-linearity
            nn.Linear(192, 128),  #Hidden layer 1
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),  #Hidden layer 2
            nn.ReLU(),
            nn.Linear(64, 32),  #Hidden layer 3
            nn.ReLU(),
            nn.Linear(32, 1)  #Output (1 value, the house price)
        )

    def forward(self, x):
        return self.net(x)
#Initialize model, loss function and optimizer
model_ens = Model_EnsembleNN()
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model_ens.parameters(), lr=0.001, weight_decay=1e-4)
#Adam with weight decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.7)

# Train Neural Network (400 epochs)
for epoch in range(400):
    model_ens.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()  #Clear gradients
        pred = model_ens(xb).squeeze()  #Forward pass: to get prediction
        loss = criterion(pred, yb)  #calculate loss
        loss.backward()  #Backward pass: compute gradients
        optimizer.step()  #Update model parameter
        total_loss += loss.item()
    scheduler.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")

# Predictions from NN
model_ens.eval()
with torch.no_grad():
    pred_train_nn = model_ens(trainX_t).squeeze().numpy()
    pred_test_nn = model_ens(testX_t).squeeze().numpy()

# Train Random Forest & GBM
rf_model = RandomForestRegressor(n_estimators=200, random_state=123)
gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=123)

rf_model.fit(trainX, trainY)  #Train Random Forest on original features
gb_model.fit(trainX, trainY)  #Train Gradient Boosting

#Get predictions from all base models
rf_train = rf_model.predict(trainX)
gb_train = gb_model.predict(trainX)
rf_test = rf_model.predict(testX)
gb_test = gb_model.predict(testX)

# Meta-Model (Stacking / Ensemble)
stack_train = np.vstack([pred_train_nn, rf_train, gb_train]).T
stack_test = np.vstack([pred_test_nn, rf_test, gb_test]).T

meta_model = LinearRegression()
meta_model.fit(stack_train, trainY)
final_pred = meta_model.predict(stack_test)  #Final ensemble predictions

# Evaluation
threshold = 0.1 * (testY.max() - testY.min())  #10% of target range accuracy threshold
correct = np.sum(np.abs(final_pred - testY) < threshold)
accuracy = 100. * correct / len(testY)  #Accuracy
mae = mean_absolute_error(testY, final_pred)  #Mean Absolute error

print(f"\nFinal Ensemble Accuracy: {accuracy:.2f}%")
print(f"Mean Absolute Error: {mae:.4f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(testY, 'b-', label='Actual')
plt.plot(final_pred, 'r-', label='Ensemble Prediction')
plt.title(f"Hybrid Ensemble Model | Accuracy: {accuracy:.1f}%")
plt.xlabel("Sample Index")
plt.ylabel("House Price")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(testY, final_pred, alpha=0.7)
plt.plot([testY.min(), testY.max()], [testY.min(), testY.max()], 'k--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual (Ensemble)")
plt.grid(True)
plt.tight_layout()
plt.show()
