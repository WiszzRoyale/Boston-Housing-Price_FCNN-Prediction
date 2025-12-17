import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(123)

# Read data from file "housing.data": (13 features and 1 target value)
path = r"C:\Users\wisnu\OneDrive\Desktop\Experiment1\housing.data"
fg = open(path, "r", encoding="utf-8")

s = list(fg)
X, Y = [], []
for i, line in enumerate(s):
    line = line.replace('\n', '')
    line = line.split(' ')
    line2 = [float(v) for v in line if v.strip() != '']
    X.append(line2[:-1])  # Get feature vector
    Y.append(line2[-1])   # Get sample labels (housing prices)
fg.close()
X = torch.FloatTensor(X)  # torch.Size([506, 13]) torch.Size([506])
Y = torch.FloatTensor(Y)

index = torch.randperm(len(X))
X, Y = X[index], Y[index]  # Randomly shuffle the order

torch.manual_seed(124)

rate = 0.8
train_len = int(len(X) * rate)
trainX, trainY = X[:train_len], Y[:train_len]  # Training set
testX, testY = X[train_len:], Y[train_len:]    # Test set

# Training and test sets should generally be normalized separately, but using the same normalization method:
def map_minmax(T):  # Normalization function
    min, max = torch.min(T, dim=0)[0], torch.max(T, dim=0)[0]
    r = (1.0 * T - min) / (max - min)
    return r

trainX, trainY = map_minmax(trainX), map_minmax(trainY)
testX, testY = map_minmax(testX), map_minmax(testY)

#--------------------
batch_size = 16  # Set batch size

# Package the training set:
train_set = TensorDataset(trainX, trainY)
train_loader = DataLoader(dataset=train_set,   # Package
                          batch_size=batch_size,
                          shuffle=False)  # Default: shuffle=False

# Package the test set:
test_set = TensorDataset(testX, testY)
test_loader = DataLoader(dataset=test_set,
                         batch_size=batch_size,
                         shuffle=False)  # Default: shuffle=False

del X, Y, trainX, trainY, testX, testY, train_set, test_set

# Define class Model2_2
class Model2_2(nn.Module):
    def __init__(self):
        super(Model2_2, self).__init__()  #[The core code for this example is omitted here; the complete code can be found on page 80 of the textbook.
        # Readers are advised to manually type in the core code and debug it to fully understand its meaning.]
        # 13 input features → 512 neurons
        self.fc1 = nn.Linear(13, 512)
        # 512 → 1 output value
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # First fully connected layer + sigmoid
        out = self.fc1(x)
        out = torch.sigmoid(out)

        # Second fully connected layer + sigmoid
        out = self.fc2(out)  #[The core code for this example is omitted here; the complete code can be found on page 80 of the textbook. Readers are advised to manually type in the core code and debug it to fully understand its meaning.]
#         return out
        out = torch.sigmoid(out)

        return out

model2_2 = Model2_2()
optimizer = torch.optim.Adam(model2_2.parameters(), lr=0.01)  # lr=0.005

ls = []
for epoch in range(200):
    for i, (x, y) in enumerate(train_loader):  # Use the packaged training set for training
        pre_y = model2_2(x)  # pre_y shape: torch.Size([30, 1])
        pre_y = pre_y.squeeze()  # Change to torch.Size([30])

        loss = nn.MSELoss()(pre_y, y)  # Mean squared error loss function
        if i % 100 == 0:
            ls.append(loss.item())

        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Backward pass to compute gradients
        optimizer.step()       # Update parameters

# Start model testing below, calculate prediction accuracy:
lsy = torch.Tensor([])
ls = torch.Tensor([])
model2_2.eval()  # Set to evaluation mode
correct = 0
with torch.no_grad():  # torch.no_grad() is a context manager that disables gradient calculation
    for x, y in test_loader:
        pre_y = model2_2(x)  # torch.Size([16, 1])
        pre_y = pre_y.squeeze()
        t = (torch.abs(pre_y - y) < 0.1)
        t = t.long().sum()
        correct += t

        ls = torch.cat((ls, pre_y))
        lsy = torch.cat((lsy, y))

s = 'Prediction accuracy on test set: {:.1f}%'.format(100. * correct / len(test_loader.dataset))
print(s)

plt.plot(ls, label='Actual values')
plt.plot(lsy, label='Predicted values')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to display Chinese labels normally - SimHei
plt.xlabel("Sample point index", fontsize=16)  # X-axis label
plt.ylabel("Housing price (normalized)", fontsize=16)  # Y-axis label
plt.tick_params(labelsize=16)
plt.grid()
plt.legend()

plt.show()

exit(0)