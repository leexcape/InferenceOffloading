import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from DNN import DNN

# Hyperparameters and Settings
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCH = 100
N_CLASSES = 10
# DATA_DIR = 'D:/SynologyDrive/coding project/InferenceOffloading/data'
DATA_DIR = '/home/lixiangchen/MyWorkspace/InferenceOffloading/data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Preparing Data
trainData = dsets.MNIST(DATA_DIR, train=True, transform=transforms.ToTensor(), download=True)
testData = dsets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)

trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

# Generate Model
dnn = DNN()
dnn.to(device)

# Loss, Optimizer & Scheduler
acc_train = []
acc_test = []
loss_train = []

# Train the model
for epoch in range(EPOCH):
    avg_loss = 0
    cnt = 0
    correct_total_train = 0
    correct_total_test = 0
    # Training
    dnn.train()
    for images, labels in trainLoader:
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)
        # Learning
        avg_loss += dnn.learn(LR=LEARNING_RATE, feature_matrix=images, label=labels)
        cnt += 1
        # Training Accuracy
        outputs = dnn(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_total_train += (predicted.cpu() == labels.cpu()).sum().numpy()

    # Testing Accuracy
    dnn.eval()
    for images, labels in testLoader:
        images = images.view(images.shape[0], -1).to(device)
        outputs = dnn(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_total_test += (predicted.cpu() == labels).sum().numpy()

    # Printing Results and Save Curves
    print("[E: %d] Testing acc: %f" % (epoch, correct_total_test / 10000))
    print("[E: %d] Average loss: %f" % (epoch, avg_loss / cnt))
    print("[E: %d] Training acc: %f" % (epoch, correct_total_train / 60000))
    acc_train.append(correct_total_train / 60000)
    acc_test.append(correct_total_test / 10000)
    loss_train.append(avg_loss / cnt)

# Plot acc curves
plt.figure()
plt.plot(acc_train, color='cyan', label='Training Acc')
plt.plot(acc_test, color='b', label='Validation Acc')
plt.legend()

plt.figure()
plt.plot(loss_train, label='Training Loss')
plt.legend()
plt.show()

# Save the Trained Model
torch.save(dnn.state_dict(), 'DNN_MNIST.pkl')
