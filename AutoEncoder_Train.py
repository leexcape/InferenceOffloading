from AutoEncoder import AutoEncoder
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from DNN import DNN
import numpy as np

BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCH = 100
PP = 5
dnn_layer_size = np.array((784, 2048, 1024, 512, 512, 128))
DATA_DIR = '/home/lixiangchen/MyWorkspace/InferenceOffloading/data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Preparing Data
trainData = dsets.MNIST(DATA_DIR, train=True, transform=transforms.ToTensor(), download=True)
testData = dsets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)
trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

# Prepare Model
dnn = DNN()
dnn.load_state_dict(torch.load('DNN_MNIST.pkl', map_location=device))
dnn.to(device)
dnn.eval()
ae = AutoEncoder(dnn_layer_size[PP])
ae.to(device)

# Test initial accuracy
correct_total_test = 0
dnn.eval()
for images, labels in testLoader:
    images = images.view(images.shape[0], -1).to(device)
    outputs = dnn(images)
    _, predicted = torch.max(outputs.data, 1)
    correct_total_test += (predicted.cpu() == labels).sum().numpy()
initial_acc = correct_total_test / 10000

# Training
Loss_Rec = []
acc_rec = []
for epoch in range(EPOCH):
    batch_counter = 0
    loss_sum = 0
    for images, labels in trainLoader:
        batch_counter += 1
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)
        feature = dnn.partial_forward_1(images, PP)
        loss = ae.learn(LR=LEARNING_RATE, feature_matrix=feature, label=feature)
        loss_sum += loss.cpu().detach().numpy()
    Loss_Rec.append(loss_sum / batch_counter)
    # Test overall accuracy
    correct_total_test = 0
    for images, labels in testLoader:
        images = images.view(images.shape[0], -1).to(device)
        feature = dnn.partial_forward_1(images, PP)
        feature = ae.forward(feature)
        outputs = dnn.partial_forward_2(feature, PP)
        _, predicted = torch.max(outputs.data, 1)
        correct_total_test += (predicted.cpu() == labels).sum().numpy()
    print("Epoch %i avg acc after compression: %f | Initial accuracy: %f" % (epoch + 1, correct_total_test / 10000, initial_acc))
    acc_rec.append(correct_total_test / 10000)

torch.save(ae.state_dict(), f'AutoEncoder_layer{PP}.pkl')
plt.figure()
plt.plot(Loss_Rec)
plt.xlabel('loss value record')
# plt.legend()
plt.figure()
plt.plot(acc_rec)
plt.xlabel('overall accuracy record')
# plt.legend()
plt.show()


