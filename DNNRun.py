import torch
import numpy as np
from DNN import DNN
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

DATA_DIR = '/home/lixiangchen/MyWorkspace/InferenceOffloading/data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
testData = dsets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testData, batch_size=32, shuffle=False)

dnn = DNN()
dnn.load_state_dict(torch.load('./check points/DNN_MNIST.pkl', map_location=device))
dnn.to(device)
dnn.eval()
time_start_1 = 0
time_end_1 = 0
print('Test start!')
correct_total_test = 0
time_start = time.time()
for images, labels in tqdm(testLoader):
    images = images.view(images.shape[0], -1).to(device)
    time_start_1 = time.time()
    outputs = dnn(images)
    time_end_1 = time.time()
    _, predicted = torch.max(outputs.data, 1)
    correct_total_test += (predicted.cpu() == labels).sum().numpy()
acc_pruning = correct_total_test / 10000

print('Test done!, Accuracy is: ', acc_pruning)
print('Time consumption for Accuracy Test: ', time.time() - time_start, 's')
print('Time consumption for single forwarding: ', time_end_1 - time_start_1, 's')


