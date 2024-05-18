import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameter Definition
N_FEATURE_LEN = 784
N_LAYER1 = 2048
N_LAYER2 = 1024
# N_LAYER1 = 1024
# N_LAYER2 = 1024
N_LAYER3 = 512
N_LAYER4 = 512
N_LAYER5 = 128
N_LABEL = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(N_FEATURE_LEN, N_LAYER1)                # Define Classifier Layer 1
        self.layer1.weight.data.normal_(0, 0.1)                        # Normalize Classifier Layer 1
        self.layer2 = nn.Linear(N_LAYER1, N_LAYER2)                     # Define Classifier Layer 2
        self.layer2.weight.data.normal_(0, 0.1)                        # Normalize Classifier Layer 2
        self.layer3 = nn.Linear(N_LAYER2, N_LAYER3)                     # Define Classifier Layer 3
        self.layer3.weight.data.normal_(0, 0.1)                        # Normalize Classifier Layer 3
        self.layer4 = nn.Linear(N_LAYER3, N_LAYER4)  # Define Classifier Layer 3
        self.layer4.weight.data.normal_(0, 0.1)
        self.layer5 = nn.Linear(N_LAYER4, N_LAYER5)  # Define Classifier Layer 3
        self.layer5.weight.data.normal_(0, 0.1)
        # Normalize Classifier Layer 3
        self.layer6 = nn.Linear(N_LAYER5, N_LABEL)                         # Define Classifier Output Layer
        self.layer6.weight.data.normal_(0, 0.1)                           # Normalize Classifier Output Layer

        self.Loss_Function = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.Loss_Function = self.Loss_Function.cuda()
        self.dropout = nn.Dropout(p=0.2)
        # self.bn = nn.BatchNorm1d()

    def forward(self, x):
        # x = torch.FloatTensor(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = self.dropout(x)
        x = F.relu(x)
        x_value = self.layer6(x)
        return x_value

    def partial_forward_1(self, x, layer_num):
        # x = self.layer1(x)
        for i in range(layer_num):
            # print(f'x = self.layer{i + 1}(x)')
            x = eval(f'self.layer{i + 1}(x)')
            if i < 5:
                x = self.dropout(x)
                x = F.relu(x)
        return x

    def partial_forward_2(self, x, layer_num):
        for i in range(layer_num, 6, 1):
            x = eval(f'self.layer{i + 1}(x)')
            if i < 5:
                x = self.dropout(x)
                x = F.relu(x)
        return x

    def detect(self, x):
        x_value = self.forward(x)
        # detect_result = torch.max(x_value, 0)[1].data.numpy()
        # detect_result = detect_result[0]
        return x_value

    def learn(self, LR, feature_matrix, label):
        detect_result = self.forward(feature_matrix)
        # detect_result = torch.sigmoid(detect_result)
        # detect_result = torch.max(detect_result, 0)[1]
        loss = self.Loss_Function(detect_result, label)
        # loss = self.Loss_Function(detect_result, torch.FloatTensor(label))
        torch.optim.Adam(self.parameters(), lr=LR).zero_grad()
        loss.backward()
        torch.optim.Adam(self.parameters(), lr=LR).step()
        return loss

    def quantize(self, bitwidth, quantization_flag):
        layer_count = 0
        # torch.set_grad_enabled(False)
        with torch.no_grad():
            for layer in self.modules():
                if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                    layer_count += 1
                    if quantization_flag[layer_count - 1]:
                        weights_max = torch.max(layer.weight)
                        weights_min = torch.min(layer.weight)
                        quantization_interval_w = (weights_max - weights_min) / (2 ** bitwidth[layer_count - 1] - 1)
                        weights_q = torch.round((layer.weight - weights_min) / quantization_interval_w) * quantization_interval_w + weights_min
                        # xx = np.linspace(weights_min.cpu().numpy(), weights_max.cpu().numpy(), 2**bitwidth)
                        layer.weight = torch.nn.Parameter(data=weights_q, requires_grad=False)

                        bias_max = torch.max(layer.bias)
                        bias_min = torch.min(layer.bias)
                        quantization_interval_b = (bias_max - bias_min) / (2 ** bitwidth[layer_count - 1] - 1)
                        bias_q = torch.round((layer.bias - bias_min) // quantization_interval_b) * quantization_interval_b + bias_min
                        layer.bias = torch.nn.Parameter(data=bias_q, requires_grad=False)

    def add_noise(self, layer_idx, snr):
        layer_count = 0
        with torch.no_grad():
            for layer in self.modules():
                if isinstance(layer, torch.nn.Linear):
                    if layer_count == layer_idx:
                        noise_power_weight = torch.norm(layer.weight, p=2) ** 2 / (layer.weight.size()[0] * layer.weight.size()[1]) * snr
                        noise_weight = np.random.normal(0, noise_power_weight.cpu() ** 0.5, size=layer.weight.size())
                        layer.weight = torch.nn.Parameter(data=layer.weight + torch.from_numpy(noise_weight).to(torch.float32).to(device), requires_grad=False)
                        noise_power_bias = torch.norm(layer.bias, p=2) ** 2 / layer.bias.size()[0] * snr
                        noise_bias = np.random.normal(0, noise_power_bias.cpu() ** 0.5, size=layer.bias.size())
                        layer.bias = torch.nn.Parameter(data=layer.bias + torch.from_numpy(noise_bias).to(torch.float32).to(device), requires_grad=False)
                    layer_count += 1
        return noise_weight, noise_bias

    def activation_size(self, x):
        x_size = []
        x = self.layer1(x)
        x_size.append(list(x.size()))
        x = self.layer2(x)
        x_size.append(list(x.size()))
        x = self.layer3(x)
        x_size.append(list(x.size()))
        x = self.layer4(x)
        x_size.append(list(x.size()))
        x = self.layer5(x)
        x_size.append(list(x.size()))
        return x_size


