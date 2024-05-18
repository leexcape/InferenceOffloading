import torch
import numpy as np
from DNN import DNN
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
from AutoEncoder import AutoEncoder

DATA_DIR = '/home/lixiangchen/MyWorkspace/InferenceOffloading/data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dnn = DNN()
dnn.load_state_dict(torch.load('./check points/DNN_MNIST.pkl', map_location=device))
dnn.to(device)
dnn.eval()
testData = dsets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testData, batch_size=50, shuffle=False)
gamma_device = 5   # number of clock cycles needed for a MAC operation at device
gamma_server = 5 / 4  # number of clock cycles needed for a MAC operation at server
f_server = 3e9  # clock rate of server processor
f_device = 0.20e9  # clock rate of mobile device processor
kappa_device = 5 / (f_device ** 3)     # power factor of mobile device
kappa_server = 200 / (f_server ** 3)   # power factor of server
pi_device = 1     # The transmit power of mobile device
channel_capacity = 0.2e9  # Transmit rate of mobile decive
omega = 1e9   # Time consumption factor
tau = 1e9     # Energy consumption factor


def ae_task(pp):
    o = []  # computation cost
    o_ae = []   # AutoEncoder computation cost
    weight_size = []
    bias_size = []
    dnn_layer_size = np.array((784, 2048, 1024, 512, 512, 128))
    ae = AutoEncoder(dnn_layer_size[pp])
    ae.load_state_dict(torch.load(f'./check points/AutoEncoder_layer{pp}.pkl', map_location=device))
    ae.to(device)
    ae.eval()

    # AutoEncoder Computation Cost
    layer_count = 0
    with torch.no_grad():
        for layer in ae.modules():
            if isinstance(layer, torch.nn.Linear):
                layer_count += 1
                o_ae.append(np.array(list(layer.weight.size())).prod())

    # Computation Cost
    layer_count = 0
    with torch.no_grad():
        for layer in dnn.modules():
            if isinstance(layer, torch.nn.Linear):
                layer_count += 1
                o.append(np.array(list(layer.weight.size())).prod())
                weight_size.append(list(layer.weight.size()))
                bias_size.append(list(layer.bias.size()))

    # Calculate model and activation size
    layer_count = 0
    weight_size = []
    bias_size = []
    activation_size = []
    with torch.no_grad():
        for layer in dnn.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                layer_count += 1
                weight_size.append(list(layer.weight.size()))
                bias_size.append(list(layer.bias.size()))
    for images, labels in testLoader:
        activation_size = dnn.activation_size(images[0].view(1, -1).to(device))
        break
    z = np.array(weight_size).prod(axis=1) + np.array(bias_size).prod(axis=1)  # layer-wise parameter size

    # Accuracy Test
    correct_total_test = 0
    for images, labels in testLoader:
        images = images.view(images.shape[0], -1).to(device)
        feature = dnn.partial_forward_1(images, pp)
        feature = ae.forward(feature)
        outputs = dnn.partial_forward_2(feature, pp)
        _, predicted = torch.max(outputs.data, 1)
        correct_total_test += (predicted.cpu() == labels).sum().numpy()
    acc = correct_total_test / 10000

    # Communication Payload
    model_weight_payload = sum([np.array(weight_size[pp + i]).prod() * 32 for i in range(layer_count - pp)])
    model_bias_payload = sum([np.array(bias_size[pp + i]).prod() * 32 for i in range(layer_count - pp)])
    model_payload = model_bias_payload + model_weight_payload
    activation_payload = 64 * 32
    communication_payload = model_payload + activation_payload

    # Time & Energy Consumption
    T_device = (sum([o[i] for i in range(pp)]) + sum([o_ae[i] for i in range(4)])) * gamma_device / f_device
    E_device = kappa_device * (f_device ** 2) * (sum([o[i] for i in range(pp)]) + sum([o_ae[i] for i in range(4)])) * gamma_device
    T_server = (sum([o[i] for i in range(pp, layer_count, 1)]) + sum([o_ae[i] for i in range(4)])) * gamma_server / f_server
    E_server = kappa_server * (f_server ** 2) * (sum([o[i] for i in range(pp, layer_count, 1)]) + sum([o_ae[i] for i in range(4)])) * gamma_server
    T_trans = communication_payload / channel_capacity
    E_trans = T_trans * pi_device
    cost = omega * (T_server + T_device + T_trans) + tau * (E_trans + 0 * E_server + E_device)
    return T_device, T_server, T_trans, E_device, E_server, E_trans, cost, acc, communication_payload


