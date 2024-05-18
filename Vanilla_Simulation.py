import torch
import numpy as np
from DNN import DNN
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_DIR = '/home/lixiangchen/MyWorkspace/InferenceOffloading/data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
testData = dsets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testData, batch_size=32, shuffle=False)
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

def vanilla_task(pp):
    activation_size = 0
    o = []  # computation cost
    weight_size = []
    bias_size = []
    dnn = DNN()
    dnn.load_state_dict(torch.load('./check points/DNN_MNIST.pkl', map_location=device))
    dnn.to(device)
    dnn.eval()

    # Computation Cost
    layer_count = 0
    with torch.no_grad():
        for layer in dnn.modules():
            if isinstance(layer, torch.nn.Linear):
                layer_count += 1
                o.append(np.array(list(layer.weight.size())).prod())
                weight_size.append(list(layer.weight.size()))
                bias_size.append(list(layer.bias.size()))

    # Test acc for pruned model
    correct_total_test = 0
    for images, labels in testLoader:
        images = images.view(images.shape[0], -1).to(device)
        outputs = dnn(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_total_test += (predicted.cpu() == labels).sum().numpy()
    acc_pruning = correct_total_test / 10000

    # Activation Size
    for images, labels in testLoader:
        activation_size = dnn.activation_size(images[0].view(1, -1).to(device))
        break

    # Time consumption & Energy Consumption
    activation_payload = np.array(activation_size[pp - 1]).prod() * 32
    model_weight_payload = sum([np.array(weight_size[pp + i]).prod() * 32 for i in range(layer_count - pp)])
    model_bias_payload = sum([np.array(bias_size[pp + i]).prod() * 32 for i in range(layer_count - pp)])
    model_payload = model_bias_payload + model_weight_payload
    T_device = sum([o[i] for i in range(pp)]) * gamma_device / f_device
    E_device = kappa_device * (f_device ** 2) * sum([o[i] for i in range(pp)]) * gamma_device
    T_server = sum([o[i] for i in range(pp, layer_count, 1)]) * gamma_server / f_server
    E_server = kappa_server * (f_server ** 2) * sum([o[i] for i in range(pp, layer_count, 1)]) * gamma_server
    T_trans = (activation_payload + model_payload) / channel_capacity
    E_trans = T_trans * pi_device
    cost = omega * (T_server + T_device + T_trans) + tau * (E_trans + 0 * E_server + E_device)
    return T_device, T_server, T_trans, E_device, E_server, E_trans, cost, acc_pruning, activation_payload + model_payload

