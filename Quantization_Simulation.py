import torch
import numpy as np
from DNN import DNN
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from overlapped_bar import overlapped_bar
import math
from scipy.io import savemat


def q_task(pp, acc_deg):   # range from 0 to (number of layer - 1), layer PARTITION_POINT is transmitted to remote server
    # Parameters
    DATA_DIR = '/home/lixiangchen/MyWorkspace/InferenceOffloading/data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    MODEL = 'DNN_MNIST'
    # QUANTIZATION_BITWIDTH = 0
    layers_number = 6
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
    snr_start = 0.001
    snr_step_size = 0.2
    # snr_step_size = 0.02
    # acc_deg = 0.03
    b = np.ones(layers_number) * 32      # Layer-wise quantization
    t = np.zeros(layers_number)
    p = np.zeros(layers_number)
    o = []  # computation cost

    # Prepare data
    testData = dsets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)
    testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

    # Prepare model
    model = DNN()
    model.load_state_dict(torch.load('./check points/DNN_MNIST.pkl', map_location=device))
    model.to(device)
    model.eval()
    model_target = DNN()
    model_target.load_state_dict(torch.load('./check points/DNN_MNIST.pkl', map_location=device))
    model_target.to(device)
    model_target.eval()

    # Calculate model and activation size
    layer_count = 0
    weight_size = []
    bias_size = []
    activation_size = []
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                layer_count += 1
                weight_size.append(list(layer.weight.size()))
                bias_size.append(list(layer.bias.size()))
    for images, labels in testLoader:
        activation_size = model.activation_size(images[0].view(1, -1).to(device))
        break
    z = np.array(weight_size).prod(axis=1) + np.array(bias_size).prod(axis=1)  # layer-wise parameter size

    # Calculate computation cost o(l)
    layer_count = 0
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer_count += 1
                o.append(np.array(list(layer.weight.size())).prod() * np.array(activation_size[layer_count - 1]).prod())
            if isinstance(layer, torch.nn.Linear):
                layer_count += 1
                o.append(np.array(list(layer.weight.size())).prod())

    # Calculate ALPHA and BETA
    ALPHA = (omega * gamma_device / f_device) + tau * gamma_device * kappa_device * (f_device ** 2)
    BETA = (omega * gamma_server / f_server) + tau * gamma_server * kappa_server * (f_server ** 2)
    SIGMA = (omega + pi_device * tau) / channel_capacity

    # Calculate b_p
    x1 = ALPHA * o[pp]
    x2 = BETA * o[pp]
    x3 = (np.array(weight_size).prod(axis=1)[pp] + np.array(bias_size).prod(axis=1)[pp]) / math.log(4)
    x4 = SIGMA * z[pp]
    b[pp] = (ALPHA * o[pp] - BETA * o[pp] - ((np.array(weight_size).prod(axis=1)[pp] + np.array(bias_size).prod(axis=1)[pp]) / math.log(4))) / (SIGMA * z[pp])
    # b[pp] = 8

    # Calculate Adversarial Noise
    adv_noise = 0
    for images, labels in testLoader:
        images = images.view(images.shape[0], -1).to(device)
        outputs = model(images)
        outputs_sort, _ = torch.sort(outputs, dim=1)
        adv_noise += torch.sum((outputs_sort[:, -1] - outputs_sort[:, -2]) ** 2 / 2).cpu().detach().numpy()
    r_star = adv_noise / 10000

    # Calculate Layer-wise quantization
    for j in range(pp, layers_number, 1):
        # Calculate t_i
        correct_total_test = 0
        y_total = 0
        r_y = 0
        acc_def_flag = 1
        i = 0
        for images, labels in testLoader:
            images = images.view(images.shape[0], -1).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct_total_test += (predicted.cpu() == labels).sum().numpy()
        acc_init = correct_total_test / 10000
        print("Layer %f in processing, initial avg acc: %f" % (j + 1, correct_total_test / 10000))
        while acc_def_flag:
            model.load_state_dict(torch.load('./check points/DNN_MNIST.pkl', map_location=device))  # reset the model model
            noise_weight, noise_bias = model.add_noise(layer_idx=j, snr=snr_start + i * snr_step_size)
            i += 1
            correct_total_test = 0
            # Test model
            for images, labels in testLoader:
                images = images.view(images.shape[0], -1).to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct_total_test += (predicted.cpu() == labels).sum().numpy()
            print(
                "avg acc after noise with snr of %f: %f" % (snr_start + i * snr_step_size, correct_total_test / 10000))
            if correct_total_test / 10000 <= acc_init - acc_deg:
                for images, labels in testLoader:
                    images = images.view(images.shape[0], -1).to(device)
                    y_target = model_target(images)
                    y = model(images)
                    y_total += torch.sum(torch.norm(y - y_target, p=2, dim=1) ** 2).cpu().detach().numpy()
                r_y = y_total / 10000
                acc_def_flag = 0
        t[j] = r_y / r_star

        # calculate p_i
        model.load_state_dict(torch.load('./check points/DNN_MNIST.pkl', map_location=device))  # Reset model
        quantization_flag = np.zeros(layers_number)
        quantization_flag[j] = 1
        model.quantize(8 * np.ones(layers_number), quantization_flag)
        y_total = 0
        for images, labels in testLoader:
            images = images.view(images.shape[0], -1).to(device)
            y_target = model_target(images)
            y = model(images)
            y_total += torch.sum(torch.norm(y - y_target, p=2, dim=1) ** 2).cpu().detach().numpy()
        r_y = y_total / 10000
        p[j] = r_y / (math.exp(-math.log(4) * 8))

        # Calculate b
        if j > pp:
            xx_0 = (z[pp] * t[pp]) / (p[pp] * math.exp(-math.log(4) * b[pp]))
            b[j] = math.log((z[j] * t[j]) / xx_0 / p[j]) / (-math.log(4))
    print(b)

    if pp == 0:
        mdict_base_model = {'computation_payload': o,
                            'activation_size': activation_size,
                            'quantization_bitwidth': b,
                            'parameter_size': z}
        savemat('/home/lixiangchen/MyWorkspace/InferenceOffloading/data/' + 'base_model_parameter.mat',
                mdict_base_model)

    activation_payload = np.array(activation_size[pp - 1]).prod() * b[pp - 1]
    model_weight_payload = sum([np.array(weight_size[pp + i]).prod() * b[pp + i] for i in range(layer_count - pp)])
    model_bias_payload = sum([np.array(bias_size[pp + i]).prod() * b[pp + i] for i in range(layer_count - pp)])
    model_payload = model_bias_payload + model_weight_payload
    T_device = sum([o[i] for i in range(pp)]) * gamma_device / f_device
    E_device = kappa_device * (f_device ** 2) * sum([o[i] for i in range(pp)]) * gamma_device
    T_server = sum([o[i] for i in range(pp, layer_count, 1)]) * gamma_server / f_server
    E_server = kappa_server * (f_server ** 2) * sum([o[i] for i in range(pp, layer_count, 1)]) * gamma_server
    T_trans = (activation_payload + model_payload) / channel_capacity
    E_trans = T_trans * pi_device
    cost = omega * (T_server + T_device + T_trans) + tau * (E_trans + E_server + E_device)


    return T_device, T_server, T_trans, E_device, E_server, E_trans, cost, weight_size, bias_size, b, activation_payload + model_payload
