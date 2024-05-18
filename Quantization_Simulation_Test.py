import matplotlib.pyplot as plt
from Quantization_Simulation import q_task
import numpy as np
import pandas as pd
from overlapped_bar import overlapped_bar
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from DNN import DNN
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = '/home/lixiangchen/MyWorkspace/InferenceOffloading/data'
testData = dsets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testData, batch_size=32, shuffle=False)

T_device_rec = []
T_server_rec = []
T_trans_rec = []
E_device_rec = []
E_server_rec = []
E_trans_rec = []
cost_rec = []
acc_rec = []
bit_width_rec = []
comm_payload_rec = []

layers_number = 6

for i in range(layers_number):
    model = DNN()
    model.load_state_dict(torch.load('./check points/DNN_MNIST.pkl', map_location=device))
    model.to(device)
    model.eval()
    T_device, T_server, T_trans, E_device, E_server, E_trans, cost, weight_size, bias_size, QUANTIZATION_BITWIDTH, payload = q_task(i, 0.03)
    QUANTIZATION_BITWIDTH = np.floor(QUANTIZATION_BITWIDTH)
    T_device_rec.append(T_device)
    T_server_rec.append(T_server)
    T_trans_rec.append(T_trans)
    E_device_rec.append(E_device)
    E_server_rec.append(E_server)
    E_trans_rec.append(E_trans)
    cost_rec.append(cost)
    comm_payload_rec.append(payload)
    bit_width_rec.append(QUANTIZATION_BITWIDTH)
    correct_total_test = 0
    model.quantize(bitwidth=QUANTIZATION_BITWIDTH, quantization_flag=np.ones(layers_number))
    for images, labels in testLoader:
        images = images.view(images.shape[0], -1).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_total_test += (predicted.cpu() == labels).sum().numpy()
    acc_q = correct_total_test / 10000
    print('Quantization Bit-width: ', QUANTIZATION_BITWIDTH)
    print('Accuracy after quantization: ', acc_q)
    acc_rec.append(acc_q)

# plot
init_param_size = np.array(weight_size).prod(axis=1) * 32 + np.array(bias_size).prod(axis=1) * 32
q_param_size = np.array([np.array(weight_size)[k, :].prod() * bit_width_rec[0][k] + np.array(bias_size)[k, :].prod() * bit_width_rec[0][k] for k in range(layers_number)])
lower_bar = q_param_size
upper_bar = init_param_size - q_param_size
df = pd.DataFrame(np.matrix([upper_bar, lower_bar]).T, columns=['Initial Size', 'Quantized Size'])
overlapped_bar(df, xlabel="layers", ylabel="parameter sizes", show=True)


plt.figure()
x = np.arange(layers_number)
E_total = np.array(E_server_rec) + np.array(E_device_rec) + np.array(E_trans_rec)
T_total = np.array(T_server_rec) + np.array(T_device_rec) + np.array(T_trans_rec)
plt.plot(x, cost_rec, color='cyan', label='overall cost')
plt.plot(x, E_total * 1e9, color='y', label='overall energy consumption')
plt.plot(x, T_total * 2e9, color='b', label='overall time consumption')
plt.legend()
plt.xlabel('partition point')
plt.show()

plt.figure()
plt.plot(x, T_total, color='b', label='overall time consumption')
plt.plot(x, T_device_rec, color='lime', label='Local inference latency')
plt.plot(x, T_trans_rec, color='turquoise', label='transmission latency')
plt.plot(x, T_server_rec, color='violet', label='remote inference latency')
plt.legend()
plt.xlabel('partition point')
plt.ylabel('Time (s)')
plt.show()

plt.figure()
plt.plot(x, E_total, color='y', label='overall energy consumption')
plt.plot(x, E_device_rec, color='bisque', label='Local inference energy')
plt.plot(x, E_trans_rec, color='orange', label='transmission energy')
# plt.plot(x, E_server_rec_q, color='peru', label='remote inference energy')
plt.legend()
plt.xlabel('partition point')
plt.ylabel('Energy (J)')
plt.show()

data_dict = {'T_device': T_device_rec,
             'T_server': T_server_rec,
             'T_trans': T_trans_rec,
             'E_device': E_device_rec,
             'E_server': E_server_rec,
             'E_trans': E_trans_rec,
             'cost': cost_rec,
             'acc': acc_rec,
             'communication_payload': comm_payload_rec
             }
dataframe = pd.DataFrame(data_dict)
dataframe.to_csv('./results/q_data.csv')

parm_size_dict = {
    'Initial_Size': init_param_size,
    'Quantized_Size': q_param_size
}
parm_size_df = pd.DataFrame(parm_size_dict)
parm_size_df.to_csv('./results/parm_size_data.csv')


