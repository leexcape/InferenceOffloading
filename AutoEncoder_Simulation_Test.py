from AutoEncoder_Simulation import ae_task
import pandas as pd

T_device_rec = []
T_server_rec = []
T_trans_rec = []
E_device_rec = []
E_server_rec = []
E_trans_rec = []
cost_rec = []
acc_rec = []
comm_payload_rec = []

for pp in range(6):
    T_device, T_server, T_trans, E_device, E_server, E_trans, cost, acc, communication_payload = ae_task(pp)
    T_device_rec.append(T_device)
    T_server_rec.append(T_server)
    T_trans_rec.append(T_trans)
    E_device_rec.append(E_device)
    E_server_rec.append(E_server)
    E_trans_rec.append(E_trans)
    cost_rec.append(cost)
    acc_rec.append(acc)
    comm_payload_rec.append(communication_payload)

# Generate Data Frame
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
dataframe.to_csv('./results/ae_data.csv')

