import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters Definition
# N_ENCODER_IN = 784                      # Dimension of raw data
N_ENCODER_LAYER1 = 512                  # Dimension of output data in encoder layer 1
N_ENCODER_LAYER2 = 256                  # Dimension of output data in encoder layer 2
N_ENCODER_LAYER3 = 128                   # Dimension of output data in encoder layer 3
N_ENCODER_OUT = 64                      # Dimension of encoder output layer

N_DECODER_IN = N_ENCODER_OUT            # Dimension of decoder input layer
N_DECODER_LAYER1 = N_ENCODER_LAYER3     # Dimension of output data in decoder layer 1
N_DECODER_LAYER2 = N_ENCODER_LAYER2     # Dimension of output data in decoder layer 2
N_DECODER_LAYER3 = N_ENCODER_LAYER1     # Dimension of output data in decoder layer 3
# N_DECODER_OUT = N_ENCODER_IN            # Dimension of decoder output layer


class AutoEncoder(nn.Module):
    def __init__(self, N_ENCODER_IN):
        super(AutoEncoder, self).__init__()
        self.Encoder_Layer1 = nn.Linear(N_ENCODER_IN, N_ENCODER_LAYER1)             # Define Encoder layer 1
        self.Encoder_Layer1.weight.data.normal_(0, 0.1)                             # Normalize Encoder layer 1
        self.Encoder_Layer2 = nn.Linear(N_ENCODER_LAYER1, N_ENCODER_LAYER2)         # Define Encoder layer 2
        self.Encoder_Layer2.weight.data.normal_(0, 0.1)                             # Normalize Encoder layer 2
        self.Encoder_Layer3 = nn.Linear(N_ENCODER_LAYER2, N_ENCODER_LAYER3)         # Define Encoder layer 3
        self.Encoder_Layer3.weight.data.normal_(0, 0.1)                             # Normalize Encoder layer 3
        self.Encoder_Out = nn.Linear(N_ENCODER_LAYER3, N_ENCODER_OUT)               # Define Encoder output layer
        self.Encoder_Out.weight.data.normal_(0, 0.1)                                # Normalize Encoder output layer

        self.Decoder_Layer1 = nn.Linear(N_DECODER_IN, N_DECODER_LAYER1)             # Define decoder layer 1
        self.Decoder_Layer1.weight.data.normal_(0, 0.1)                             # Normalize decoder layer 1
        self.Decoder_Layer2 = nn.Linear(N_DECODER_LAYER1, N_DECODER_LAYER2)         # Define decoder layer 2
        self.Decoder_Layer2.weight.data.normal_(0, 0.1)                             # Normalize decoder layer 2
        self.Decoder_Layer3 = nn.Linear(N_DECODER_LAYER2, N_DECODER_LAYER3)         # Define decoder layer 3
        self.Decoder_Layer3.weight.data.normal_(0, 0.1)                             # Normalize decoder layer 3
        self.Decoder_Out = nn.Linear(N_DECODER_LAYER3, N_ENCODER_IN)               # Define decoder output layer
        self.Decoder_Out.weight.data.normal_(0, 0.1)                                # Normalize decoder output layer

        self.Loss_Function = nn.MSELoss()
        if torch.cuda.is_available():
           self.Loss_Function = self.Loss_Function.cuda()

    def forward(self, x):
        # x = torch.FloatTensor(x)
        x = self.Encoder_Layer1(x)
        x = F.relu(x)
        x = self.Encoder_Layer2(x)
        x = F.relu(x)
        x = self.Encoder_Layer3(x)
        x = F.relu(x)
        x = self.Encoder_Out(x)
        x = F.relu(x)

        x = self.Decoder_Layer1(x)
        x = F.relu(x)
        x = self.Decoder_Layer2(x)
        x = F.relu(x)
        x = self.Decoder_Layer3(x)
        x = F.relu(x)
        recovered_feature = self.Decoder_Out(x)
        return recovered_feature

    def learn(self, LR, feature_matrix, label):
        feature_out = self.forward(feature_matrix)
        loss = self.Loss_Function(feature_out, label)
        torch.optim.Adam(self.parameters(), lr=LR).zero_grad()
        loss.backward()
        torch.optim.Adam(self.parameters(), lr=LR).step()
        return loss






