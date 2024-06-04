import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader

training_SNRdb = 10
num_workers = 8
batch_size = 200
batch_size_DML = 256
training_data_len = 20000
indicator = -1
data_len_for_test = 10000
Pilot_num=128

CNN_for_scenario0 = DCE_P128()
CNN_for_scenario0 = torch.nn.DataParallel(CNN_for_scenario0).to(device)
fp = os.path.join(f'./workspace/Pn_{Pilot_num}/DCE',f'{self.training_data_len}_{self.training_SNRdb}dB_best_scenario0.pth')
try:
    CNN_for_scenario0.load_state_dict(torch.load(fp))
except:
    CNN_for_scenario0.load_state_dict(torch.load(fp)['cnn'])

Yp_input=pd.read_csv('Yp_hr.csv').values
Hhat0 = CNN_for_scenario0(Yp_input).detach().cpu()
df=pd.DataFrame(Hhat0)
df.to_csv('Hhat0.csv') 