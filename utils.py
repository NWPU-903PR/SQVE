import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def write_result_train(fileloc, epoch, loss, mse, KLD):
    with open(fileloc, "a") as f:
        data = "Epoch: " + str(epoch) + "\tTrainLoss " + str(loss) + "\tmseLoss " + str(mse) + "\tKLDLoss " + str(
            KLD)  + "\n"
        f.write(data)

def write_result_test(fileloc, epoch, test_avg_loss,test_mse, test_KLD):
    with open(fileloc,"a") as f:
        data = "Epoch: "+ str(epoch) + "\tTestLoss " + str(test_avg_loss)+"\tmseLoss " + str(test_BCE) + "\tKLDLoss " + str(
            test_KLD1)  + "\n"
        f.write(data)

class LoadData(Dataset):
    def __init__(self, txt_path):
        self.snps_info = self.get_snps(txt_path)

    def get_snps(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            snps_info = f.readlines()
            snps_info = list(map(lambda x:x.strip().split('\t'), snps_info))
        return snps_info


    def __getitem__(self, index):
        snp_path, label = self.snps_info[index]
        snp = pd.read_csv(snp_path)#
        col = snp.iloc[:,0]
        snp = np.array(col)
        snp = snp.astype(float)
        snp = torch.from_numpy(snp)
        snp = torch.unsqueeze(snp, 0)
        label = int(label)
        return snp, label

    def __len__(self):
        return len(self.snps_info)