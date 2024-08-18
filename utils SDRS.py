import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

def write_result(fileloc, epoch, trainloss, testloss, testaccuracy):
    with open(fileloc,"a") as f:
        data = "Epoch: "+ str(epoch) + "\tTrainLoss " + str(trainloss) + "\tTestLoss " + str(testloss)+ "\tTestAccuracy " + str(testaccuracy)+ "\n"
        f.write(data)

class LoadData(Dataset):
    def __init__(self, txt_path,  all_flag=True):
        self.snps_info = self.get_snps(txt_path)
        self.all_flag = all_flag


    def get_snps(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            snps_info = f.readlines()
            snps_info = list(map(lambda x:x.strip().split('\t'), snps_info))
        return snps_info

    def __getitem__(self, index):
        snp_path, label = self.snps_info[index]
        snp = pd.read_csv(snp_path)
        col = snp.iloc[:,0]
        snp = np.array(col)
        snp = snp.astype(float)
        snp = torch.from_numpy(snp)
        label = int(label)

        return snp, label

    def __len__(self):
        return len(self.snps_info)


