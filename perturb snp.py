import torch
import pandas as pd
import numpy as np
from models.VAE import VAE
from train VAE import loss_function
from utils import LoadData
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

class perturb_categorical_data(Dataset):
    def __init__(self, txt_path):
        self.snps_info = self.get_snps(txt_path)

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

        perturbed_row = 391
        if snp[perturbed_row] == 0:
            snp[perturbed_row] = 2

        snp = torch.from_numpy(snp)
        snp = torch.unsqueeze(snp, 0)

        label = int(label)
        return snp, label

    def __len__(self):
        return len(self.snps_info)


def test(model, mnist_test, best_test_loss):
    test_avg_loss = 0.0
    with torch.no_grad():

        for test_batch_index, (test_x, _) in enumerate(mnist_test):
            test_x = test_x.type(torch.FloatTensor)
            test_x = test_x
            test_x_hat, test_mu, test_log_var= model(test_x)
            test_loss, test_mse, test_KLD = loss_function(test_x_hat, test_x, test_mu, test_log_var)
            test_avg_loss += test_loss

        test_avg_loss /= len(mnist_test.dataset)
        print(f"Test: \n Test_Avg_loss: {test_avg_loss:>.6f}% \n")
        best_test_loss =test_avg_loss

        x_concat = torch.cat([test_x.view(-1, 1, 619), test_x_hat.view(-1, 1, 619)], dim=1)
        x_concat_2d = x_concat.view(-1, 619)
        np.savetxt('./association_result/391_rs34505186/x_hat-9%d.csv' % test_batch_index, x_concat_2d.cpu().detach().numpy(),
                   fmt='%.4f', delimiter=",")

        return best_test_loss


def main():
    weight_dir = "association_result/391_rs34505186"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    mnist_test = perturb_categorical_data(txt_path="data/VAE2_input_500/perturb_391_rs34505186.txt")
    mnist_test = DataLoader(dataset=mnist_test, batch_size=201, shuffle=False)

    x, label = iter(mnist_test).__next__()
    print(' x : ', x.shape)
    best_test_loss = []

    model = VAE(z_dim=100)
    model.load_state_dict(torch.load("./VAE/model_final/checkpoint/checkpoint_latent=100_p=0.1_hdim=3000_lr=0.0001.pt"))
    best_test_loss = test(model, mnist_test, best_test_loss)
    return best_test_loss



if __name__ == "__main__":
    best_test_loss =main()