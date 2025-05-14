import torch
from torch import nn
from torch.utils.data import DataLoader
from Models.SDRS import SDRS
from utils SDRS import LoadData, write_result
from tqdm import tqdm
import pandas as pd

def eval(dataloader, model):
    label_list = []
    likelihood_list = []
    label_ground_list =[]
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader,desc="Model is predicting"):
            X = X.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            pred = model(X)
            label = pred.cpu().numpy()
            if label >= 0.5:
                label = 1
            else:
                label = 0
            label_list.append(label)
            likelihood = pred.cpu().numpy().max()
            likelihood_list.append(likelihood)
            label_ground_list.append(y.item())
        return label_list,likelihood_list,label_ground_list

if __name__ == "__main__":
    model = SDRS(input_channels=1, input_sample_points=135854, classes=1, num_frags=40)
    model_loc = "./weights/best model.pt"
    model_dict = torch.load(model_loc)
    model.load_state_dict(model_dict)

    test_data = LoadData("test.txt",False)
    test_dataloader = DataLoader(dataset=test_data, pin_memory=True, batch_size=1)

    label_list, likelihood_list, label_ground_list= eval(test_dataloader, model)
    label_names = ["0CN","1AD"]
    result_name = [label_names[i] for i in label_list]
    list = [result_name, likelihood_list,label_ground_list]
    df = pd.DataFrame(data=list)
    df = pd.DataFrame(df.values.T,columns=['label','likelihood','label_ground'])
