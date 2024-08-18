import time
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, Subset
from utils SDRS import LoadData, write_result
from Models.SDRS import SDRS
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    avg_total = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X = X.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        X = torch.unsqueeze(X, 1)
        y = torch.unsqueeze(y, -1)
        X, y = X.cuda(), y.cuda()
        pred = model(X)
        loss = loss_fn(pred, y)
        avg_total = avg_total+loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>5f}  [{current:>5d}/{size:>5d}]")
    train_avg_loss = f"{(avg_total % batch):>5f}"
    return train_avg_loss

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            X = torch.unsqueeze(X, 1)
            y = torch.unsqueeze(y, -1)
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            _, pred = torch.max(pred.data, 1)
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)

        accuracy = 100 * total_correct / total_samples
        test_loss /= size
        print('Accuracy: {}%'.format(accuracy), 'test_loss: {}%'.format(test_loss))
        return accuracy,test_loss

if __name__=='__main__':
    batch_size = 1
    weight_dir = "weights/model_pth/"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    all_data = LoadData("dataset.txt", True)
    learning_rate=0.0001
    k =10
    kf = KFold(n_splits=k)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = SDRS(input_channels=1, input_sample_points=135854, classes=1, num_frags=40)
    model.to(device)
    print(model)
    time_start = time.time()
    loss_fn = nn.BCELoss()
    loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for fold, (train_index, test_index) in enumerate(kf.split(all_data)):
        print(f'Fold {fold + 1}:')
        train_data = Subset(all_data, train_index)
        valid_data = Subset(all_data, test_index)

        train_dataloader = DataLoader(dataset=train_data, num_workers=16, pin_memory=True, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(dataset=valid_data, num_workers=16, pin_memory=True, batch_size=1, shuffle=False)

        epochs = 500
        best = 0.0

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            train_loss = train(train_dataloader, model, loss_fn, optimizer)
            test_accuracy, test_avg_loss = test(test_dataloader, model)
            write_result(f'fold_fragment_cv_{fold + 1}_results.txt', t + 1, train_loss, test_avg_loss, test_accuracy)

            if (t + 1) % 10 == 0:
                torch.save(model.state_dict(), weight_dir + "SDRS_epoch_" + str(t + 1) + "_acc_" + str(test_accuracy) + ".pth")
            if float(test_accuracy) > best:
                best = float(test_accuracy)
                torch.save(model.state_dict(), weight_dir +"BEST SDRS_1_epoch_" + str(t + 1) + "_acc_" + str(test_accuracy) + ".pth")

    time_end = time.time()
    print(f"train time: {(time_end - time_start)}")
    print("Train SDRS Model Success!")






