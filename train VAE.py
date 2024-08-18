import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.VAE import VAE
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import numpy as np
from utils import LoadData
from utils import write_result_train,write_result_test
from pytorchtools import EarlyStopping
from metrics import calculate_cosine_similarity
import csv
import pandas as pd


plt.style.use("ggplot")
cuda = torch.cuda.is_available()
device = torch.device("cuda:1")

parser = argparse.ArgumentParser(description="Dual Encoder VAE")
parser.add_argument('--result_dir', type=str, default='./VAE/model_final', metavar='DIR', help='output directory')
parser.add_argument('--save_dir', type=str, default='./VAE/model_final', metavar='N', help='model saving directory')
parser.add_argument('--batch_size', type=int, default=5, metavar='N', help='batch size for training(default: 128)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train(default: 200)')
parser.add_argument('--seed', type=int, default=10, metavar='S', help='random seed(default: 1)')
parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint(default: None)')
parser.add_argument('--test_every', type=int, default=10, metavar='N', help='test after every epochs')
parser.add_argument('--num_worker', type=int, default=16, metavar='N', help='the number of workers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate(default: rs429358.001)')
parser.add_argument('--z_dim', type=int, default=100, metavar='N', help='the dim of latent variable z(default: 20)')
parser.add_argument('--input_dim', type=int, default=1 * 619, metavar='N', help='input dim(default: 28*28 for MNIST)')
parser.add_argument('--input_channel', type=int, default=1, metavar='N', help='input channel(default: 1 for MNIST)')
args = parser.parse_args()
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}


def dataloader(batch_size=5, num_workers=16):
    mnist_train = LoadData("data/VAE_input_500/train.txt")
    mnist_test = LoadData("data/VAE_input_500/test.txt")
    mnist_train = DataLoader(dataset=mnist_train, num_workers=num_workers,batch_size=batch_size,shuffle=True,drop_last=True)
    mnist_test = DataLoader(dataset=mnist_test, num_workers=num_workers,  batch_size=batch_size, shuffle=False,drop_last=True)
    return mnist_test, mnist_train

def loss_function(x_hat, x, mu, log_var):
    mse= F.mse_loss(x_hat, x)
    mse.to(device)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    KLD.to(device)
    loss = mse + KLD
    loss.to(device)
    return loss, BCE, KLD

def save_checkpoint(state, is_best, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)

def test(model, optimizer, mnist_test, epoch, best_test_loss):
    test_avg_loss = 0.0
    test_losses = []
    with torch.no_grad():
        for test_batch_index, (test_x, _) in enumerate(mnist_test):
            test_x = test_x.type(torch.FloatTensor)
            test_x = test_x.to(device)
            test_x_hat, test_mu, test_log_var= model(test_x)
            test_loss, test_mse, test_KLD = loss_function(test_x_hat, test_x, test_mu, test_log_var)
            test_avg_loss += test_loss

        test_avg_loss /= len(mnist_test.dataset)
        print(f"Test: \n Test_Avg_loss: {test_avg_loss:>.6f}% \n")
        write_result_test('./VAE2/model_final/write_result_loss/TestLoss/TestLoss.txt', epoch + 1, test_avg_loss,test_mse, test_KLD)

        is_best = test_avg_loss < best_test_loss
        best_test_loss = min(test_avg_loss, best_test_loss)

        save_checkpoint({
            'epoch': epoch,
            'best_test_loss': best_test_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_dir)
        return best_test_loss

def main():
    mnist_test, mnist_train = dataloader(args.batch_size, args.num_worker)
    x, label = iter(mnist_train).__next__()
    x.to(device)
    label.to(device)
    print(' x : ', x.shape)

    model = VAE(z_dim=args.z_dim).to(device)
    print('The structure of our model is shown below: \n')
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    start_epoch = 0
    best_test_loss = np.finfo('f').max
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    loss_epoch = []

    for epoch in range(start_epoch, args.epochs):
        loss_batch = []
        for batch_index, (x, _) in enumerate(mnist_train):
            x = x.float()
            x = x.to(device)
            x_hat, mu, log_var = model(x)
            loss, mse, KLD = loss_function(x_hat, x, mu, log_var)
            loss_batch.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_index + 1) % 10 == 0:
                print('Epoch [{}/{}], Batch [{}/{}] : Total-loss = {:.4f}, mse-Loss = {:.4f}, KLD-loss = {:.4f}'
                      .format(epoch + 1, args.epochs, batch_index + 1, len(mnist_train.dataset) // args.batch_size,
                              loss.item() / args.batch_size, BCE.item() / args.batch_size, KLD1.item() / args.batch_size))
                write_result_train('./VAE2/model_final/write_result_loss/TrainLoss/TrainLoss.txt', epoch + 1, loss, mse, KLD)

            if batch_index == 0:
                x_concat = torch.cat([x.view(-1, 1, 619), x_hat.view(-1, 1, 619)], dim=1)
                x_concat_2d = x_concat.view(-1, 619)
                score_x_xhat = F.cosine_similarity(x, x_hat, dim=0)
                score_x_xhat = score_x_xhat.view(-1,619)
                np.savetxt('./%s/score-%d.csv' % (args.result_dir, epoch + 1),
                           score_x_xhat.cpu().detach().numpy(), delimiter=",")
                np.savetxt('./%s/reconstructed-%d.csv' % (args.result_dir, epoch + 1), x_concat_2d.cpu().detach().numpy(),fmt='%.4f',delimiter=",")

        loss_epoch.append(np.sum(loss_batch) / len(mnist_train.dataset))
        if (epoch + 1) % args.test_every == 0:
            best_test_loss = test(model, optimizer, mnist_test, epoch, best_test_loss)

        early_stopping(best_test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return loss_epoch

if __name__ == '__main__':
    loss_epoch = main()
    plt.plot(loss_epoch)
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.title('Train Loss Curve')
    plt.show()

