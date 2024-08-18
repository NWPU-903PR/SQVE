import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class VAE(nn.Module):
    def __init__(self, input_dim=619, h_dim=3000, z_dim=100):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.relu = nn.RReLU()
        self.dropoutlayer = nn.Dropout(p=0.1)


        self.fc1 = nn.Linear(500, h_dim)
        self.bn1  = nn.BatchNorm1d(h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)

        self.fc4 = nn.Linear(119, h_dim)
        self.bn4 = nn.BatchNorm1d(h_dim)
        self.fc5 = nn.Linear(h_dim, z_dim)
        self.fc6 = nn.Linear(h_dim, z_dim)

        self.fc7 = nn.Linear(z_dim*2, h_dim)
        self.bn7 = nn.BatchNorm1d(h_dim)
        self.fc8 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_dim)

        # encoder
        mu1, log_var1, mu2,log_var2 = self.encode(x)
        mu= torch.cat([mu1,mu2],dim=1)
        log_var = torch.cat([log_var1,log_var2],dim=1)
        z= self.reparameterization(mu,log_var)

        # decoder
        x_hat = self.decode(z)
        x_hat = x_hat.view(batch_size,1,619)
        return x_hat, mu,log_var


    def encode(self, x):
        x1, x2 = torch.split(x, [500, 119], dim=1)
        h1 = self.fc1(x1)
        h1 = self.relu(h1)
        h1 = self.dropoutlayer(h1)
        h1 = self.bn1(h1)
        mu1 = self.fc2(h1)
        log_var1 = self.fc3(h1)

        h2 = self.fc4(x2)
        h2 = self.relu(h2)
        h2 = self.dropoutlayer(h2)
        h2 = self.bn4(h2)
        mu2 = self.fc5(h2)
        log_var2 = self.fc6(h2)
        return mu1, log_var1, mu2, log_var2


    def reparameterization(self, mu, log_var):

        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        return z

    def decode(self, z):
        h3 = self.bn7(self.dropoutlayer(self.relu(self.fc7(z))))
        x_hat = self.fc8(h3)
        return x_hat
