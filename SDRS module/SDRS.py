import torch
from torchsummary import summary
import math

class SDRS(torch.nn.Module):
    def __init__(self, input_channels, input_sample_points, classes, num_frags):
        super(SDRS, self).__init__()

        self.input_channels = input_channels
        self.input_sample_points = input_sample_points
        self.num_frags = num_frags
        self.frag_len = math.ceil(input_sample_points / num_frags)

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 1, kernel_size=5, stride=1),
            torch.nn.BatchNorm1d(1),
            torch.nn.MaxPool1d(kernel_size=2, stride=1),
        )
        self.After_features_channels = 1
        self.After_features_sample_points = (((self.frag_len - 5) // 1 + 1) - 2) // 1 + 1
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.After_features_channels * self.After_features_sample_points, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2+5, classes),
            torch.nn.Softmax()
        )

    def forward(self, x):
        frags = torch.split(x, self.frag_len, dim=2)
        frags_out = []
        with open('clinical data.csv', 'r') as f:
            clinical_data = f.read().strip().split(',')
            clinical = torch.tensor([float(val) for val in clinical_data]).unsqueeze(0)

        for frag in frags:
            if frag.shape[2] < self.frag_len:
                padding = torch.zeros((frag.shape[0], self.input_channels, self.frag_len - frag.shape[2])).to(frag.device)
                frag = torch.cat((frag, padding), dim=2)
            out = self.features(frag)
            out = out.view(-1, self.After_features_channels * self.After_features_sample_points)
            out = torch.cat((out, clinical), dim=1)
            out = self.classifier(out)
            frags_out.append(out)

        x = torch.cat(frags_out, dim=1)  
        x = torch.mean(x, dim=1)
        x = x.unsqueeze(0)
        return x

