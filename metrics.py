import torch
from torch import nn
from torch.utils.data import DataLoader
from models.synthtic_VAE import synthtic_VAE
from utils import LoadData
import numpy as np
from sklearn.metrics import *
import pandas as pd
from __future__ import print_function
import math
from tqdm import tqdm
import pandas as pd

def RBO_score(l1, l2, p=0.1):
    if l1 == None: l1 = []
    if l2 == None: l2 = []
    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0
    ss = set([])
    ls = set([])
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        else:
            ls.add(x)
            if y != None: ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
        sum1 += x_d[d] / d * pow(p, d)
    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)
    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    return rbo_ext

def calculate_CBR(dataloader, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        CBR_score = correct / total
        return CBR_score







