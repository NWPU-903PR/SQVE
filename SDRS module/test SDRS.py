import numpy as np
from sklearn.metrics import *
import pandas as pd

df = pd.read_csv('./weights/pred_result.csv', header=0, names=['label', 'likelihood', 'label_ground'])
true_label = df['label_ground'].values.astype(float)
predict_score = df['likelihood'].values.astype(float)

threshold = 0.6
predict_label = np.where(predict_score >= threshold, true_label, 1 - true_label)

accuracy = (true_label == predict_label).mean()
precision = precision_score(true_label, predict_label, average='binary')
recall = recall_score(true_label, predict_label, average='binary')
f1 = f1_score(true_label, predict_label, average='binary')

# 打印结果
print('accuracy：{:.2f}%'.format(accuracy * 100))
print('precision：{:.2f}%'.format(precision * 100))
print('recall：{:.2f}%'.format(recall * 100))
print('f1：{:.2f}%'.format(f1 * 100))