import os
import random

all_ratio = 1
rootdata = r"dataset"
all_list =[]
data_list = []

class_flag = -1
for a,b,c in os.walk(rootdata):
    print(a)
    for i in range(len(c)):
        data_list.append(os.path.join(a,c[i]))

    for i in range(0,int(len(c)*all_ratio)):
        all_data = os.path.join(a, c[i])+'\t'+str(class_flag)+'\n'
        all_list.append(all_data)
    class_flag += 1

print(data_list)
random.shuffle(all_list)

with open('dataset.txt', 'w', encoding='UTF-8') as f:
    for all_snp in all_list:
        f.write(str(all_snp))
