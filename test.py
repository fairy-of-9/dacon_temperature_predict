import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv

out_f = open('result.csv','w', newline='')
wr = csv.writer(out_f)
wr.writerow(['id,Y18'])
wr.writerow([1,2.0])