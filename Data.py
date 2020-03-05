import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



def split_attr(x):
    temperatures = torch.stack([x[:,0], x[:,7], x[:,28], x[:,31], x[:,32]]).t()
    temperatures = torch.unsqueeze(temperatures, -1)
    
    atmospheric_pressure = torch.stack([x[:,1], x[:,6], x[:,22], x[:,27], x[:,29]]).t()
    atmospheric_pressure = torch.unsqueeze(atmospheric_pressure, -1)
    
    wind_speed = torch.stack([x[:,2], x[:,3], x[:,18], x[:,24], x[:,26]]).t()
    wind_speed = torch.unsqueeze(wind_speed, -1)
    
    precipitation = torch.stack([x[:,4], x[:,10], x[:,21], x[:,36], x[:,39]]).t()
    precipitation = torch.unsqueeze(precipitation, -1)
    
    barometric_pressure = torch.stack([x[:,5], x[:,8], x[:,9], x[:,23], x[:,33]]).t()
    barometric_pressure = torch.unsqueeze(barometric_pressure, -1)
    
    insolation = torch.stack([x[:,11], x[:,14], x[:,16], x[:,19], x[:,34]]).t()
    insolation = torch.unsqueeze(insolation, -1)
    
    humidity = torch.stack([x[:,12], x[:,20], x[:,30], x[:,37], x[:,38]]).t()
    humidity = torch.unsqueeze(humidity, -1)
    
    wind_direction = torch.stack([x[:,13], x[:,15], x[:,17], x[:,25], x[:,35]]).t()
    wind_direction = torch.unsqueeze(wind_direction, -1)

    return [
        temperatures,
        atmospheric_pressure,
        wind_speed,
        precipitation,
        barometric_pressure,
        insolation,
        humidity,
        wind_direction,
    ]

def generate_datasets(train_path, test_path):
    #attribution keys
    k_temperatures = ['X00', 'X07', 'X28', 'X31', 'X32']
    k_atmospheric_pressure = ['X01', 'X06', 'X22', 'X27', 'X29']
    k_wind_speed = ['X02', 'X03', 'X18', 'X24', 'X26']
    k_precipitation = ['X04', 'X10', 'X21', 'X36', 'X39']
    k_barometric_pressure = ['X05', 'X08', 'X09', 'X23', 'X33']
    k_insolation = ['X11', 'X14', 'X16', 'X19', 'X34']
    k_humidity = ['X12', 'X20', 'X30', 'X37', 'X38']
    k_wind_direction = ['X13', 'X15, ''X17', 'X25', 'X35']


    f = open(train_path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    train_dicts = []
    dev_dicts = []
    for i, line in enumerate(rdr):
        if i == 0:
            keys = line
            continue
        dic = dict()
        for ii, item in enumerate(line):
            if item == '':
                continue
            dic[keys[ii]] = float(item)

        if 'Y18' not in dic.keys():
            train_dicts.append(dic)
        else:
            dev_dicts.append(dic)
    f.close()

    f = open(test_path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    test_dicts = []
    for i, line in enumerate(rdr):
        if i == 0:
            keys = line
            continue
        dic = dict()
        for ii, item in enumerate(line):
            if item == '':
                continue
            dic[keys[ii]] = float(item)

        test_dicts.append(dic)
    f.close()

    train_x = []
    train_y = []
    dev_x = []
    dev_y = []
    test_x = []
    test_y = []

    dics = [train_dicts, dev_dicts, test_dicts]
    xs = [train_x, dev_x, test_x]
    ys = [train_y, dev_y, test_y]

    for i, cur_dic in enumerate(dics):
        for dic in cur_dic:
            x = []
            y = []
            for k,v in dic.items():
                if v == '':
                    continue

                if k[0] == 'X':
                    x.append(v)
                elif k[0] == 'Y':
                    y.append(v)

            xs[i].append(x)
            ys[i].append(np.mean(y))

    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)
    dev_x = torch.FloatTensor(dev_x)
    dev_y = torch.FloatTensor(dev_y)
    test_x = torch.FloatTensor(test_x)
    test_y = torch.FloatTensor(test_y)
    return train_x, train_y, dev_x, dev_y, test_x,test_y

class Data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


