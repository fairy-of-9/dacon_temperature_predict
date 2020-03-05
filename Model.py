import torch
import torch.nn as nn
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from Data import *
import json
from abc import ABC, abstractmethod

def jsonload(fname):
	with open(fname, encoding="UTF8") as f:
		j = json.load(f)
	return j

def jsondump(obj, fname):
	with open(fname, "w", encoding="UTF8") as f:
		json.dump(obj, f, ensure_ascii=False, indent="\t")

class Argument(ABC):
	@classmethod
	def load_argument(cls, model_name):
		args = cls(model_name)
		json_file = jsonload(args.path)
		for attr, value in json_file.items():
			try:
				setattr(args, attr, value)
			except AttributeError:
				pass
		return args

	def save(self):
		jsondump(self.__dict__, self.path)

	@property
	@abstractmethod
	def path(self):
		pass



class Args(Argument):
    def __init__(self, model_name):
        self.model_name = model_name
        # model config
        self.hidden_dim = 300

        # train config
        self.train_epoch = 10
        self.eval_per_epoch = 1
        self.batch_size = 32

        self.code_test = True

    @property
    def path(self):
        return "models/%s_args.json" % self.model_name

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.device = 'cpu'
        self.model_name = args.model_name
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size

        self.lstm1 = LSTM(args)
        self.lstm2 = LSTM(args)
        self.lstm3 = LSTM(args)
        self.lstm4 = LSTM(args)
        self.lstm5 = LSTM(args)
        self.lstm6 = LSTM(args)
        self.lstm7 = LSTM(args)
        self.lstm8 = LSTM(args)

        self.lstms = [
            self.lstm1, self.lstm2, self.lstm3, self.lstm4, self.lstm5, self.lstm6, self.lstm7, self.lstm8
        ]

        self.ffnn = nn.Linear(self.hidden_dim*2*8, 1) # 4800 -> 1

    def forward(self, x): # [32,40]
        xs = split_attr(x)
        outs = []
        for i, input in enumerate(xs): # input = [32,5,1]
            out, (hidden, cell) = self.lstms[i](input) # out = [32,5,600]
            out = out[:,-1,:] # [32,600]
            outs.append(out)
        output = torch.cat(outs, dim=1) # [32,4800]
        output = self.ffnn(output) # [32,1]
        output = output.squeeze() # [32]
        return output



class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 'cpu'
        self.model_name = args.model_name
        self.hidden_dim = args.hidden_dim

        self.lstm = nn.LSTM(input_size=1, hidden_size=args.hidden_dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)

    def forward(self, x):
        out = self.lstm(x)
        return out
