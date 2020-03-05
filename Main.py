import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Data import *
from Model import *
from sklearn.metrics import r2_score

train_path = 'data/train.csv'
test_path = 'data/test.csv'

class Main:
    def __init__(self, mode, model_name, model_type):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 'cpu'

        self.model_name = model_name
        self.args = Args(model_name)
        self.model = Model(self.args)
        self.model.to(self.device)

    def save_model(self):
        torch.save(self.model.state_dict(), "models/%s" % self.model_name)
        self.args.save()

    def load_model(self):
        self.model.load_state_dict(torch.load("models/%s" % self.model_name, map_location=self.device))

    def train(self, train_path, test_path):
        print("Loading dataset")
        train_x, train_y, dev_x, dev_y, test_x, test_y = generate_datasets(train_path, test_path)
        train_data = Data(train_x, train_y)
        train_loader = DataLoader(dataset=train_data, batch_size=self.args.batch_size, shuffle=False)
        dev_data = Data(dev_x, dev_y)
        dev_loader = DataLoader(dataset=dev_data, batch_size=self.args.batch_size, shuffle=False)
        test_data = Data(test_x, test_y)
        test_loader = DataLoader(dataset=test_data, batch_size=self.args.batch_size, shuffle=False)

        print("Loading complete")
        print("Dataset size: Train %d, Dev %d, Test %d" % (len(train_data), len(dev_data), len(test_data)))

        max_score = 0
        max_score_epoch = 0
        loss_fn = nn.MSELoss()

        optimizer = torch.optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad])
        for epoch in tqdm(range(self.args.train_epoch)):
            self.model.train()
            losssum = 0
            for batch in train_loader:
                x, y = [x.to(self.device) for x in batch] # [32,40], [32]
                y_pred = self.model(x) # [32]
                loss = loss_fn(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % self.args.eval_per_epoch == 0:
                self.model.eval()

                labels = []
                preds = []
                for batch in dev_loader:
                    x, y = batch # [32,40], [32]
                    output = self.model(x) # [32]
                    preds.append(output)
                    labels.append(y)

                preds = torch.cat(preds) # [432]
                labels = torch.cat(labels) # [432]

                score = r2_score(labels.detach().numpy(), preds.detach().numpy())
                if score > max_score:
                    max_score_epoch = epoch
                    max_score = score
                    self.save_model()
                print("Epoch %d eval score: %.2f" % (epoch, score * 100))
                print("Max eval score @ epoch %d: %.2f" % (max_score_epoch, max_score * 100))

        self.load_model()
        self.model.eval()

        labels = []
        preds = []
        for batch in test_loader:
            x,y = batch
            output = self.model(x)
            preds.append(output)
            labels.append(y)

        preds = torch.cat(preds)
        out_f = open('result.csv','w', newline='')
        wr = csv.writer(out_f)
        wr.writerow(['id,Y18'])

        preds = preds.tolist()
        for i, id in enumerate(range(4752, 16272)):
            wr.writerow([id, preds[i]])
        out_f.close()

main = Main('train', 'temp', 'LSTM')
main.train(train_path, test_path)