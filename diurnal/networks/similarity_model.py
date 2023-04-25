from sklearn.metrics import f1_score, confusion_matrix
import torch.optim as optim
import torch

from .mlp import RNA_MLP_classifier
from .cnn import RNA_CNN_family_aware
import sys
sys.path.append("..")
from utils import datahandler

class SimilarityModel():
    """
    """
    def __init__(self, n: int, n_families: int, device: str):
        self.device = device
        self.classifier = RNA_MLP_classifier(n, n_families).to(device).half()
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(), eps=1e-04)
        self.classifier_loss_fn = torch.nn.MSELoss()
        self.predictor  = RNA_CNN_family_aware(n, n_families).to(device).half()
        self.predictor_optimizer = optim.Adam(
            self.predictor.parameters(), eps=1e-04)
        self.predictor_loss_fn = torch.nn.MSELoss()

    def train_classifier(self, data, epochs) -> None:
        self.classifier.train()
        for epoch in range(epochs):
            for batch, (x, _, f) in enumerate(data):
                x, f = x.to(self.device).half(), f.to(self.device).half()
                self.classifier_optimizer.zero_grad()
                pred = self.classifier(x)
                loss = self.classifier_loss_fn(pred, f)
                loss.backward()
                self.classifier_optimizer.step()
            #print(f"ec{epoch}")
    
    def test_classifier(self, data) -> None:
        self.classifier.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for x, _, f in data:
                x, f = x.to(self.device).half(), f.to(self.device).half()
                output = self.classifier(x)
                for i, j in zip(output, f):
                    pred = i.tolist()
                    y_pred.append(pred.index(max(pred)))
                    true = j.tolist()
                    y_true.append(true.index(max(true)))
                    #if len(y_pred) == 1:
                    #    print(f"P: {pred}: {pred.index(max(pred))}")
                    #    print(f"R: {true}: {true.index(max(true))}")
        print(confusion_matrix(y_true, y_pred))
        print(f1_score(y_true, y_pred, average='weighted'))
        return f1_score(y_true, y_pred, average='weighted')

    def train_predictor(self, data, epochs) -> None:
        self.predictor.train()
        for epoch in range(epochs):
            for batch, (x, y, _) in enumerate(data):
                x, y = x.to(self.device).half(), y.to(self.device).half()
                self.predictor_optimizer.zero_grad()
                pred = self.predictor(x, self.classifier(x))
                loss = self.predictor_loss_fn(pred, y)
                loss.backward()
                self.predictor_optimizer.step()
            #print(f"ep{epoch}")

    def train(self, data, epochs: int) -> None:
        self.train_classifier(data, 1)
        self.test_classifier(data)
        self.train_predictor(data, epochs)

    def predict(self, x) -> tuple:
        return self.predictor(x, self.classifier(x))

    def test(self, data) -> float:
        self.predictor.eval()
        f1 = []
        with torch.no_grad():
            for x, y, _ in data:
                x, y = x.to(self.device).half(), y.to(self.device).half()
                output = self.predict(x)
                for i, j in zip(output, y):
                    prediction = i.tolist()
                    real_seq = j.tolist()
                    y_pred = datahandler.prediction_to_classes(
                        prediction, real_seq)
                    y_true = datahandler.prediction_to_classes(
                        real_seq, real_seq)
                    f1.append(f1_score(y_true, y_pred, average='weighted'))
                    # debug
                    #p = datahandler.prediction_to_secondary_structure(prediction)
                    #r = datahandler.prediction_to_secondary_structure(real_seq)
                    #print(f"{len(f1)} P: {p}")
                    #print(f"{len(f1)} R: {r}")
                    #print(f"F1: {f1[-1]}")
                    #print()
        return f1
