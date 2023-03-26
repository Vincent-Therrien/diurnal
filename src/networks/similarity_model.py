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



"""
INTER-FAMILY TESTING
No similarity:
F1-score with family 16s (67 samples): 0.43955937438560744
F1-score with family 23s (15 samples): 0.4321405305174956
F1-score with family 5s (1283 samples): 0.45680260866988887
F1-score with family grp1 (74 samples): 0.4074431289632272
F1-score with family RNaseP (454 samples): 0.41820261053936697
F1-score with family srp (918 samples): 0.4472104326477982
F1-score with family telomerase (35 samples): 0.4117851247874015
F1-score with family tmRNA (462 samples): 0.42745302484641917
F1-score with family tRNA (557 samples): 0.48021449994968546

With similarity:
[[  15    0    0    0    0    0    0    0]
 [   0 1283    0    0    0    0    0    0]
 [   0    0   72    1    1    0    0    0]
 [   1    0    0  452    1    0    0    0]
 [   0    0    0    0  916    0    0    2]
 [   0    0    0    0    0   35    0    0]
 [   0    0    0    0    2    0  460    0]
 [   0    0    0    0    0    0    0  557]]
0.9978945929261496
F1-score with family 16s (67 samples): 0.41178862829087476
[[  65    0    0    1    1    0    0    0]
 [1283    0    0    0    0    0    0    0]
 [  71    0    0    1    2    0    0    0]
 [   0    0    0  454    0    0    0    0]
 [   9    0    0    0  906    0    0    3]
 [  35    0    0    0    0    0    0    0]
 [   2    0    0    0    0    0  460    0]
 [   0    0    0    0    0    0    0  557]]
0.6196499974104509
F1-score with family 23s (15 samples): 0.39141484814935057
[[ 66   0   0   0   0   0   0   1]
 [ 15   0   0   0   0   0   0   0]
 [ 74   0   0   0   0   0   0   0]
 [454   0   0   0   0   0   0   0]
 [ 15   0   0   0 901   0   0   2]
 [ 35   0   0   0   0   0   0   0]
 [  1   0   0   0   1   0 460   0]
 [  0   0   0   0   0   0   0 557]]
0.7504213014987834
F1-score with family 5s (1283 samples): 0.4709967555389486
[[  66    0    0    0    1    0    0    0]
 [  15    0    0    0    0    0    0    0]
 [   2    0 1281    0    0    0    0    0]
 [ 454    0    0    0    0    0    0    0]
 [  13    0    0    0  902    0    0    3]
 [  35    0    0    0    0    0    0    0]
 [   2    0    0    0    0    0  460    0]
 [   0    0    0    0    0    0    0  557]]
0.8497636504154361
F1-score with family grp1 (74 samples): 0.4020507109454451
[[  62    0    0    2    3    0    0    0]
 [   0   15    0    0    0    0    0    0]
 [   0    0 1283    0    0    0    0    0]
 [   0    0    0   73    1    0    0    0]
 [   0    0    0    0  918    0    0    0]
 [   0    0    0    0   35    0    0    0]
 [   0    0    0    0    0    0  462    0]
 [   0    0    0    0    0    0    0  557]]
0.9829430571355285
F1-score with family RNaseP (454 samples): 0.39669986768739596
[[  67    0    0    0    0    0    0    0]
 [  15    0    0    0    0    0    0    0]
 [1283    0    0    0    0    0    0    0]
 [  74    0    0    0    0    0    0    0]
 [ 454    0    0    0    0    0    0    0]
 [  35    0    0    0    0    0    0    0]
 [ 462    0    0    0    0    0    0    0]
 [ 557    0    0    0    0    0    0    0]]
0.0010107790158763684
F1-score with family srp (918 samples): 0.4481540712754037
[[  63    0    1    3    0    0    0    0]
 [   0    0    0   15    0    0    0    0]
 [   0    0 1283    0    0    0    0    0]
 [   1    0    0   72    0    1    0    0]
 [   0    0    0    8  446    0    0    0]
 [   0    0    0    0    0  916    0    2]
 [   0    0    0    1    0    0  461    0]
 [   0    0    0    0    0    0    0  557]]
0.9902097117500479
F1-score with family telomerase (35 samples): 0.3768664701682824
[[  61    0    0    4    1    1    0    0]
 [  15    0    0    0    0    0    0    0]
 [   0    0 1283    0    0    0    0    0]
 [   9    0    0   59    3    2    0    1]
 [   9    0    0    1  444    0    0    0]
 [   5    0    0    2    2  908    0    1]
 [  32    0    0    0    2    0    0    1]
 [   0    0    0    0    0    0    0  557]]
0.9693251663152499
F1-score with family tmRNA (462 samples): 0.4064219088123012
[[  64    0    0    0    0    2    0    1]
 [  15    0    0    0    0    0    0    0]
 [   0    0 1283    0    0    0    0    0]
 [  65    0    0    0    6    0    0    3]
 [   3    0    0    0  450    1    0    0]
 [   3    0    0    0    2  913    0    0]
 [   1    0    0    0    1    0   33    0]
 [   1    0    0    0    0    0    0  461]]
0.9604503419240636
F1-score with family tRNA (557 samples): 0.4774521298032328


OMNI-FAMILY TESTING
Without similarity (1 epoch for the classifier):
[[  18    0    7    0    0   24    0    8    1]
 [   3    0    0    0    0    6    0    2    0]
 [   0    0 1028    0    0   10    0    0    0]
 [  25    0    0    0    0   21    0   14    0]
 [ 141    0    2    0    0  138    0   79    0]
 [  14    0   16    0    0  645    0   21   41]
 [  11    0    1    0    0    4    0   12    0]
 [   1    0    0    0    0    3    0  367    0]
 [   0    0    0    0    0    5    0    0  424]]
0.7576660352238401
K=0: 0.873120480457951
[[   0    0   10    0    6   36    0    4    2]
 [   0    0    0    0    0   10    0    2    0]
 [   0    0 1015    0    0    5    0    0    0]
 [   0    0    1    0   12   30    0   16    3]
 [   0    0    2    0   68  242    0   44    1]
 [   0    0   55    0    7  641    0    9   27]
 [   0    0    1    0    3    5    0   21    0]
 [   0    0    0    0    0   12    0  351    0]
 [   0    0    0    0    0   12    0    0  439]]
0.7703469394206865
K=1: 0.8679295747540834
[[  5   0   0   0   5  42   0   0   0]
 [  0   0   0   0   9   3   0   0   0]
 [955   0   0   0   0  67   0   0   0]
 [  2   0   0   0  16  36   0   0   0]
 [  0   0   0   0 269  90   0   4   0]
 [  1   0   0   0  13 710   0   0  19]
 [  4   0   0   0   1  20   0   0   0]
 [  0   0   0   0  22   1   0 356   0]
 [  0   0   0   0   0  21   0   0 421]]
0.5421132705547326
K=2: 0.8622394600396672
[[   0    0    0    0    0   47    0    0    0]
 [   0    0    0    0    0   12    0    0    0]
 [   0    0    0    0    0 1037    0    0    0]
 [   0    0    0    0    0   59    0    0    0]
 [   0    0    0    0    0  362    0    0    0]
 [   0    0    0    0    0  739    0    0    0]
 [   0    0    0    0    0   30    0    0    0]
 [   0    0    0    0    0  365    0    0    0]
 [   0    0    0    0    0  441    0    0    0]]
0.09220770976067438
K=3: 0.8572934103860692
[[   0    0    0    0   28   25    0    0    0]
 [   0    0    0    0    8    5    0    0    0]
 [   0    0    0    0    3 1002    0    0   10]
 [   0    0    0    0   44   17    0    0    0]
 [   0    0    0    0  302   70    0    2    0]
 [   0    0    0    0   61  578    0    0   75]
 [   0    0    0    0   26    0    0    1    0]
 [   0    0    0    0   56    2    0  304    8]
 [   0    0    0    0    0   20    0    0  445]]
0.4316255098751688
K=4: 0.8648481980478672
0.8153130341903037

With similarity (40 epochs for the classifier):
[[  52    0    1    0    0    1    0    0    0]
 [  10    0    0    0    0    0    0    0    0]
 [   0    0 1004    0    0    0    0    0    0]
 [  68    0    0    0    0    1    0    0    0]
 [   4    0    0    0  370    0    0    0    0]
 [   9    0    0    0    0  716    0    0    0]
 [  28    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0  377    0]
 [   0    0    0    0    0    0    0    0  451]]
0.9534032392190126
K=0: 0.8687709698183493
[[  49    0    0    2    0    0    0    0    0]
 [  14    0    0    0    0    0    0    0    0]
 [   0    0 1030    0    0    3    0    0    0]
 [   7    0    0   41    2    3    0    0    0]
 [   0    0    0    0  368    0    0    0    0]
 [  15    0    2    0    0  716    0    0    2]
 [  21    0    0    1    0    0    0    0    0]
 [   1    0    0    0    0    0    0  370    0]
 [   0    0    0    0    0    0    0    0  445]]
0.9737479208678902
K=1: 0.8548418866972881
[[  59    0    0    0    0    0    0    0    0]
 [  12    0    0    0    0    0    0    0    0]
 [   0    0 1019    0    0    0    0    0    0]
 [   1    0    0   55    0    1    0    0    0]
 [   0    0    0    0  360    0    0    0    0]
 [   0    0    0    0    0  737    0    0    0]
 [   0    0    0    0    0    0   29    0    0]
 [   1    0    0    0    0    0    0  374    0]
 [   0    0    0    0    0    0    0    0  444]]
0.9934425092257037
K=2: 0.8626132955724127
[[  49    0    1    1    0    1    0    0    0]
 [   0    4    0    0    0    0    2    6    0]
 [   0    0 1046    0    0    0    0    0    0]
 [   0    0    1   54    0    0    0    0    0]
 [   0    0    0    0  348    0    0    0    0]
 [   0    0    0    0    0  740    0    0    1]
 [   0    0    0    0    0    0   32    0    0]
 [   0    0    0    0    0    0    0  366    0]
 [   0    0    0    0    0    0    0    0  440]]
0.9951525425179455
K=3: 0.8597069475656621
[[  52    0    0    0    0    0    0    0    0]
 [  12    0    0    0    0    0    0    0    0]
 [   0    0 1030    0    0    0    0    0    0]
 [  61    0    0    0    0    1    0    0    0]
 [   0    0    0    0  366    0    0    0    0]
 [   8    0    0    0    0  726    0    0    0]
 [   4    0    0    0    0    0   25    0    0]
 [   0    0    0    0    0    0    0  359    0]
 [ 448    0    0    0    0    0    0    0    0]]
0.8149482817578494
K=4: 0.8662037474311067
0.8102312239815377

"""