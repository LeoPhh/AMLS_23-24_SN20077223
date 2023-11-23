from A.pneumoniaSVM import PneumoniaSVMClassifier
from A.pneumoniaCNN import PneumoniaCNNClassifier

if __name__ == "__main__":

    SVM = PneumoniaSVMClassifier('./Datasets/pneumoniamnist.npz')
    CNN = PneumoniaCNNClassifier('./Datasets/pneumoniamnist.npz')