# Import the models
from A.pneumoniaSVM import PneumoniaSVMClassifier
from A.pneumoniaCNN import PneumoniaCNNClassifier
from B.pathCNN import PathCNNClassifier

if __name__ == "__main__":
    """
    This code contains lines which execute all the models discussed in the report.

    It is recommended that when testing these, they are executed separately to avoid
    overcrowding the terminal with too much information. Specifically, the following lines 
    must always be executed:

    # Create class instances for the two Task A models
    TaskA_SVM = PneumoniaSVMClassifier('./Datasets/pneumoniamnist.npz')
    TaskA_CNN = PneumoniaCNNClassifier('./Datasets/pneumoniamnist.npz')

    # Create class instance for the Task B model
    TaskB_CNN = PathCNNClassifier('./Datasets/pathmnist.npz')

    --------------------------------------------

    Then, the following lines can be run while commenting everything else out.
    These lines train and test the SVM model for Task A:

    TaskA_SVM.train_model()
    TaskA_SVM.test_model()

    --------------------------------------------

    This line trains and tests the CNN model for Task A without cross validation:

    TaskA_CNN.train_without_cross_validation(epochs=35, batch_size=32)

    --------------------------------------------

    This line trains and tests the CNN model for Task A with cross validation:

    TaskA_CNN.train_with_cross_validation(epochs=40, batch_size=32, folds=10)

    --------------------------------------------

    This line trains and tests the CNN model for Task B without cross validation:

    TaskB_CNN.train_without_cross_validation(epochs=25, batch_size=64)

    --------------------------------------------

    Finally, this line trains and tests the CNN model for Task B with cross validation:

    TaskB_CNN.train_with_cross_validation(epochs=25, batch_size=64, folds=5)

    """
    # Create class instances for the two Task A models
    TaskA_SVM = PneumoniaSVMClassifier('./Datasets/pneumoniamnist.npz')
    TaskA_CNN = PneumoniaCNNClassifier('./Datasets/pneumoniamnist.npz')

    # Create class instance for the Task B model
    TaskB_CNN = PathCNNClassifier('./Datasets/pathmnist.npz')

    # Task A - Train and Test SVM Model
    TaskA_SVM.train_model()
    TaskA_SVM.test_model()

    # Task A - Train and Test CNN without CV
    TaskA_CNN.train_without_cross_validation(epochs=35, batch_size=32)

    # Task A - Train and Test CNN with CV
    TaskA_CNN.train_with_cross_validation(epochs=40, batch_size=32, folds=10)

    # Task B - Train and Test CNN without CV
    TaskB_CNN.train_without_cross_validation(epochs=25, batch_size=64)

    # Task B - Train and Test CNN with CV
    TaskB_CNN.train_with_cross_validation(epochs=25, batch_size=64, folds=5)