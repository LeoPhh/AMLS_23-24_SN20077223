# Import the models
from A.pneumoniaSVM import PneumoniaSVMClassifier
from A.pneumoniaCNN import PneumoniaCNNClassifier
from B.pathCNN import PathCNNClassifier

if __name__ == "__main__":
    # Create class instances for the two Task A models
    TaskA_SVM = PneumoniaSVMClassifier('./Datasets/pneumoniamnist.npz')
    TaskA_CNN = PneumoniaCNNClassifier('./Datasets/pneumoniamnist.npz')

    # Create class instance for the Task B model
    TaskB_CNN = PathCNNClassifier('./Datasets/pathmnist.npz')

    # Task A - SVM
    # TaskA_SVM.print_individual_label_counts()
    TaskA_SVM.train_model()
    TaskA_SVM.test_model()

    # Task A - CNN
    TaskA_CNN.train_without_cross_validation(epochs=35, batch_size=32)
    TaskA_CNN.train_with_cross_validation(epochs=40, batch_size=32, folds=10)

    # Task B - CNN
    # TaskB_CNN.print_individual_label_counts()
    TaskB_CNN.train_without_cross_validation(epochs=25, batch_size=64)
    TaskB_CNN.train_with_cross_validation(epochs=25, batch_size=64, folds=5)

    # Method to plot the sample image in histogram for the report
    TaskA_SVM.display_sample_histogram()

    # Methods to plot the training process and convergence for the CNNs in Task A and B
    # Note: in order for these methods to work, you must first train the CNNs
    TaskA_CNN.plot_training_process()
    TaskB_CNN.plot_training_process()