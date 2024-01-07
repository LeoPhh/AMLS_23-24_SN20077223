# AMLS Assignment 2023/24

This repository contains the models built for the AMLS Final Assignment. The assigment consisted of two tasks: Task A and Task B, and all code files for each task can be found in the folder with the corresponding task letter (e.g. the code files for Task A can be found in folder 'A'). The Datasets folder is intentionally empty, as instructed in the assignment brief. This was done by including a .gitignore file which ignores the data files ('pathmnist.npz' and 'pneumoniamnist.npz') when pushing the code to the remote repository. The code follows an Object-Oriented approach, as this made it more reusable and modifiable.

The models are laid out as follows: Folder 'A' contains two code files, one for each model (SVM and CNN) for Task A. Folder 'B' contains one code file, as only one CNN model was used for Task B. Specifically:

- pneumoniaSVM.py (in folder A) contains the SVM code for Task A
- pneumoniaCNN.py (in folder A) contains the CNN code for Task A
- pathCNN.py (in folder B) contains the CNN code for Task B

All files can be run from the main.py file in the root directory. The code is designed in such a way that the user has the flexibility to modify a few parameters when calling the methods from the main.py file, such as the number of epochs, the batch size and the cross validation folds for the CNN models. I have set the main.py file to call all the important methods from each model, however it is recommended that the user comments out most of them and test each one-by-one, because running all of them with one execution would result in too much information being displayed in the terminal.

There are also some other methods in the model files that entail plotting graphs such as histograms or showing the training/validation accuracy convergence while training a CNN model. These methods were used to generate plots for the final report.

The libraries required to run this code are:

- matplotlib
- numpy
- keras
- sklearn