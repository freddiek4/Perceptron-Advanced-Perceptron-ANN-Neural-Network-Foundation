This project covers the implementation of two machine learning models from scratch: 1. Perceptron: A simple linear classifier. 2. Artificial Neural Network (ANN): A multi-layer perceptron model for classifying dry beans using a real-world dataset. Each part of the project demonstrates the basic principles of machine learning: training a model to learn patterns from data and making predictions on new data.

Part 1: Perceptron from Scratch

What is a Perceptron? A Perceptron is the simplest form of a neural network. It is a linear classifier that tries to find a straight line (or hyperplane) to separate two classes. The model adjusts its weights through a learning process until it can correctly classify the input data.

How It Works: 1. Inputs and Weights: The perceptron takes inputs (features) and multiplies them by weights. 2. Activation: It sums the results and applies a threshold to decide if the output is class 1 or -1. 3. Learning: The model updates its weights whenever it makes a wrong prediction, pushing the decision boundary to better separate the two classes.

Usage: The Perceptron model is implemented from scratch and trained on a synthetic dataset with two features. The accuracy of the model is evaluated on a test set, and its decision boundary is visualized using a scatter plot.

Part 2: Artificial Neural Network for Dry Beans Classification

What is an Artificial Neural Network (ANN)? An Artificial Neural Network (ANN) is a more complex version of a perceptron. It contains multiple layers of neurons and is capable of learning non-linear patterns. ANNs are used for various tasks, such as classification, regression, and pattern recognition. In this project, we use an ANN to classify seven different types of dry beans based on 16 attributes related to size and shape.

How It Works: 1. Input Layer: Receives the 16 features of each bean. 2. Hidden Layers: Two hidden layers process the inputs and extract complex patterns. 3. Output Layer: Produces the final classification of the bean type (e.g., Seker, Barbunya, etc.). 4. Training: The model learns by adjusting its weights using a process called backpropagation, minimizing the error between the predicted and actual outputs. 5. Evaluation: After training, the model is evaluated using accuracy, mean squared error (MSE), precision, and recall.

Usage: The Dry Beans Dataset is used, which contains 13,611 instances of seven different bean types. A feed-forward neural network is trained using stochastic gradient descent (SGD). The network’s performance is evaluated using a confusion matrix and other metrics.

Getting Started

Prerequisites The following Python libraries are required: pandas, numpy, matplotlib, seaborn, scikit-learn You can install these dependencies by running: pip install pandas numpy matplotlib seaborn scikit-learn

Running the Perceptron Model 1. Load the Perceptron code. 2. Generate or load a simple binary classification dataset. 3. Train the perceptron model on the training set and evaluate its performance on the test set.

Running the ANN Model 1. Download the Dry Beans Dataset from UCI Machine Learning Repository. 2. Preprocess the dataset using one-hot encoding and min-max normalization. 3. Train the neural network with the specified hyperparameters. 4. Evaluate the performance using accuracy, MSE, precision, recall, and confusion matrices.

Evaluation Metrics

Accuracy: Proportion of correctly predicted instances. Mean Squared Error (MSE): Average squared difference between predicted and actual values. Precision: Proportion of true positives out of all predicted positives. Recall: Proportion of true positives out of all actual positives.

Files perceptron.py: Contains the Perceptron class and training script. ann_drybeans.py: Contains the ANN implementation for classifying dry beans. Dry_Beans_Dataset.csv: The dataset used for the ANN model.

Results

The Perceptron model achieves high accuracy on linearly separable data. The ANN model successfully classifies dry beans with an overall accuracy of approximately 90% and reasonable precision and recall for each class.

Conclusion

This project demonstrates the basics of machine learning with simple and multi-layer perceptrons. The perceptron model provides a good introduction to linear classifiers, while the ANN shows the power of neural networks in handling more complex data.
