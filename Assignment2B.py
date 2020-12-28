"""This is a python program where we are reading CSV file with the dataset of Breast Cancer with label of Benign or Malignant and features attached with it. We are shuffling the data set and we will use first 80% of the data set in training set and other 20% of the data set in testing data. We will use Decision Tree Algorithm to train and test the data to return the accuracy of the algorithm.

Source for the data set : http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Prerak Patel, Student, Mohawk College, 2020
"""

from sklearn import tree
import numpy as np
import csv

## DECLARE FUNCTIONS

# function to create shuffled data set and using first 80% of data set for training and remaining 20% for testing data
def createDataSet(data_set):
    # shuffle the data set
    np.random.shuffle(data_set)
    # seperating the attributes data from labels and ID number
    data = (data_set[:,2:].astype(np.float)).tolist()
    data = np.array(data)
    # seperating labels and storing
    labels = data_set[:,1:2]
    # using the first 80% of the data set into training data
    train_data = data[:int(len(data_set)*0.8)]
    train_labels = labels[:int(len(data_set)*0.8)]

    # using the other 20% of the data set into testing data
    test_data = data[int(len(data_set)*0.8):]
    test_labels = labels[int(len(data_set)*0.8):]

    return train_data, train_labels, test_data, test_labels

# function to calculate accuracy of the algorithm based on the Decision Tree Classifier and returning the average of the total runs
def calculateAccuracy(clf,total_runs):
    # declaring the training accuracy array
    training_accuracy_array = []
    # declaring the testing accuracy array
    testing_accuracy_array = []
    # looping through the total number of runs
    for each_run in range(total_runs):
        # creating new data set before running the algorithm
        train, train_target, test, test_target = createDataSet(data_set)
        # making 2d array of train targets into 1d using function ravel()
        train_target = train_target.ravel()
        # making 2d array of test targets into 1d using function ravel()
        test_target = test_target.ravel()

        # passing the training data to the decision tree classifier to train
        clf = clf.fit(train, train_target)

        # predicting the class of samples
        training_prediction = clf.predict(train)
        # returning the total correct prediction by the training data
        training_correct = (training_prediction == train_target).sum()
        # returning the percentage value of the accuracy for the training data
        training_accuracy = training_correct/len(training_prediction)*100

        # predicting the class of samples
        testing_prediction = clf.predict(test)
        # returning the total correct prediction by the testing data
        testing_correct = (testing_prediction == test_target).sum()
        # returning the percentage value of the accuracy for the testing data
        testing_accuracy = testing_correct/len(testing_prediction)*100

        # add accuracy data for the current run for training data to the accuracy array
        training_accuracy_array.append(training_accuracy)
        # add accuracy data for the current run for testing data to the accuracy array
        testing_accuracy_array.append(testing_accuracy)
    # returning training accuracy array and testing accuracy array
    return np.array(training_accuracy_array), np.array(testing_accuracy_array)

## READ IT

# opening the CSV file here
train_data_file = open("wdbc.csv","r")

# creating CSV readers
csv_reader = csv.reader(train_data_file, delimiter=",")

# declaring the arrays for the storing data
train_data = []
train_labels = []
test_data = []
test_labels = []
header_count = 0
for row in csv_reader:
    if header_count == 0:
        # reading features
        feature_names = row
        header_count += 1
        continue
    # train_data

    train_data += [[str(num) for num in row]]

# convert to NumPy arrays
data_set = np.array(train_data)
feature_names = np.array(feature_names)
target_names = np.array(['malignant', 'benign'])

# variable to change the total number of runs to carry out
total_runs = 500


# Declaring default decision tree
clf = tree.DecisionTreeClassifier()
training_accuracy_array, testing_accuracy_array = calculateAccuracy(clf,total_runs)
print("Decision Tree Classifier: Default with no args\nTraining Data Accuracy: " + str(np.around(training_accuracy_array.mean(), decimals=2)) + "\nTesting Data Accuracy: " + str(np.around(testing_accuracy_array.mean(), decimals=1)) + "\n")

# Declaring decision tree with criterion = entropy
clf = tree.DecisionTreeClassifier(criterion = "entropy")
training_accuracy_array, testing_accuracy_array = calculateAccuracy(clf,total_runs)
print("Decision Tree Classifier: criterion = entropy\nTraining Data Accuracy: " + str(np.around(training_accuracy_array.mean(), decimals=2)) + "\nTesting Data Accuracy: " + str(np.around(testing_accuracy_array.mean(), decimals=1)) + "\n")

# Declaring decision tree with criterion = "entropy", min_samples_split = 30
clf = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_split = 30)
training_accuracy_array, testing_accuracy_array = calculateAccuracy(clf,total_runs)
print("Decision Tree Classifier: criterion = entropy, min_samples_split = 30\nTraining Data Accuracy: " + str(np.around(training_accuracy_array.mean(), decimals=2)) + "\nTesting Data Accuracy: " + str(np.around(testing_accuracy_array.mean(), decimals=1)) + "\n")

# Declaring decision tree with criterion = "entropy", max_features = 2
clf = tree.DecisionTreeClassifier(criterion = "entropy", max_features = 2)
training_accuracy_array, testing_accuracy_array = calculateAccuracy(clf,total_runs)
print("Decision Tree Classifier: criterion = entropy, max_features = 2\nTraining Data Accuracy: " + str(np.around(training_accuracy_array.mean(), decimals=2)) + "\nTesting Data Accuracy: " + str(np.around(testing_accuracy_array.mean(), decimals=1)) + "\n")

# Declaring decision tree with criterion = "entropy", max_leaf_nodes = 10, min_samples_split = 30
clf = tree.DecisionTreeClassifier(criterion = "entropy", max_leaf_nodes = 10, min_samples_split = 30)
training_accuracy_array, testing_accuracy_array = calculateAccuracy(clf,total_runs)
print("Decision Tree Classifier: criterion = entropy, max_leaf_nodes = 10, min_samples_split = 30\nTraining Data Accuracy: " + str(np.around(training_accuracy_array.mean(), decimals=2)) + "\nTesting Data Accuracy: " + str(np.around(testing_accuracy_array.mean(), decimals=1)) + "\n")

# Declaring decision tree with criterion = "entropy", max_depth=4
clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth=4)
training_accuracy_array, testing_accuracy_array = calculateAccuracy(clf,total_runs)
print("Decision Tree Classifier: criterion = entropy, max_depth = 5\nTraining Data Accuracy: " + str(np.around(training_accuracy_array.mean(), decimals=2)) + "\nTesting Data Accuracy: " +  str(np.around(testing_accuracy_array.mean(), decimals=1)) + "\n")

# Declaring decision tree with criterion = "entropy", max_leaf_nodes = 20, max_depth = 4
clf = tree.DecisionTreeClassifier(criterion = "entropy", max_leaf_nodes = 20, max_depth = 4)
training_accuracy_array, testing_accuracy_array = calculateAccuracy(clf,total_runs)
print("Decision Tree Classifier: criterion = entropy, max_leaf_nodes = 20, max_depth = 4\nTraining Data Accuracy: " + str(np.around(training_accuracy_array.mean(), decimals=2)) + "\nTesting Data Accuracy: " + str(np.around(testing_accuracy_array.mean(), decimals=1)) + "\n")

# Declaring decision tree with criterion = "entropy", max_leaf_nodes = 15
clf = tree.DecisionTreeClassifier(criterion = "entropy", max_leaf_nodes = 15)
training_accuracy_array, testing_accuracy_array = calculateAccuracy(clf,total_runs)
print("Decision Tree Classifier: criterion = entropy, max_leaf_nodes = 15\nTraining Data Accuracy: " + str(np.around(training_accuracy_array.mean(), decimals=2)) + "\nTesting Data Accuracy: " + str(np.around(testing_accuracy_array.mean(), decimals=1)) + "\n")

# to create a tree.dot.pdf
import graphviz
dot_data = tree.export_graphviz(clf,
    out_file=None,
    feature_names = feature_names[2:],
    class_names = target_names,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("tree.dot")











