"""This is a python program where we are reading CSV file with the dataset of Breast Cancer with label of Benign or Malignant and features attached with it. We are shuffling the data set and we will use first 75% of the data set in training set and other 25% of the data set in testing data. We use this data to plot scatter plots for three pairs of the features. Next, we are plotting the bar chart for the frequency of the labels.

Source for the data set : http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Prerak Patel, Student, Mohawk College, 2020
"""

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

# function to normalize data
def noramalizeData(train_data,test_data):
    # loop through all the columns in the training data
    for index in range(np.size(train_data,1)):
        # check if the maximum value of the column is greater than 1
        # (tried) if train_data[:,index:index+1].max() > 1:
        # updating training data coloumn with the normalized data
        train_data[:,index:index+1] = (train_data[:,index:index+1] - train_data[:,index:index+1].min())/train_data[:,index:index+1].max()
        # updating testing data coloumn with the normalized data
        test_data[:,index:index+1] = (test_data[:,index:index+1] - test_data[:,index:index+1].min())/test_data[:,index:index+1].max()

    return train_data, test_data

# function to classify if the testing item is "Benign" or "Malignant"
def classify(train_data,test_row,train_labels,k,power,root):
    # declaring dictionary to let labels vote
    labelsDictionary = {
    "benign": 0,
    "malignant": 0
    }
    # calculating the distance between testing item and training items
    distance = (abs((train_data - test_row)**power)).sum(axis=1)**root
    # getting the sorting sequence for distance in ascending
    sort = distance.argsort()
    # sorting the array in ascending to find out the closest training points to test item
    distanceSorted = train_labels[sort]
    # seperating the closest points to test item
    closestPoints = distanceSorted[:k]
    # looping through the closest training point to check the labels on them
    for eachPoint in closestPoints:
        # check if the label is "Benign" then update the vote value
        if eachPoint == 'B':
            labelsDictionary["benign"] += 1
        # check if the label is "Malignant" then update the vote value
        if eachPoint == 'M':
            labelsDictionary["malignant"] += 1

    # check if the "Benign" has more votes than "Malignant" then classify the test item as "Benign"
    if labelsDictionary["benign"] > labelsDictionary["malignant"]:
        return 'B'
    # else classify the test item as "Malignant"
    else:
        return 'M'

# function to calculate accuracy of algorithm and return the array of results
def accuracyCalculation(train_data,test_data,train_labels,test_labels,votes,power,root):
    # keeping track of number of test items
    test_count = 0
    # keeping track of number of times algorithm was right to label the test item
    success_counter = 0
    # looping through test data
    for test_row in test_data:
        # calling the classify function to label the current test item
        result = classify(train_data,test_row,train_labels,votes,power,root)
        #  checking if the label given to the test item is correct or not
        if test_labels[test_count] == result:
            success_counter += 1
        test_count += 1
    # calculating the percentage times the algorithm was correct in labelling the test item
    accuracy = (success_counter*100)/test_count

    return accuracy

# function to use Euclid Distance algorithm to label the testing item
def euclidDistance(frequency,votes,power,root):
    # array to store the results for number of runs
    accuracyArray = []
    # looping through algorithm mulitples times till it reaches frequency
    for time in range(frequency):
        # shuffling the data set
        train_data, train_labels, test_data, test_labels = createDataSet(data_set)
        # calculating accuracy using accuracyCalculation function
        accuracy = accuracyCalculation(train_data,test_data,train_labels,test_labels,votes,power,root)
        accuracyArray.append(accuracy)
    return np.array(accuracyArray)

# function to use Euclid Distance algorithm to label the testing item with Normalized data
def euclidDistanceNormalized(frequency,votes,power,root):
    # array to store the results for number of runs
    accuracyArray = []
    # looping through algorithm mulitples times till it reaches frequency
    for time in range(frequency):
        # shuffling the data set
        train_data, train_labels, test_data, test_labels = createDataSet(data_set)
        # calling noramalizeData function to normalize data
        normalize_train_data, normalize_test_data = noramalizeData(train_data,test_data)
        # calculating accuracy using accuracyCalculation function
        accuracy = accuracyCalculation(train_data,test_data,train_labels,test_labels,votes,power,root)
        accuracyArray.append(accuracy)
    return np.array(accuracyArray)

# function to use Minkowski Distance algorithm to label the testing item
def minkowskiDistance(frequency,votes,power,root):
    # array to store the results for number of runs
    accuracyArray = []
    # looping through algorithm mulitples times till it reaches frequency
    for time in range(frequency):
        # shuffling the data set
        train_data, train_labels, test_data, test_labels = createDataSet(data_set)
        # calculating accuracy using accuracyCalculation function
        accuracy = accuracyCalculation(train_data,test_data,train_labels,test_labels,votes,power,root)
        accuracyArray.append(accuracy)
    return np.array(accuracyArray)

# function to use Minkowski Distance algorithm to label the testing item with Normalized data
def minkowskiDistanceNormalized(frequency,votes,power,root):
    # array to store the results for number of runs
    accuracyArray = []
    # looping through algorithm mulitples times till it reaches frequency
    for time in range(frequency):
        # shuffling the data set
        train_data, train_labels, test_data, test_labels = createDataSet(data_set)
        # calling noramalizeData function to normalize data
        normalize_train_data, normalize_test_data = noramalizeData(train_data,test_data)
        # calculating accuracy using accuracyCalculation function
        accuracy = accuracyCalculation(normalize_train_data,normalize_test_data,train_labels,test_labels,votes,power,root)
        accuracyArray.append(accuracy)
    return np.array(accuracyArray)

# function to use Manhattan Distance algorithm to label the testing item
def manhattanDistance(frequency,votes,power,root):
    # array to store the results for number of runs
    accuracyArray = []
    # looping through algorithm mulitples times till it reaches frequency
    for time in range(frequency):
        # shuffling the data set
        train_data, train_labels, test_data, test_labels = createDataSet(data_set)
        # calculating accuracy using accuracyCalculation function
        accuracy = accuracyCalculation(train_data,test_data,train_labels,test_labels,votes,power,root)
        accuracyArray.append(accuracy)
    return np.array(accuracyArray)

# function to use Manhattan Distance algorithm to label the testing item with Normalized data
def manhattanDistanceNormalized(frequency,votes,power,root):
    # array to store the results for number of runs
    accuracyArray = []
    # looping through algorithm mulitples times till it reaches frequency
    for time in range(frequency):
        # shuffling the data set
        train_data, train_labels, test_data, test_labels = createDataSet(data_set)
        # calling noramalizeData function to normalize data
        normalize_train_data, normalize_test_data = noramalizeData(train_data,test_data)
        # calculating accuracy using accuracyCalculation function
        accuracy = accuracyCalculation(normalize_train_data,normalize_test_data,train_labels,test_labels,votes,power,root)
        accuracyArray.append(accuracy)
    return np.array(accuracyArray)


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



## PRINTING IT
# set the number of the runs for each variation
frequency = 500

## Euclidean Distance
accuracyResult = euclidDistance(frequency, 3, 2, 0.5)
print("K=3, Euclidean Distance \nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")

accuracyResult = euclidDistance(frequency, 5, 2, 0.5)
print("K=5, Euclidean Distance \nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")

## Minkowski Distance
accuracyResult = minkowskiDistance(frequency, 3, 3, 3)
print("K=3, Minkowski Distance \nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")

accuracyResult = minkowskiDistance(frequency, 5, 3, 3)
print("K=5, Minkowski Distance \nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")


## Manhattan Distance
accuracyResult = manhattanDistance(frequency, 3, 1, 1)
print("K=3, Manhattan Distance \nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")

accuracyResult = manhattanDistance(frequency, 5, 1, 1)
print("K=5, Manhattan Distance \nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")

## Euclidean Distance, Normalized
accuracyResult = euclidDistanceNormalized(frequency, 3, 2, 0.5)
print("K=3, Euclidean Distance, Normalized\nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")

accuracyResult = euclidDistanceNormalized(frequency, 5, 2, 0.5)
print("K=5, Euclidean Distance, Normalized\nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")

## Minkowski Distance, Normalized
accuracyResult = minkowskiDistanceNormalized(frequency, 3, 3, 3)
print("K=3, Minkowski Distance, Normalized\nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")

accuracyResult = minkowskiDistanceNormalized(frequency, 5, 3, 3)
print("K=5, Minkowski Distance, Normalized\nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")

## Manhattan Distance, Normalized
accuracyResult = manhattanDistanceNormalized(frequency, 3, 1, 1)
print("K=3, Manhattan Distance, Normalized\nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")

accuracyResult = manhattanDistanceNormalized(frequency, 5, 1,1)
print("K=5, Manhattan Distance, Normalized\nAverage Accuracy: " + str(np.around(accuracyResult.mean(), decimals=1)) + "\n")