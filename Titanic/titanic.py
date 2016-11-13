'''
Author: Maroof
Dataset: https://www.kaggle.com/c/titanic/data
'''

''' Import all libraries '''
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import preprocess
import seaborn as sns

'''--------------------------------------------------------------------'''
''' Initialize variables '''

trainData_fileName = "train.csv"
testData_fileName = "test.csv"
testTarget_fileName = "gendermodel.csv"

'''--------------------------------------------------------------------'''
''' Setting up train data and train target '''

# Read data from csv file
trainData = pd.read_csv(trainData_fileName)
trainData = trainData.set_index('PassengerId')  # Set the index as passenger ID
trainTarget = trainData['Survived']     # copying the target values to trainTarget
del trainData['Survived']           # deleting the target column in traindata
# print(trainData.head(n=5))

'''--------------------------------------------------------------------'''
''' Setting up test data and test target '''

testData = pd.read_csv(testData_fileName)
testData = testData.set_index('PassengerId')
testTarget = pd.read_csv(testTarget_fileName)
testTarget = testTarget.set_index('PassengerId')
# print(testTarget.head(n=5))

'''--------------------------------------------------------------------'''
''' Display test and train features '''

# trainData.info()
# print('-----------------------------------------------')
# testData.info()

'''--------------------------------------------------------------------'''
''' Pre-processing data '''

# Dropping useless information:
trainData.drop(['Name','Ticket'],axis = 1, inplace=True)
testData.drop(['Name','Ticket'],axis = 1, inplace=True)

# Mode imputation of missing values
trainData['Embarked'] = trainData['Embarked'].fillna('S')
# print(trainData['Embarked'].value_counts(dropna=False))



