#Import relevant packages
import pandas as pd
import os

#Load data
training_data = pd.read_csv('datasets/titanic/train.csv')
testing_data = pd.read_csv('datasets/titanic/test.csv')

print(training_data.columns)