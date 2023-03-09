#Import relevant packages
import pandas as pd
import os
from data_preprocessing import preprocessing_pipeline
from train_models import train_rfc, train_svc

#Define our files
training_file = 'datasets/titanic/train.csv'
testing_file = 'datasets/titanic/test.csv'

#Categorise our attributes
numerical_attribs = ["Age", "SibSp", "Parch", "Fare"]
categorical_attribs = ["Pclass", "Sex", "Embarked"]

#Define our index column
index_col = "PassengerId"

#Create our pipeline
pp_func_res = preprocessing_pipeline(training_file, testing_file, index_col, numerical_attribs, categorical_attribs)

training_data, testing_data, pl= pp_func_res[0], pp_func_res[1], pp_func_res[2]

X_train = pl.fit_transform(training_data[numerical_attribs + categorical_attribs])
y_train = training_data["Survived"]

print(train_rfc(pl, training_file, testing_file, index_col, numerical_attribs, categorical_attribs))
print(train_svc(pl, training_file, testing_file, index_col, numerical_attribs, categorical_attribs))


