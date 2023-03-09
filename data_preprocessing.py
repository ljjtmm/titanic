#Import relevant packages
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_pipeline(training_file,testing_file, index_col, numerical_attribs, categorical_attribs):
    #Load the data
    training_data = pd.read_csv(training_file)
    testing_data = pd.read_csv(testing_file)

    #Index the data
    training_data = training_data.set_index(index_col)
    testing_data = testing_data.set_index(index_col)

    #Create numerical pipeline
    numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    #Create categorical pipeline
    categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cat_encoder", OneHotEncoder(sparse=False)),
        ])
    
    preprocess_pipeline = ColumnTransformer([
            ("num", numerical_pipeline, numerical_attribs),
            ("cat", categorical_pipeline, categorical_attribs),
        ])

training_file = 'datasets/titanic/train.csv'
testing_file = 'datasets/titanic/test.csv'

numerical_attribs = ["Age", "SibSp", "Parch", "Fare"]
categorical_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline(training_file, testing_file, "PassengerId", numerical_attribs, categorical_attribs)