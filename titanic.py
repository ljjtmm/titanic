#Import relevant packages
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#Load data
training_data = pd.read_csv('datasets/titanic/train.csv')
testing_data = pd.read_csv('datasets/titanic/test.csv')

#Index the data
training_data = training_data.set_index("PassengerId")
testing_data = testing_data.set_index("PassengerId")

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

#Define attribute types, and combine into a single pipeline
numerical_attribs = ["Age", "SibSp", "Parch", "Fare"]
categorical_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
        ("num", numerical_pipeline, numerical_attribs),
        ("cat", categorical_pipeline, categorical_attribs),
    ])