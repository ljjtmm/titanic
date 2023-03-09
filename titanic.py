#Import relevant packages
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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

#Define our training sets
X_train = preprocess_pipeline.fit_transform(training_data[numerical_attribs + categorical_attribs])
y_train = training_data["Survived"]

#Train a Random Forest classifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)

#Create RF predictions
X_test = preprocess_pipeline.transform(testing_data[numerical_attribs + categorical_attribs])
y_pred = forest_clf.predict(X_test)

#Find the mean of our predictions using cross-validation
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print("Result of cross-validation for Random Foest model: ",forest_scores.mean()) #Returns 0.8137578027465668

