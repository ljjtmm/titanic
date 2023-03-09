#Import relevant packages
import pandas as pd
import os
from data_preprocessing import preprocess_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

#Define our files
training_file = 'datasets/titanic/train.csv'
testing_file = 'datasets/titanic/test.csv'

#Categorise our attributes
numerical_attribs = ["Age", "SibSp", "Parch", "Fare"]
categorical_attribs = ["Pclass", "Sex", "Embarked"]

#Create our pipeline
pl = preprocess_pipeline(training_file, testing_file, "PassengerId", numerical_attribs, categorical_attribs)


#Define our training sets
training_data = pd.read_csv(training_file)
testing_data = pd.read_csv(testing_file)

X_train = pl.fit_transform(training_data[numerical_attribs + categorical_attribs])
y_train = training_data["Survived"]

#Train a Random Forest classifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)

#Create RF predictions
X_test = pl.transform(testing_data[numerical_attribs + categorical_attribs])
y_pred = forest_clf.predict(X_test)

#Find the mean of our predictions using cross-validation
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print("Result of cross-validation for Random Foest model: ",forest_scores.mean()) #Returns 0.8137578027465668

#Test a Support Vector Classifier
svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print("Mean of Support Vector Classififier model scores :",svm_scores.mean()) #Returns 0.8249313358302123

