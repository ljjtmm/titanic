from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from data_preprocessing import preprocessing_pipeline

def train_rfc(pipeline,training_file,testing_file, index_col, numerical_attribs, categorical_attribs):
    #Read in result of preprocessing_pipeline function
    pp_func_res = preprocessing_pipeline(training_file, testing_file, index_col, numerical_attribs, categorical_attribs)
    training_data, testing_data, pl= pp_func_res[0], pp_func_res[1], pp_func_res[2]

    #Create training set
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

    res_string = "Mean of Cross-Validation score for RF Classifer model is: " + str(round(forest_scores.mean(),2))

    return res_string

def train_svc(pipeline,training_file,testing_file, index_col, numerical_attribs, categorical_attribs):
    #Read in result of preprocessing_pipeline function
    pp_func_res = preprocessing_pipeline(training_file, testing_file, index_col, numerical_attribs, categorical_attribs)
    training_data, testing_data, pl= pp_func_res[0], pp_func_res[1], pp_func_res[2]

    #Create training set
    X_train = pl.fit_transform(training_data[numerical_attribs + categorical_attribs])
    y_train = training_data["Survived"]

    svm_clf = SVC(gamma="auto")
    svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
    res_string = "Mean of Support Vector Classififier model scores :" + str(round(svm_scores.mean(), 2)) 

    return res_string