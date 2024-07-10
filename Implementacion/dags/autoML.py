import pandas as pd
import numpy as np
import json
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold
import lightgbm as lgbm
from joblib import dump


path_train = '/opt/airflow/dags/data/temp/train_temp.csv'
path_test = '/opt/airflow/dags/data/temp/test_temp.csv'
    
def load_data(path):
    return pd.read_csv(path)


def split_data(df):
    X = df.drop(columns = ['id','target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    return X_train, X_test, y_train, y_test

def model():
    # return lgbm.LGBMClassifier(n_jobs = -1, objective = 'multiclass', 
    #                         eval_metric= 'accuracy', verbose= 200, random_state=123, 
    #                         n_estimators =  300, min_samples_split = 10, max_depth = 5, 
    #                         learning_rate = '0.05')
    return RandomForestClassifier(n_jobs = -1, verbose= 200, random_state=123)



def random_search(model):
    # Parametros
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=123)
    cv = 5
    param_dist_lgbm = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'min_samples_split': [5, 10], 
            'learning_rate':['0.01','0.05']
    }

    param_dist_rf = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'min_samples_split': [5, 10]
    }

    return RandomizedSearchCV(estimator = model, n_iter = 10, n_jobs = -1, cv = cv, 
                                        random_state = 123, scoring = "accuracy", 
                                        param_distributions = param_dist_rf, 
                                        error_score='raise')



def fit_model(model,X,y):
    model.fit(X,y)


def print_score(model,X,y):
    print(model.score(X,y))


def save_model(model):
    output_model = '/opt/airflow/dags/data/outputs/model_final.pkl'
    dump(model, output_model)


def running_model():
    dfTrain = load_data(path_train)
    X_train, X_test, y_train, y_test = split_data(dfTrain)
    modelo_init_ = model()
    model_rf_random = random_search(modelo_init_)
    fit_model(model_rf_random, X_train, y_train)
    print_score(model_rf_random,X_test,y_test)
    model_final = model_rf_random.best_estimator_
    save_model(model_final)


