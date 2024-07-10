import pandas as pd
import json
from joblib import load
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np


def pipeline_test():
    
    model_path = '/opt/airflow/dags/data/outputs/model_final.pkl'
    test_path = '/opt/airflow/dags/data/temp/test_temp.csv'

    # Leer data test
    dfTest = pd.read_csv(test_path)
    dfTest_Pred = dfTest.drop(columns = ['id'])

    # Leer el modelo 
    model = load(model_path)

    # Predecir
    colPredict = model.predict(dfTest_Pred)
    dfTest['target'] = colPredict.tolist()

    return dfTest

    
def submitt_test(ti):
    dfPredict = ti.xcom_pull(task_ids='pipeline_test')
    
    # Generando df output
    result = dfPredict[['id','target']].rename(columns = {'target':'Target'})

    result.to_csv( '/opt/airflow/dags/data/outputs/submission_final.csv', index = False)

    return result

def post_test_kaggle(ti):
    result = ti.xcom_pull(task_ids='submitt_test')

    # Parametrias para cargar información
    with open('/home/airflow/.kaggle/kaggle.json') as file:
        credentialsKaggle = json.load(file)

    os.environ['KAGGLE_USERNAME'] = credentialsKaggle['username']
    os.environ['KAGGLE_KEY'] = credentialsKaggle['key']

    api = KaggleApi()
    api.authenticate()
    
    competition_name = 'playground-series-s4e6'
    file_path = '/opt/airflow/dags/data/outputs/submission_final.csv'
    num_random = np.random.randint(100)

    # Enviando información
    api.competition_submit(file_name=file_path,
                        message=f"Submission número {num_random}",
                        competition=competition_name)

    
    