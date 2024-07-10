import pandas as pd
import numpy as np
import json
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Parametrias para descargar información
with open('/home/airflow/.kaggle/kaggle.json') as file:
    credentialsKaggle = json.load(file)

os.environ['KAGGLE_USERNAME'] = credentialsKaggle['username']
os.environ['KAGGLE_KEY'] = credentialsKaggle['key']



def descarga_informacion():
    api = KaggleApi()
    api.authenticate()

    # Download the competition files
    competition_name = 'playground-series-s4e6'
    download_path = '/opt/airflow/dags/data/inputs'
    api.competition_download_files(competition_name, path=download_path)

    # Unzip the downloaded files
    for item in os.listdir(download_path):
        if item.endswith('.zip'):
            zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
            zip_ref.extractall(download_path)
            zip_ref.close()
            print(f"Unzipped {item}")


def load_data(path):
    return pd.read_csv(path)

def rename_columns(df):
    
    # Leer nombres de columnas
    ruta_nombres = '/opt/airflow/dags/data/inputs/dict_nombres.json'
    with open(ruta_nombres) as file:
        dict_nombres = json.load(file)

    # Cambiar nombres de columna
    return df.rename(columns = dict_nombres) 


def feature_engineering(df):
    
    # Leer la segmentacion de columnas
    ruta_columnas = '/opt/airflow/dags/data/inputs/seg_columnas.json'
    with open(ruta_columnas) as file:
        dict_columnas = json.load(file)

    # Crear dummies
    for col in dict_columnas["col_categoricas"]:
        dfDummies = pd.get_dummies(df[col], prefix = col).astype(int)
        df = pd.concat([df, dfDummies], axis = 1)
    
    df['est_civil_relacion'] = df.apply(lambda row: 1 if row['est_civil'] in (2,5) else 0, axis = 1)
    
    return df


def processing():
    # Rutas
    path_train = '/opt/airflow/dags/data/inputs/train.csv'
    path_test = '/opt/airflow/dags/data/inputs/test.csv'
    path_temp = '/opt/airflow/dags/data/temp'

    # Lectura de archivos
    dfTrain_t = load_data(path_train)
    dfTrain_t['flag_train'] = 1
    dfTest_t = load_data(path_test)
    dfTest_t['flag_train'] = 0
    dfCons = pd.concat([dfTrain_t, dfTest_t], ignore_index=True)
    
    # Transformación de bases
    dfCons = rename_columns(dfCons)
    dfCons = feature_engineering(dfCons)
    
    # Exportación temporal
    dfTrain = dfCons[dfCons['flag_train'] == 1].drop(columns = 'flag_train')
    dfTest = dfCons[dfCons['flag_train'] == 0].drop(columns = ['flag_train','target'])

    dfTrain.to_csv(f'{path_temp}/train_temp.csv', index = False)
    dfTest.to_csv(f'{path_temp}/test_temp.csv', index = False)

