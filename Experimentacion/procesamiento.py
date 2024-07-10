import pandas as pd
import json

def load_data(path):

    return pd.read_csv(path)

def rename_columns(df):
    
    # Leer nombres de columnas
    ruta_nombres = 'dict_nombres.json'#'/opt/airflow/dags/data/inputs/dict_nombres.json'
    with open(ruta_nombres) as file:
        dict_nombres = json.load(file)

    # Cambiar nombres de columna
    return df.rename(columns = dict_nombres)


def feature_engineering(df):
    
    # Leer la segmentacion de columnas
    ruta_columnas = 'seg_columnas.json'#'/opt/airflow/dags/data/inputs/seg_columnas.json'
    with open(ruta_columnas) as file:
        dict_columnas = json.load(file)

    # Crear dummies
    for col in dict_columnas["col_categoricas"]:
        dfDummies = pd.get_dummies(df[col], prefix = col).astype(int)
        df = pd.concat([df, dfDummies], axis = 1)
    
    # Crear variable
    df['est_civil_relacion'] = df.apply(lambda row: 1 if row['est_civil'] in (2,5) else 0, axis = 1)

    return df