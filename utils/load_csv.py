import pandas as pd

def load_csv(path: str):
    df = pd.read_csv(path, index_col=0)
    return df

def load_csv_columns(path: str, columns):
    df = pd.read_csv(path, names=columns)
    return df
