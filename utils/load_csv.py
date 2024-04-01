import pandas as pd

def load_csv(path: str, columns):
    df = pd.read_csv(path, names=columns)
    # df.index.name = "Index"
    return df