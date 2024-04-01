import pandas as pd

def create_csv(file_name, df):
    df.to_csv(file_name)