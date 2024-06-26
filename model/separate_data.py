from utils.load_csv import load_csv_columns
from utils.create_csv import create_csv


class Separate_Data:
    def __init__(self, path):
        columns = self.create_columns()
        self.df = load_csv_columns(path, columns)

    def create_columns(self):
        columns = ["Id", "Diagnosis"]
        for i in range(1, 31):
            columns.append(f'feature{i}')
        return columns

    def divide_data(self, train_data):
        train_value = int(len(self.df) * train_data)
        self.train_df = self.df.loc[:train_value - 1].copy()
        self.test_df = self.df.loc[train_value:].copy()

    def create_files(self, path):
        create_csv(f'{path}/train_data.csv', self.train_df)
        create_csv(f'{path}/test_data.csv', self.test_df)

    def clean_data(self):
        del(self.df["Id"])
        # self.df["Diagnosis"].replace(['M', 'B'], [0, 1], inplace=True)

    def get_df(self):
        return self.df
