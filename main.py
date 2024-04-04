from model.separate_data import Separate_Data
from graphs.scatter_plot import scatter_plot
from graphs.histogram import histogram
from graphs.pair_plot import pair_plot
from model.train import layers
from model.train import model
from utils.load_csv import load_csv

def main():
    try:
        # # Separate Data ans clean data
        # separate_data = Separate_Data("datasets/data.csv")
        # separate_data.clean_data() # 0 = Malignant, 1 = Bening
        # separate_data.divide_data(0.80)
        # separate_data.create_files("datasets")

        # df = separate_data.get_df().drop("Diagnosis", axis=1, inplace=False)
    #    #Analyse the data
    #     # scatter_plot(df)
    #     # histogram(df)
    #     # pair_plot(separate_data.get_df())



        #Train data
        # layer = layers()
        # layer.DenseLayer(4, "sigmoid", "init")
        # df = separate_data.get_df()
        test_data = load_csv("datasets/test_data.csv")
        train_data = load_csv("datasets/train_data.csv")
        model_test = model()
        # print(test_data)
        model_test.fit(None, train_data, test_data, loss='binaryCrossentropy', learning_rate=0.0314, batch_size=8, epochs=84)


    except Exception as error:
        print("Error:", error)


if __name__ == "__main__":
    main()
