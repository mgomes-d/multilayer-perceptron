from model.separate_data import Separate_Data
from graphs.scatter_plot import scatter_plot
from graphs.histogram import histogram
from graphs.pair_plot import pair_plot
from model.train import layers

def main():
    try:
    #     # # Separate Data ans clean data
    #     separate_data = Separate_Data("datasets/data.csv")
    #     separate_data.clean_data() # 0 = Malignant, 1 = Bening
    #     # separate_data.divide_data(0.80)
    #     # separate_data.create_files("datasets")
    #     # df = separate_data.get_df().drop("Diagnosis", axis=1, inplace=False)
       
    #    #Analyse the data
    #     # scatter_plot(df)
    #     # histogram(df)
    #     # pair_plot(separate_data.get_df())



        #Train data
        layer = layers()
        layer.DenseLayer(4, "sigmoid", "init")


    except Exception as error:
        print("Error:", error)


if __name__ == "__main__":
    main()
