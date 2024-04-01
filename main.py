from model.separate_data import Separate_Data

def main():
    try:
        # Separate Data
        separate_data = Separate_Data("datasets/data.csv")
        separate_data.clean_data()
        separate_data.divide_data(0.80)
        separate_data.create_files("datasets")

        #Train data
        

    except Exception as error:
        print("Error:", error)

    print("main")

if __name__ == "__main__":
    main()