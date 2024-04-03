import numpy as np

class perceptron:
    def __init__(self, nb, init_weight, activation):
        self.id = nb
        self.weight = init_weight
        self.activation = activation
    def hello(self):
        print("HEllo")

class layers:
    def __init__(self):
        self.layers = []

    def DenseLayer(self, nb_perceptron, activation, weights_initializer):
        perceptrons = []
        for i in range(nb_perceptron):
            perceptrons.append(perceptron(i, self.init_weight(), activation))
        self.layers.append(perceptrons)
        print(self.layers)
        
    def init_weight(self):
        return 0

    def sigmoid(self, x):
        return (1 / 1 + np.exp(-x))


class model:
    def __init__(self, df):

        self.parse_data(df)

        self.network = []
        self.data_train = {}
        self.data_valid = {}
        self.loss = ""
        self.learning_rate = 0
        self.batch_size = 0
        self.epochs = 0

    def parse_data(self, data):
        data_predict = data.select_dtypes(include=[object])
        first_value = ""
        second_value = ""
        assert len(data_predict.columns) == 1, "Need only 1 object type to predict"
        for (key, value) in data_predict.items():
            print(value)


    def make_mean_std(self, data):
        self.mean = {}
        self.std = {}
        for column_name, content in data.select_dtypes(include=[float]).items():
            self.mean[column_name] = content.values.sum() / len(content.values)
            self.std[column_name] = (content.map(lambda x: (x - self.mean[column_name])**2).sum() / len(content.values))**0.5

    def normalize_data(self, data):
        for column_name, content in data.select_dtypes(include=[float]).items():
            data[column_name] = content.map(lambda x: (x - self.mean[column_name]) / self.std[column_name])

    def init_values(self, network, data_train, data_valid, loss, learning_rate, batch_size, epochs):
        self.network = network
        self.data_train = data_train
        self.data_valid = data_valid
        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs


    def fit(self, network, data_train, data_valid, loss, learning_rate, batch_size, epochs):
        self.init_values(network, data_train, data_valid, loss, learning_rate, batch_size, epochs)
        self.make_mean_std(data_train)
        self.normalize_data(data_train)
        self.normalize_data(data_valid)
        self.parse_data(self.data_train)
        self.parse_data(self.data_valid)


