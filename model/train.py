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
    def __init__(self):
        self.network = []
        self.data_train = {}
        self.data_valid = {}
        self.loss = ""
        self.learning_rate = 0
        self.batch_size = 0
        self.epochs = 0
        
        self.mean = {}
        self.std = {}

        self.first_predict = "" # replace to 0
        self.second_predict = "" # replace to 1
        self.Y_train = {}
        self.Y_valid = {}
        self.X_train = {}
        self.X_valid = {}

    def get_predict_values(self, data):
        data_predict = data.select_dtypes(include=[object])
        assert len(data_predict.columns) == 1, "Need only 1 object type to predict, get_predict_values"
        for (key, value) in data_predict.items():
            for content in value.values:
                if not self.first_predict:
                    self.first_predict = content
                if not self.second_predict and content != self.first_predict:
                    self.second_predict = content
                if self.first_predict and self.second_predict \
                and content != self.first_predict and content != self.second_predict:
                    raise Exception("More than 2 values to predict")

    def parse_data(self, data, train_data=False):
        assert self.Y_train is not True and self.X_train is not True \
        and self.Y_valid is not True and self.X_valid is not True, \
        "X or Y _train and X or Y _valid need to be empty"
        if train_data is True:
            self.Y_train = data.select_dtypes(include=[object])
            assert len(self.Y_train.columns) == 1, "Need only 1 object type to predict, parse_data: train_data"
            self.X_train = data.select_dtypes(exclude=[object])
            self.Y_train.replace([self.first_predict, self.second_predict], [0, 1], inplace=True)
        else:
            self.Y_valid = data.select_dtypes(include=[object])
            assert len(self.Y_valid.columns) == 1, "Need only 1 object type to predict, parse_data: valid_data"
            self.X_valid = data.select_dtypes(exclude=[object])
            self.Y_valid.replace([self.first_predict, self.second_predict], [0, 1], inplace=True)


    def make_mean_std(self, data):
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
        self.normalize_data(self.data_train)
        self.normalize_data(self.data_valid)
        self.get_predict_values(self.data_train)
        self.parse_data(self.data_train, True)
        self.parse_data(self.data_valid)