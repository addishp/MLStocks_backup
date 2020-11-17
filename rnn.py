import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from keras_adabound import AdaBound
from matplotlib import pyplot as plt

upscale_value = 0

def scale_data(df):
    # scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # save this value to convert stock prediction to nominal data
    upscale_value = scaler.scale_[0]
    return scaled_data, upscale_value


def reverse_order(df):
    # reverse order of data so earliest day is day 0
    reversed_df = df[::-1].reset_index(drop=True)
    return reversed_df


def to_dataframe(csv):
    # returns dataframe
    df = pd.read_csv(csv, date_parser=True)
    return df

def split_data():


class Rnn:
    # set values for Rnn object
    def __init__(self, optimizer, low_bound = None, high_bound = None):
        self.low_bound = 0
        self.high_bound = 0
        if optimizer == 'adabound':
            self.low_bound = low_bound
            self.high_bound = high_bound
        self.model = None


    # train function for Rnn class
    def create(self, rows, columns, layers, units_for_layers, dropouts_for_layers):
        #initilize Sequential rnn
        self.model = Sequential()
        # add first layer and define input shape
        self.model.add(LSTM(units_for_layers[0], activation = 'relu', return_sequences = True, input_shape = (rows, columns)))
        self.model.add(Dropout(dropouts_for_layers[0]))
        # for adding additional layers
        if layers > 2:
            for i in range(1,layers-1):
                self.model.add(LSTM(units_for_layers[i], activation = 'relu', return_sequences = True))

        #the penultimate layer is different (doesn't return sequences)
        self.model.add(LSTM(units_for_layers[])
        #final output is given


        self.model = model

        return None

    def train(self, xTrain, yTrain, epochs, batch_size, optimizer, ada_low = None, ada_high = None):
        #compiles model that was created
        if optimizer != 'adaboost':
            self.model.compile(optimizer=optimizer, loss = 'mean_squared_error')
        else:
            self.model.compile(optimizer=AdaBound(lr=ada_low, final_lr=ada_high), loss = 'mean_squared_error')
        #fit model to data
        self.model.fit(xTrain, yTrain, epochs=epochs, batch_size=batch_size)

    def predict(self, input_data):
        y_hat = self.model.predict(input_data)
        return y_hat


)






    def predict(self, ):








def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",
                        help="filename for dataset")
    parser.add_argument("target_stock_indicator",
                        help="stock indicator that will be used to find label column: 'stock_open'")
    csv = parser[0]
    target_stock = parser[1]

    data = scale_data(to_dataframe(csv))
    scale


if __name__ == "__main__":
    main()
