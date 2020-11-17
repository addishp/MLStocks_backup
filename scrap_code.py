
################################## If we want activation function to be switchable
# if self.optimizer == 'adabound':
#     regressior.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(self.data.shape[0],)))
# else:
#
# # additional layers
# for i in range(1, layers - 1):
#     if self.optimizer == 'adabound':
#
#         if layer ==
#
#         model.add(LSTM(units=units_for_layers[i],
#                        activation=AdaBound(self.low_bound, self.high_bound),
#                        return_sequences=True))
#         model.add(Dropout(dropouts_for_layers[i]))