import keras

class Brain():
    def __init__(self, input_size, output_size, lr):
        self._weights_file_name = "dqntrain.weights.h5"
        self.numInputs = input_size
        self.numOutputs = output_size
        self.learningRate = lr

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(units=512, activation='relu', input_shape=(self.numInputs, )))
        # self.model.add(keras.layers.Dense(units=16, activation='relu'))
        self.model.add(keras.layers.Dense(units=self.numOutputs))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learningRate), loss='mean_squared_error')

    # save model weights
    def save_weights(self):
        self.model.save_weights(self._weights_file_name)

    #load model weights
    def load_weights(self):
        self.model.load_weights(self._weights_file_name)

# print(brain.model.summary())