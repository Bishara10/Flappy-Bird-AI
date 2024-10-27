import keras

class Brain():
    def __init__(self, input_size, output_size, lr):
        self.numInputs = input_size
        self.numOutputs = output_size
        self.learningRate = lr

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(units=24, activation='relu', input_shape=(self.numInputs, )))
        self.model.add(keras.layers.Dense(units=8, activation='relu'))
        self.model.add(keras.layers.Dense(units=self.numOutputs))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learningRate), loss='mean_squared_error')

brain = Brain(2, 3, 0.01)
print(brain.model.summary())