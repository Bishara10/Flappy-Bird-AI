import keras
keras.utils.disable_interactive_logging()

class Brain():
    def __init__(self, hidden_nodes, input_size, output_size, lr):
        self.numInputs = input_size
        self.numOutputs = output_size
        self.learningRate = lr

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(units=hidden_nodes*2, activation='relu', input_shape=(self.numInputs, )))
        self.model.add(keras.layers.Dense(units=hidden_nodes, activation='relu'))
        self.model.add(keras.layers.Dense(units=self.numOutputs))
        self.model.compile(optimizer=keras.optimizers.AdamW(learning_rate=self.learningRate, amsgrad=True), loss='mean_squared_error')

    # save model weights
    def save_weights(self, fname):
        self.model.save_weights(fname)

    #load model weightss
    def load_weights(self, fname):
        self.model.load_weights(fname)

# print(brain.model.summary())