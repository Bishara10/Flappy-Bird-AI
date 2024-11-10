import numpy as np
import tensorflow as tf
from brain import Brain
import keras
keras.utils.disable_interactive_logging()

class Dqn():
    def __init__(self, hidden_nodes, lr, maxMemory, discount):
        self.maxMemory = maxMemory
        self.discount = discount
        self.memory = list()

        self.model = Brain(hidden_nodes, 5, 2, lr).model

        self.target_dqn = Brain(hidden_nodes, 5, 2, lr).model
        self.update_target_dqn()


    # Remembering new experience
    def remember(self, transition, gameOver):
        self.memory.append([transition, gameOver])
        if len(self.memory) > self.maxMemory:
            del self.memory[0]


    # @tf.function
    # Getting batches of inputs and targets
    def getBatch(self, batchSize, ddqn=False):
        lenMemory = len(self.memory)
        numInputs = self.memory[0][0][0].shape[1]
        numOutputs = self.model.output_shape[-1]

        # Initializing the inputs and targets
        inputs = np.zeros((batchSize, numInputs))
        targets = np.zeros((batchSize, numOutputs))

        # Extracting transitions from random experiences
        for i, inx in enumerate(np.random.randint(0, lenMemory, size=batchSize)):
            currentState, action, reward, nextState = self.memory[inx][0]
            gameOver = self.memory[inx][1]

            # Updating inputs and targets
            inputs[i] = currentState
            targets[i] = self.model.predict(currentState)[0]

            if ddqn:
                best_action = np.argmax(self.model.predict(nextState)[0])
                if gameOver:
                    targets[i][action] = reward

                else:
                    targets[i][action] = (reward + self.discount * self.target_dqn.predict(nextState)[0][best_action])
            else:
                if gameOver:
                    targets[i][action] = reward

                else:
                    targets[i][action] = reward + self.discount * np.max(self.model.predict(nextState)[0])

        return inputs, targets

    def remember(self, transition, gameOver):
        self.memory.append([transition, gameOver])
        if len(self.memory) > self.maxMemory:
            del self.memory[0]

    def update_target_dqn(self):
        self.target_dqn.set_weights(self.model.get_weights())


    def soft_update_target_dqn(self, tau: float):
        main_network_weights = self.model.get_weights()
        target_network_weights = self.target_dqn.get_weights()

        # Apply the soft update formula
        new_weights = []
        for target_weight, main_weight in zip(target_network_weights, main_network_weights):
            updated_weight = tau * main_weight + (1 - tau) * target_weight
            new_weights.append(updated_weight)

        # Set the new weights to the target model
        self.target_dqn.set_weights(new_weights)


    def save_weights(self, fname):
        self.model.save_weights(fname)


    def load_weights(self, fname):
        self.model.load_weights(fname)