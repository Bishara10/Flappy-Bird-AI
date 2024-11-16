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


    # Getting batches of inputs and targets
    @tf.function
    def getBatch(self, batchSize, ddqn=False):
        if len(self.memory) < batchSize:
            return None, None

        # Randomly sample batch indices
        indices = np.random.randint(0, len(self.memory), size=batchSize)

        # Extract data and convert to tensors
        inputs = tf.convert_to_tensor([tf.squeeze(self.memory[idx][0][0]) for idx in indices], dtype=tf.float32)
        actions = tf.convert_to_tensor([self.memory[idx][0][1] for idx in indices], dtype=tf.int32)
        rewards = tf.convert_to_tensor([self.memory[idx][0][2] for idx in indices], dtype=tf.float32)
        nextStates = tf.convert_to_tensor([tf.squeeze(self.memory[idx][0][3]) for idx in indices], dtype=tf.float32)
        gameOvers = tf.convert_to_tensor([self.memory[idx][1] for idx in indices], dtype=tf.float32)

        # Reshape inputs and nextStates according to model's input shape
        expected_shape = self.model.input_shape[1:]
        inputs = tf.reshape(inputs, (batchSize,) + expected_shape)
        nextStates = tf.reshape(nextStates, (batchSize,) + expected_shape)

        # Batch predictions for inputs and nextStates
        currentQValues = self.model(inputs, training=False)
        nextQValues = self.model(nextStates, training=False)

        # if it's a DDQN model, use the best action according to the revised Bellman equation
        if ddqn:
            bestActions = tf.argmax(nextQValues, axis=1, output_type=tf.int32)
            nextQTargetValues = self.target_dqn(nextStates, training=False)
            targetQValues = rewards + (1 - gameOvers) * self.discount * tf.gather_nd(
                nextQTargetValues, tf.stack([tf.cast(tf.range(batchSize), dtype=tf.int32), bestActions], axis=1)
            )
        # Otherwise, use the traditional equation
        else:
            targetQValues = rewards + (1 - gameOvers) * self.discount * tf.reduce_max(nextQValues, axis=1)

        # Create targets tensor based on current Q-values and update specific actions
        targets = tf.identity(currentQValues)
        indices = tf.stack([tf.range(batchSize, dtype=tf.int32), actions], axis=1)
        targets = tf.tensor_scatter_nd_update(targets, indices, targetQValues)

        return inputs, targets
    

    # remember new experiences and remove the oldest if the memory is full
    def remember(self, transition, gameOver):
        self.memory.append([transition, gameOver])
        if len(self.memory) > self.maxMemory:
            del self.memory[0]


    # Update target DQM weights to match main DQN (hard update)
    def update_target_dqn(self):
        self.target_dqn.set_weights(self.model.get_weights())


    # Soft update target DQN weights
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