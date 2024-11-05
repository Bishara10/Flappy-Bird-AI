import numpy as np
from brain import Brain

class Dqn():
    def __init__(self, maxMemory, discount, model):
        self.maxMemory = maxMemory
        self.discount = discount
        self.memory = list()

        self.model = model
        
        self.target_dqn = Brain(4, 2, 0.001)
        self.update_target_dqn()

    #Remembering new experience
    def remember (self, transition, gameOver): 
        self.memory.append([transition, gameOver]) 
        if len(self.memory) > self.maxMemory: 
            del self.memory[0]

    #Getting batches of inputs and targets
    def getBatch(self, batchSize, ddqn = False): 
        lenMemory = len(self.memory) 
        numInputs = self.memory[0][0][0].shape[1] 
        numOutputs = self.model.output_shape[-1]

        #Initializing the inputs and targets
        inputs = np.zeros((min(batchSize, lenMemory), numInputs))
        targets = np.zeros((min(batchSize, lenMemory), numOutputs))

        #Extracting transitions from random experiences
        for i, inx in enumerate (np.random.randint(0, lenMemory, size = min(batchSize, lenMemory))):
            currentState, action, reward, nextState = self.memory[inx][0] 
            gameOver = self.memory[inx] [1]

            # Updating inputs and targets
            inputs[i] = currentState
            targets[i] = self.model.predict(currentState)[0]

            if ddqn:
                best_action = np.argmax(self.model.predict(nextState)[0])
                if gameOver:
                    targets[i][action] = reward

                else:
                    targets[i][action] = reward + self.discount * self.target_dqn.model.predict(nextState)[0][best_action]
            else:
                if gameOver:
                    targets[i][action] = reward

                else:
                    targets[i][action] = reward + self.discount * np.max(self.model.predict(nextState)[0])

        return inputs, targets
    
    def update_target_dqn(self):
        self.target_dqn.model.set_weights(self.model.get_weights())
