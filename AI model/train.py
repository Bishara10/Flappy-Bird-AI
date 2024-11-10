from olddqn import Dqn
from brain import Brain
import numpy as np
import game_training_env

# Parameters
learningRate = 0.001
maxMemory = 5000
gamma = 0.9
batchSize = 32
epsilon = 1.
epsilonDecayRate = 0.995


# Initialize environment, the brain and the experience replay memory
# 3 inputs: 
#     1. horizontal distance from bird to top pipe
#     2. vertical distance from the top of the bird to the top pipe
#     3. vertical distance from bottom of the bird to the bottom pipe 
brain = Brain(3, 1, learningRate)
model = brain.model
DQN = Dqn(maxMemory, gamma)

# main loop
epoch = 0
currentState = np.zeros((1, 2))
nextState = currentState
totReward = 0
rewards = list()

while True:
    epoch += 1

    #Starting to play the game
    env.reset()
    currentState = np.zeros((1, 3))
    nextState = currentState
    gameOver = False
    while not gameOver:
        #Taking an action
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 3)

        else:
            qvalues = model.predict(currentState)[0] 
            action = np.argmax(qvalues)

        #Updating the Environment

        nextState[0], reward, gameOver, = env.step(action)
        env.render()

        totReward += reward

        #Remembering new experience, training the AI ar
        DQN.remember([currentState, action, reward, nextState], gameOver)
        inputs, targets = DQN.getBatch(model, batchSize)
        model.train_on_batch(inputs, targets)

        currentState = nextState

    #Lowering epsilon and displaying the results
    epsilon *= epsilonDecayRate
    print('Epoch: + str(epoch) + Epsilon: {:.5f}'.format(epsilon) + ' Total Reward: {:.2f}'.format(totReward))
    rewards.append(totReward)
    totReward = 0