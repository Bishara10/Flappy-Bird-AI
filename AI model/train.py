from dqn import Dqn
from brain import Brain
import numpy as np
import matplotlib.pyplot as plt

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
