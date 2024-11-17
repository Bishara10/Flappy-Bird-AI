from flappybirdenv.flappybird import FlappyBird
from dqn import Dqn
import numpy as np
import keras
import yaml
from datetime import datetime
keras.utils.disable_interactive_logging()

# load hyperparameters
parameters = None
with open("hyperparameters.yml", "r") as parameters_file:
    parameters = yaml.safe_load(parameters_file)

learningRate = parameters["learningRate"]
maxMemory = parameters["maxMemory"]
gamma = parameters["gamma"]
batchSize = parameters["batchSize"]
epsilon = parameters["epsilon"]
epsilonDecayRate = parameters["epsilonDecayRate"]
epsilonMin = parameters["epsilonMin"]
hidden_nodes = parameters["hidden_nodes"]
ddqn_enable = parameters["ddqn_enable"] 
tau = parameters["tau"]
training = parameters["train"]

# Initialize environment, and the experience replay memory
DQN = Dqn(hidden_nodes=hidden_nodes, lr=learningRate, maxMemory=maxMemory, discount=gamma)
weights_file_name = "dqntrain.weights.h5"
if not training:
    DQN.load_weights(weights_file_name)

# main loop
epoch = 0
currentState = np.zeros((1, 5))
nextState = currentState
totReward = 0


# Create a new log file and log the parameters
log_file = f"./logs/log{datetime.now().strftime('%m-%d--%H-%M')}.txt"
def log_default(epoch, totReward, epsilon, score, mode="+a"):
        with open(log_file, mode) as log:
            log.write(f"{datetime.now()}: epoch: {epoch} | totalReward = {totReward} | epsilon = {epsilon} | pipes passed = {score}\n")

def log_parameters(mode="a"):
    with open("hyperparameters.yml", "r") as parameters_file:
        parameters = yaml.safe_load(parameters_file)

    with open(log_file, mode) as f:
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")
    
        f.write("\n")

log_parameters()


# initialize game environment.
env = FlappyBird()

# Training loop 
while epoch < 50000:
    epoch += 1

    # get current game state:
    env.resetGame()
    currentState = np.zeros((1, 5))
    nextState = np.zeros((1, 5))
    currentState[0] = env.getGameState()
    gotReward = False
    topCollision = False
    pipes_passed = 0

    # Game loop until game is not over
    gameOver = False
    while not gameOver:
        #Taking an action using the epsilon greedy policy
        # if random number is less than epsilon, take a random action, otherwise, 
        # let the model predict an action and take the action with the highest Q-value
        action = None
        if np.random.rand() <= epsilon and training:
            # The bird jumps if action = 1, if action = 0, do nothing. Here, a random number is generated
            # between 0 and 10, even though only 0 and 1 are used. If the number is not 1, then the
            # bird does not jump. This is done to prevent the bird from jumping almost every frame, and helps
            # with the exploration. 
            action = np.random.randint(0, 10)

        else:
            qvalues = DQN.model(currentState)[0]
            action = np.argmax(qvalues)


        # Only 1 and 0 are used. If the value is higher than 1 (as a result of taking a random action), 
        # then the action is set to 0, otherwise, it is set to 1
        action = 0 if action != 1 else 1

        # Take the action and get the game state.
        gameOver, gotReward = env.step(action, epoch)
        nextState[0] = env.getGameState()


        #rewards:
        if gotReward:
            reward_this_round = 2.
            pipes_passed += 1
        elif gameOver:
            reward_this_round = -2.
        else:
            reward_this_round = 0.1

        # Remeber new experience
        if training:
            DQN.remember([np.copy(currentState), action, reward_this_round, np.copy(nextState)], gameOver)

        currentState = np.copy(nextState)
        gotReward = False #reset
        totReward += reward_this_round
        # print(state_params, totReward)

    # Log the current epoch's information
    log_default(epoch, totReward, epsilon, pipes_passed)

    # Train the model on the current state and the expected values of the action taken (Q values). Get these
    # vlaues from the getBatch() function then feed it into the model for training.
    if training:
        inputs, targets = DQN.getBatch(batchSize, True)
        if inputs is not None and targets is not None:
            DQN.model.train_on_batch(inputs, targets)

        # Save the weights whenever the model performs better
        DQN.save_weights(weights_file_name)

        # If it's a DDQN model, update the target network weights using the soft update.
        # Can be updated using the hard update in update_target_dqn() function in dqn.py for experimentation. 
        if (ddqn_enable):
            DQN.soft_update_target_dqn(tau)


        # decrease epsilon and reset the total reward for this epoch
        epsilon = max(epsilon * epsilonDecayRate, epsilonMin)
        totReward = 0
