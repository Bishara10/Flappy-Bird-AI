from components.AllComponents import *
from components.misc import *
from dqn import Dqn
import numpy as np
import keras
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


# Initialize environment, and the experience replay memory
DQN = Dqn(hidden_nodes=hidden_nodes, lr=learningRate, maxMemory=maxMemory, discount=gamma)
weights_file_name = "dqntrain.weights.h5"
maxReward = -99999

# main loop
epoch = 0
currentState = np.zeros((1, 5))
nextState = currentState
totReward = 0

# Create a new log file and log the parameters
log = Log()
log.log_parameters()

screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')
pygame.mixer.set_num_channels(10)

BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))

def prepare_game_scenario():
    bird_group = pygame.sprite.Group()
    bird = Bird()
    bird_group.add(bird)


    ground_group = pygame.sprite.Group()
    for i in range (2):
        ground = Ground(GROUND_WIDHT * i)
        ground_group.add(ground) 


    pipe_group = pygame.sprite.Group()
    reward_group = pygame.sprite.Group()
    for i in range (3):
        pos =  250 * i + 400
        pipes = get_random_pipes(pos)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

        # print(pipe_group)
    
        reward = Reward(pos + PIPE_WIDHT/2) # was xpos + PIPE_WIDTH/2 might consider deleting it and get it from get_random_pipes
        reward_group.add(reward)

    top_boundary = TopBoundary()

    score = 0

    return bird_group, bird, ground_group, pipe_group, reward_group, top_boundary, score


def getGameState(pipe_group: pygame.sprite.Group, bird_obj: Bird):
    state_params = []

    chosenpipeindex = 0
    nextpipe_x = pipe_group.sprites()[chosenpipeindex].rect[0] + PIPE_WIDHT
    if (nextpipe_x < bird_obj.rect[0]):
        chosenpipeindex = 2  # if the next pipe is to the left of the bird, choose the right pipe

    # nextpipe_x = pipe_group.sprites()[chosenpipeindex].rect[0]
    # nextpipe_bottom_y = pipe_group.sprites()[chosenpipeindex].rect[1]
    # nextpipe_top_y = nextpipe_bottom_y - PIPE_GAP

    nextpipe_x = pipe_group.sprites()[chosenpipeindex].rect[0] - bird_obj.rect[0]
    nextpipe_bottom_y = pipe_group.sprites()[chosenpipeindex].rect[1]
    nextpipe_top_y = nextpipe_bottom_y - PIPE_GAP

    d_bird_nextpipe_bottom_y = nextpipe_bottom_y - bird_obj.rect[1]
    d_bird_nextpipe_top_y = bird_obj.rect[1] - nextpipe_top_y

    pipe_middle_y = nextpipe_bottom_y - (PIPE_GAP / 2)

    # state #14 and #5: bird's vertical position and bird speed
    bird_y = bird_obj.rect[1]
    bird_speed = bird_obj.speed

    # state_params.append(nextpipe_x / SCREEN_WIDHT)
    state_params.append(nextpipe_top_y)
    state_params.append(nextpipe_bottom_y)
    state_params.append(pipe_middle_y)
    state_params.append(bird_y)
    state_params.append(bird_speed)

    return state_params

bird_group, bird, ground_group, pipe_group, reward_group, top_boundary, SCORE = prepare_game_scenario()
clock = pygame.time.Clock()

bird.begin()
score = Score()


# Training loop 
while True:
    epoch += 1
    currentState = np.zeros((1, 5))
    # get current game state:
    nextState = np.zeros((1, 5))
    currentState[0] = getGameState(pipe_group, bird)
    gotReward = False
    topCollision = False

    # Game loop until game is not over
    gameOver = False
    while not gameOver:
        clock.tick(30) # was 15 -------------------------------------------------------------------------------
        action = None

        display_score = score.update(new_val=SCORE)
        display_epoch = font2.render(f"epoch: {epoch}", True, (255, 255, 255))

        #Taking an action
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 10)  # 0 and 2: do nothing, 1: jump. This is done in order to increase the odds of doing nothing
            # print(action)

        else:
            qvalues = DQN.model(currentState)[0]
            action = np.argmax(qvalues)
            # print(f"action: {action}, qvals = {qvalues}")

        if action == 1:
            bird.bump()
        elif action > 1:
            # do nothing
            action = 0 # still does nothing but it's necessary to assign it to 0 as the next code blocks rely on 1 or 0
        elif action == None:
            print("no action")


        for event in pygame.event.get():
            if event.type == QUIT:
                DQN.save_weights(fname=f"quit{weights_file_name}")
                pygame.quit()

            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    DQN.model.save_weights(weights_file_name)


        screen.blit(BACKGROUND, (0, 0))

        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])

            new_ground = Ground(GROUND_WIDHT - 20)
            ground_group.add(new_ground)

        if is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])

            pipes = get_random_pipes(SCREEN_WIDHT +  250)

            pipe_group.add(pipes[0])
            pipe_group.add(pipes[1])

            reward_group.add(pipes[2])

        bird_group.update()
        ground_group.update()
        pipe_group.update()
        reward_group.update()


        bird_group.draw(screen)
        pipe_group.draw(screen)
        ground_group.draw(screen)
        reward_group.draw(screen)

        screen.blit(display_score, score.pos)
        screen.blit(display_epoch, (SCREEN_WIDHT // 18 - font.get_height() // 2, SCREEN_HEIGHT//20))

        screen.blit(top_boundary.surf, (0, -4))


        if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.collide_mask(bird, top_boundary)):
            pygame.mixer.find_channel().play(hit_sound)
            gameOver = True


        if (pygame.sprite.groupcollide(bird_group, reward_group, False, False, pygame.sprite.collide_mask)):
            # reward_group.remove(reward_group.sprites()[0])
            pygame.mixer.find_channel().play(point_sound)
            SCORE += 1
            gotReward = True

        pygame.display.update()

        # compile state parameters in nextState
        # nextState[0] = np.array(state_params)
        nextState[0] = getGameState(pipe_group, bird)


        #rewards:
        if gotReward:
            reward_this_round = 2.
            reward_group.remove(reward_group.sprites()[0])
        elif gameOver:
            reward_this_round = -2.
        # elif topCollision:
            # reward_this_round = -1
        else:
            reward_this_round = 0.1

        # Remeber new experience
        DQN.remember([np.copy(currentState), action, reward_this_round, np.copy(nextState)], gameOver)

        currentState = np.copy(nextState)
        gotReward = False #reset
        totReward += reward_this_round
        # print(state_params, totReward)

    inputs, targets = DQN.getBatch(batchSize, True)
    if inputs is not None and targets is not None:
        DQN.model.train_on_batch(inputs, targets)

    if (totReward > maxReward):
            maxReward = totReward
            DQN.save_weights(weights_file_name)

    if (ddqn_enable):
        DQN.soft_update_target_dqn(tau)
        # DQN.update_target_dqn()

    log.log_default(epoch, totReward, epsilon, SCORE)

    # restart game scenario
    pipe_group.empty()
    reward_group.empty()
    bird.begin()

    pipe_group = pygame.sprite.Group()
    reward_group = pygame.sprite.Group()
    for i in range (3):
        pos = 240 * i + 400
        pipes = get_random_pipes(pos)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

        reward = Reward(pos + PIPE_WIDHT/2) # was xpos + PIPE_WIDTH/2
        reward_group.add(reward)

    epsilon = max(epsilon * epsilonDecayRate, epsilonMin)
    SCORE = 0
    totReward = 0
    print(len(DQN.memory))