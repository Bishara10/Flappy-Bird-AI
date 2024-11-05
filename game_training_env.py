import pygame, time
from components.AllComponents import *
from pygame.locals import *
from dqn import Dqn
from brain import Brain
import numpy as np
from datetime import datetime

# Parameters
learningRate = 0.0001
maxMemory = 100000
gamma = 0.95
batchSize = 30
epsilon = 1.
epsilonDecayRate = 0.995
epsilonMin = 0.05
train_on_frames = 32  # train model every 32 frames
action_flag = 5 # take action every 5 frames
ddqn_enable = True

# Initialize environment, the brain and the experience replay memory
# 3 inputs: 
#     1. horizontal distance from bird to top pipe
#     2. vertical distance from the top of the bird to the top pipe
#     3. vertical distance from bottom of the bird to the bottom pipe 
brain = Brain(4, 2, learningRate)
model = brain.model
DQN = Dqn(maxMemory, gamma, model)

# main loop
epoch = 0
currentState = np.zeros((1, 4))
nextState = currentState
totReward = 0
rewards_list = list()
log_file = f"./logs/log{datetime.now().strftime('%m-%d--%H-%M')}.txt"

with open(log_file, "+a") as log:
    log.write(f"Hyperparameters:\nlearningRate = {learningRate}\nmaxMemory = {maxMemory}\ngamma = {gamma}\nbatchSize = {batchSize}\nepsilon = {epsilon}\nepsilonDecayRate = {epsilonDecayRate}\nepsilonMin = {epsilonMin}\nddqn_enable = {ddqn_enable}\n\n")


screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')

BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))

# nextstate parameters:
d_reward_bird = 0
d_topPipe_bird = 0
d_bottomPipe_bird = 0

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
    
        reward = Reward(pos + 20) # was xpos + PIPE_WIDTH/2 might consider deleting it and get it from get_random_pipes
        reward_group.add(reward)

    top_boundary = TopBoundary()

    score = 0

    return bird_group, bird, ground_group, pipe_group, reward_group, top_boundary, score


bird_group, bird, ground_group, pipe_group, reward_group, top_boundary, SCORE = prepare_game_scenario()
clock = pygame.time.Clock() 

bird.begin()


# Training loop 
while True:
    epoch += 1
    currentState = np.zeros((1, 4))
    nextState = currentState
    gotReward = False
    topCollision = False

    # Game loop until game is not over
    gameOver = False
    while not gameOver:
        clock.tick(30) # was 15 -------------------------------------------------------------------------------
        train_on_frames -= 1
        action_flag -= 1
        action = -1

        score = Score()
        display_score = score.update(new_val=SCORE)

        display_epoch = font2.render(f"epoch: {epoch}", True, (255, 255, 255))

        #Taking an action
        if (action_flag == 0):
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 2)
                # print(action)

            else:
                qvalues = DQN.model.predict(currentState)[0] 
                action = np.argmax(qvalues)
                print(f"action: {action}, qvals = {qvalues}")

            if action == 1:
                bird.bump()
            elif action == -1:
                print("wtf")

            action_flag = 5

        for event in pygame.event.get():
            if event.type == QUIT:
                brain.save_weights()
                pygame.quit()
            
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    bird.bump()


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

        screen.blit(top_boundary.surf, (0, 0))


        if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
            pygame.mixer.find_channel().play(hit_sound)

            # # restart game scenario
            # pipe_group.empty()
            # reward_group.empty()
            # bird.begin()

            # pipe_group = pygame.sprite.Group()
            # reward_group = pygame.sprite.Group()
            # for i in range (2):
            #     pos = 240 * i + 400
            #     pipes = get_random_pipes(pos)
            #     pipe_group.add(pipes[0])
            #     pipe_group.add(pipes[1])

            #     reward = Reward(pos + PIPE_WIDHT) # was xpos + PIPE_WIDTH/2
            #     reward_group.add(reward)
                
            # SCORE = 0
            gameOver = True


        if (pygame.sprite.groupcollide(bird_group, reward_group, False, False, pygame.sprite.collide_mask)):
            # reward_group.remove(reward_group.sprites()[0])
            pygame.mixer.find_channel().play(point_sound)
            SCORE += 1
            gotReward = True

        if (pygame.sprite.collide_mask(bird, top_boundary)):
            topCollision = True

        pygame.display.update()


        # Get state parameters, the way they are computed relies on the fact that pipe_group always has 3 pipes
        # and the game always displays 3 pipes at a time.
        # Get state parameter #1, #2 and #3: Get next pipe horizontal position, top and bottom vertical position
        state_params = []

        chosenpipeindex = 0
        nextpipe_x = pipe_group.sprites()[chosenpipeindex].rect[0] + PIPE_WIDHT
        if (nextpipe_x < bird.rect[0]):
            chosenpipeindex = 2 # if the next pipe is to the left of the bird, choose the right pipe

        # nextpipe_x = pipe_group.sprites()[chosenpipeindex].rect[0]
        # nextpipe_bottom_y = pipe_group.sprites()[chosenpipeindex].rect[1]
        # nextpipe_top_y = nextpipe_bottom_y - PIPE_GAP

        nextpipe_x = pipe_group.sprites()[chosenpipeindex].rect[0] - bird.rect[0]
        nextpipe_bottom_y = pipe_group.sprites()[chosenpipeindex].rect[1]
        nextpipe_top_y = nextpipe_bottom_y - PIPE_GAP

        d_bird_nextpipe_bottom_y = nextpipe_bottom_y - bird.rect[1]
        d_bird_nextpipe_top_y = bird.rect[1] - nextpipe_top_y

        # state #14 and #5: bird's vertical position and bird speed
        bird_y = bird.rect[1]
        bird_speed = bird.speed

        state_params.append(nextpipe_x )
        state_params.append(d_bird_nextpipe_top_y )
        state_params.append(d_bird_nextpipe_bottom_y )
        # state_params.append(bird_y / SCREEN_HEIGHT)
        state_params.append(bird_speed)

        # compile state parameters in nextState
        nextState[0] = np.array(state_params)
        

        tst = [nextpipe_x, d_bird_nextpipe_top_y, d_bird_nextpipe_bottom_y, bird_speed]

        #rewards:
        if gotReward:
            reward_this_round = 12
            reward_group.remove(reward_group.sprites()[0])
        elif gameOver:
            reward_this_round = -10
        elif topCollision: 
            reward_this_round = -5
        else:
            reward_this_round = 0.1

        # Remeber new experience
        DQN.remember([currentState, action, reward_this_round, nextState], gameOver)

        if train_on_frames == 0:
            inputs, targets = DQN.getBatch(batchSize, ddqn=ddqn_enable)
            DQN.model.train_on_batch(inputs, targets)
            train_on_frames = 32

        currentState = nextState
        gotReward = False #reset 
        totReward += reward_this_round
        print(tst, totReward)

    # inputs, targets = DQN.getBatch(model, batchSize)
    # model.train_on_batch(inputs, targets)
    # train_on_frames = 20

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

        reward = Reward(pos + 20) # was xpos + PIPE_WIDTH/2
        reward_group.add(reward)
        
    SCORE = 0
    action_flag = 5

    brain.save_weights()
    epsilon = max(epsilon * epsilonDecayRate, epsilonMin)
    rewards_list.append(totReward)
    with open(log_file, "a") as log:
        log.write(f"{datetime.now()}: epoch: {epoch} | totalReward = {totReward} | epsilon = {epsilon}\n")
    totReward = 0


