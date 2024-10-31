import pygame, time
from components.AllComponents import *
from pygame.locals import *
from dqn import Dqn
from brain import Brain
import numpy as np

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
brain = Brain(3, 2, learningRate)
model = brain.model
DQN = Dqn(maxMemory, gamma)

# main loop
epoch = 0
currentState = np.zeros((1, 2))
nextState = currentState
totReward = 0
rewards_list = list()

# nextstate parameters:
d_reward_bird = 0
d_topPipe_bird = 0
d_bottomPipe_bird = 0

screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')

BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()


bird_group = pygame.sprite.Group()
bird = Bird()
bird_group.add(bird)


ground_group = pygame.sprite.Group()
for i in range (2):
    ground = Ground(GROUND_WIDHT * i)
    ground_group.add(ground)


pipe_group = pygame.sprite.Group()
reward_group = pygame.sprite.Group()
for i in range (2):
    pos = 250 * i + 400
    pipes = get_random_pipes(pos)
    pipe_group.add(pipes[0])
    pipe_group.add(pipes[1])

    # print(pipe_group)
 
    reward = Reward(pos + PIPE_WIDHT) # was xpos + PIPE_WIDTH/2
    reward_group.add(reward)



clock = pygame.time.Clock() 

bird.begin()

# Training loop 
while True:
    epoch += 1
    currentState = np.zeros((1, 3))
    nextState = currentState
    gotReward = 0

    # Game loop until game is not over
    gameOver = False
    while not gameOver:
        clock.tick(2)

        score = Score()
        display_score = score.update(new_val=SCORE)

        display_epoch = font2.render(f"epoch: {epoch}", True, (255, 255, 255))

        #Taking an action
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 2)
            # print(action)

        else:
            qvalues = model.predict(currentState)[0] 
            action = np.argmax(qvalues)


        for event in pygame.event.get():
            if event.type == QUIT:
                brain.save_weights()
                pygame.quit()

            # if event.type == KEYDOWN:
            #     if event.key == K_SPACE or event.key == K_UP:
            #         bird.bump()
            #         pygame.mixer.find_channel().play(wing_sound)

        #take action
        print(f"action: {action}")
        if action == 1:
            bird.bump()


        screen.blit(BACKGROUND, (0, 0))

        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])

            new_ground = Ground(GROUND_WIDHT - 20)
            ground_group.add(new_ground)
        
        if is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])

            pipes = get_random_pipes(SCREEN_WIDHT * 2)

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


        if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
            pygame.mixer.find_channel().play(hit_sound)
            time.sleep(0.5)

            # restart game scenario
            pipe_group.empty()
            reward_group.empty()
            bird.begin()

            pipe_group = pygame.sprite.Group()
            reward_group = pygame.sprite.Group()
            for i in range (2):
                pos = 240 * i + 400
                pipes = get_random_pipes(pos)
                pipe_group.add(pipes[0])
                pipe_group.add(pipes[1])

                reward = Reward(pos + PIPE_WIDHT) # was xpos + PIPE_WIDTH/2
                reward_group.add(reward)
                
            SCORE = 0
            gameOver = True


        if (pygame.sprite.groupcollide(bird_group, reward_group, False, False, pygame.sprite.collide_mask)):
            reward_group.remove(reward_group.sprites()[0])
            pygame.mixer.find_channel().play(point_sound)
            SCORE += 1
            gotReward = 1

        pygame.display.update()

        # Get state parameter #1: Find distance from the nearest pipes that haven't been crossed.
        # though i'm trying to find the distance from the nearest pipes, I will find the distance between the bird and the nearest reward
        # which is "attached" to the pipes
        d_reward_bird = reward_group.sprites()[0].rect[0] - bird.rect[0]
        
        # Get state parameter #2 and #3: distance between top pipe from bird, and the bottom pipe from bird
        # first find the nearest pipes that haven't been crossed. 
        for i in range(len(pipe_group.sprites())):
            if pipe_group.sprites()[i].rect[0] - bird.rect[0] < 0:
                # print(pipe_group.sprites()[i].rect[0])
                continue

            d_bottomPipe_bird = pipe_group.sprites()[i].rect[1] - bird.rect[1]
            d_topPipe_bird = pipe_group.sprites()[i+1].rect[1] - bird.rect[1] 
            break
    
        nextState[0] = np.array([d_reward_bird, d_topPipe_bird, d_bottomPipe_bird])
        print(nextState[0])

        #rewards:
        if gotReward:
            reward_this_round = 1
        elif gameOver:
            reward_this_round = -1
        # elif touched_top_screen: 
            # reward_this_round = -0.5
        else:
            reward_this_round = 0.1

        # Remeber new experience
        DQN.remember([currentState, action, reward_this_round, nextState], gameOver)
        inputs, targets = DQN.getBatch(model, batchSize)
        model.train_on_batch(inputs, targets)

        currentState = nextState
        gotReward = 0 #reset 
        totReward += reward_this_round

    brain.save_weights()
    epsilon *= epsilonDecayRate
    rewards_list.append(totReward)
    totReward = 0






                


        



