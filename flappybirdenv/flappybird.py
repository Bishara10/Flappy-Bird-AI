import pygame
from pygame.locals import *
from .AllComponents import *
from .flappybird_constants import *


class FlappyBird():
    def __init__(self):
        #initialize pygame resources
        pygame.init()
        pygame.mixer.init()
        pygame.font.init()
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # initialize game window 
        self.screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')
        pygame.mixer.set_num_channels(10)

        # constant score and epoch positions
        self.score_pos = (SCREEN_WIDHT // 2 - font.get_height() // 2, SCREEN_HEIGHT//20)
        self.epoch_pos = (SCREEN_WIDHT // 18 - font.get_height() // 2, SCREEN_HEIGHT//20)

        # load background        
        self.BACKGROUND = pygame.image.load(os.path.join(current_dir, "assets", "sprites","background-day.png"))
        self.BACKGROUND = pygame.transform.scale(self.BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))

        # initialize essential groups to store game objects
        self.pipe_group = pygame.sprite.Group()
        self.reward_group = pygame.sprite.Group()
        self.ground_group = pygame.sprite.Group()
        self.bird_group = pygame.sprite.Group()
        self.top_boundary = TopBoundary()

        # initialize bird and add it to the bird group
        self.bird = Bird()
        self.bird_group.add(self.bird)
        # spawn bird at the left hand side of the screen at a predefined height
        self.bird.begin()

        # initialize ground sprites
        for i in range (2):
            ground = Ground(GROUND_WIDHT * i)
            self.ground_group.add(ground) 

        # initialize pipes and reward groups
        self._spawnFirstPipes()

        # start score at 0
        self.score = 0

        self.clock = pygame.time.Clock()

    # Spawn the first 3 pipes and rewards at predefined locations
    def _spawnFirstPipes(self):
        for i in range (3):  # spawn 3 pipes
            pos =  250 * i + 400  # spawn locations
            pipesAndReward = self._getRandomPipesAndReward(pos)  # generate pipes at random heights
            self.pipe_group.add(pipesAndReward[0])
            self.pipe_group.add(pipesAndReward[1])
            self.reward_group.add(pipesAndReward[2])


    # Generate random pipe and reward at a predefined position
    def _getRandomPipesAndReward(self, xpos):
        # chooses a random pipe height 
        size = random.randint(100, 350)

        # Create pipes and a reward
        pipe = Pipe(False, xpos, size)
        pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
        reward = Reward(xpos + PIPE_WIDHT/2)

        return pipe, pipe_inverted, reward
    

    # resets the game to the starting positions for the necessary objects
    def resetGame(self):
        # empty groups
        self.pipe_group.empty()
        self.reward_group.empty()
        # spawn new pipes
        self._spawnFirstPipes()
        # reset bird's position
        self.bird.begin()
        #reset score
        self.score = 0

    # returns the current game state
    def getGameState(self):
        state_params = []

        # choose the nearest pipes by checking their distance from the bird
        chosenpipeindex = 0
        nextpipe_x = self.pipe_group.sprites()[chosenpipeindex].rect[0] + PIPE_WIDHT
        if (nextpipe_x < self.bird.rect[0]):
            chosenpipeindex = 2  # if the next pipe is to the left of the bird, choose the right pipe

        # get the nearest pipes' vertical positions
        nextpipe_x = self.pipe_group.sprites()[chosenpipeindex].rect[0] - self.bird.rect[0]
        nextpipe_bottom_y = self.pipe_group.sprites()[chosenpipeindex].rect[1]
        nextpipe_top_y = nextpipe_bottom_y - PIPE_GAP

        # gap middle vertical position
        pipe_middle_y = nextpipe_bottom_y - (PIPE_GAP / 2)

        # bird's vertical position and bird speed
        bird_y = self.bird.rect[1]
        bird_speed = self.bird.speed

        # compile the state parameters
        state_params.append(nextpipe_top_y)
        state_params.append(nextpipe_bottom_y)
        state_params.append(pipe_middle_y)
        state_params.append(bird_y)
        state_params.append(bird_speed)

        return state_params
    
    # Advance the game by 1 frame
    def _handleEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

    # takes one action and advances the game by 1 step (1 frame)
    def step(self, action=None, epoch=-1):
        # tick pygame clock and reset variables 
        self.clock.tick_busy_loop(30)
        gameOver = gotReward = False

        # render epoch if necessary and the score
        display_score = font.render(str(self.score), True, (255, 255, 255))
        if epoch != -1:
            display_epoch = font2.render(f"epoch: {epoch}", True, (255, 255, 255))

        # handle user events
        self._handleEvents()

        # take action
        if action == 1:
            self.bird.bump()

        elif action == None:
            raise Exception("no action")


        # chec if the ground sprite is off screen and update it
        if is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])

            new_ground = Ground(GROUND_WIDHT - 20)
            self.ground_group.add(new_ground)

        # Spawn new pipes if the first pipe is off screen
        if is_off_screen(self.pipe_group.sprites()[0]):
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            self.pipe_group.remove(self.pipe_group.sprites()[0])

            pipesAndReward = self._getRandomPipesAndReward(DEFAULT_PIPE_SPAWN_POINT)

            self.pipe_group.add(pipesAndReward[0])
            self.pipe_group.add(pipesAndReward[1])
            self.reward_group.add(pipesAndReward[2])

        # Call the update() method for each of the sprite group objects, which updates their positions
        self.bird_group.update()
        self.ground_group.update()
        self.pipe_group.update()
        self.reward_group.update()

        # draw the background
        self.screen.blit(self.BACKGROUND, (0, 0))

        # Call the draw() method for each of the sprite group objects and draw them on the screen
        self.bird_group.draw(self.screen)
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)
        self.reward_group.draw(self.screen)

        # Display the score, epoch and top boundary.
        self.screen.blit(display_score, self.score_pos)
        self.screen.blit(display_epoch, self.epoch_pos)
        self.screen.blit(self.top_boundary.surf, (0, -4))

        # Check for collisions between the bird and any of the ground, pipes, or top boundary
        if (pygame.sprite.groupcollide(self.bird_group, self.ground_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.collide_mask(self.bird, self.top_boundary)):
            # pygame.mixer.find_channel().play(hit_sound)
            gameOver = True

        # Check if the bird has captured a reward
        if (pygame.sprite.groupcollide(self.bird_group, self.reward_group, False, False, pygame.sprite.collide_mask)):
            # remove the captured reward object from game
            self.reward_group.remove(self.reward_group.sprites()[0])

            # play point sound
            pygame.mixer.find_channel().play(point_sound)

            # increment score
            self.score += 1

            # set gotReward to true
            gotReward = True

        # update entire game display
        pygame.display.update()

        return gameOver, gotReward

        

    


