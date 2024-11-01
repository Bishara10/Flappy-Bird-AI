import pygame, random, time
from pygame.locals import *
from math import log10

#VARIABLES
SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600
SPEED = 14
GRAVITY = 1.9
GAME_SPEED = 6
GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT= 60


PIPE_WIDHT = 80
PIPE_HEIGHT = 500
PIPE_GAP = 120

SCORE = 0

wing = 'assets/audio/wing.wav'
hit = 'assets/audio/hit.wav'
point = 'assets/audio/point.wav'

pygame.init()
pygame.mixer.init()
pygame.font.init()
font = pygame.font.Font("./assets/flappy-bird-font.ttf", 42)
font2 = pygame.font.Font("./assets/flappy-bird-font.ttf", 28)

wing_sound = pygame.mixer.Sound(wing)
hit_sound = pygame.mixer.Sound(hit)
point_sound = pygame.mixer.Sound(point)

class Bird(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.images =  [pygame.image.load('assets/sprites/yellowbird-upflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/yellowbird-midflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/yellowbird-downflap.png').convert_alpha()]

        self.speed = SPEED

        self.current_image = 0
        self.image = self.images[0]
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2.5

        self.current_angle = 0

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.image = pygame.transform.rotate(self.image, 30)
        self.speed += GRAVITY
        if self.speed > 14:
            self.speed = 14
        self.current_angle = (self.current_angle - 4) if (self.current_angle > -90) else -90

        self.image = pygame.transform.rotate(self.image, self.current_angle)
        # self.rect = self.image.get_rect(center=self.rect.center)

        #UPDATE HEIGHT
        self.rect[1] += self.speed

    def bump(self):
        self.current_angle = 30
        self.image = pygame.transform.rotate(self.image, self.current_angle)
        # self.rect = self.image.get_rect(center=self.rect.center)
        self.speed = -SPEED

        # make sure bird doesn't fly above the boundaries of the screen
        if self.rect[1] < 0 :
            self.rect[1] = 0

    def begin(self):
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2.5
        self.current_image = (self.current_image + 1) % 3
        self.current_angle = 0
        self.image = self.images[self.current_image]
        self.image = pygame.transform.rotate(self.image, self.current_angle)



class Pipe(pygame.sprite.Sprite):

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))


        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize


        self.mask = pygame.mask.from_surface(self.image)


    def update(self):
        self.rect[0] -= GAME_SPEED


class Score():
    def __init__(self):
        self.score = 0
        self.pos = (SCREEN_WIDHT // 2 - font.get_height() // 2, SCREEN_HEIGHT//20)


    def update(self, new_val):
        self.score = new_val
        score_disp = font.render(str(new_val), True, (255, 255, 255))
        return score_disp
    

class TopBoundary(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.surf = pygame.Surface((SCREEN_WIDHT, 1))
        self.surf.fill((255, 255, 255))
        self.surf.set_alpha(0)

        self.rect = self.surf.get_rect()

        self.mask = pygame.mask.from_surface(self.surf)

        # self.rect[0] = 0
        # self.rect[1] = 0


class Reward(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.surf = pygame.Surface((3, SCREEN_HEIGHT - GROUND_HEIGHT))
        self.surf.fill((255, 255, 255))
        self.surf.set_alpha(120)  ######### Makes it invisible
          
        self.rect = self.surf.get_rect()
        self.rect[0] = xpos

        self.mask = pygame.mask.from_surface(self.surf)

        self.image = self.surf

        self.initial_xpos = xpos


    def get_position(self):
        return (self.rect[0], self.rect[1])
    
    def update(self):
        self.rect[0] -= GAME_SPEED

    def reset(self):
        self.rect[0] = SCREEN_WIDHT * 2


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDHT, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):
    size = random.randint(100, 300)

    # pipes
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)

    reward = Reward(xpos + PIPE_WIDHT/2) # was xpos + PIPE_WIDTH/2 

    return pipe, pipe_inverted, reward


    