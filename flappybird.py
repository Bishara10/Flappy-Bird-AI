import pygame
from pygame.locals import *

class FlappyBird():
    def __init__(self):
        #VARIABLES
        self.SCREEN_WIDHT = 400
        self.SCREEN_HEIGHT = 600
        self.SPEED = 14
        self.MAXSPEED = 14
        self.GRAVITY = 1.9
        self.GAME_SPEED = 6
        self.GROUND_WIDHT = 2 * self.SCREEN_WIDHT
        self.GROUND_HEIGHT= 60


        PIPE_WIDHT = 80
        PIPE_HEIGHT = 500
        PIPE_GAP = 130

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
        self.screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')

        BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
        BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
        