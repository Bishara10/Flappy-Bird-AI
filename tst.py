import pygame
from components.AllComponents import *

pygame.init()

font = pygame.font.Font("./assets/flappy-bird-font.ttf", 38)
screen = pygame.display.set_mode((800, 600))
screenhight = 600
run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    screen.fill(0)


    # image = pygame.image.load('assets/sprites/yellowbird-upflap.png').convert_alpha()
    # mymask = pygame.mask.from_surface(image)

    # masksurf = mymask.to_surface()

    # rectangle = pygame.Surface((100, 100))
    # rectangle.fill((255, 255, 255))

    # bird = Bird()
    # bird_point = pygame.Surface((10, 10))
    # birdpointrect = bird.rect
    # bird_point.fill((0, 255, 0))

    toppoint = pygame.Surface((10, 10))
    toppoint.fill((0, 255, 0))
    botpoint = pygame.Surface((10, 10))
    botpoint.fill((0, 255, 0))

    pipe = Pipe(False, 400   , 100)
    pipesurfmask = pipe.mask.to_surface()
    pipesurfmask.fill((255, 255, 255))

    pipetop = Pipe(True, 400 , screenhight - 100 - PIPE_GAP)
    pipetopsurfmask = pipe.mask.to_surface()
    pipetopsurfmask.fill((255, 255, 255))

    screen.blit(pipesurfmask, pipe.rect)
    screen.blit(pipetopsurfmask, pipetop.rect)
    screen.blit(botpoint, pipe.rect)
    screen.blit(toppoint, (pipe.rect[0], pipe.rect[1] - PIPE_GAP))
    screen.blit(toppoint, (pipetop.rect[0], pipetop.rect[1] + screenhight - PIPE_GAP))



    pygame.display.update()


pygame.quit()