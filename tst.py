import pygame
from components.AllComponents import *

pygame.init()

font = pygame.font.Font("./assets/flappy-bird-font.ttf", 38)
screen = pygame.display.set_mode((800, 600))

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
    surf = pygame.Surface((10, 10))
    surf.fill((0, 255, 0))
    
    

    pipe = Pipe(False, 0, 100)
    pipesurfmask = pipe.mask.to_surface()
    pipesurfmask.fill((255, 255, 255))

    screen.blit(pipesurfmask, pipe.rect)


    pygame.display.update()


pygame.quit()