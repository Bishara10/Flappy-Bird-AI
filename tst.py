import pygame

pygame.init()

screen = pygame.display.set_mode((800, 400))

run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    screen.fill(0)


    image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
    mymask = pygame.mask.from_surface(image)

    masksurf = mymask.to_surface()

    


    rectangle = pygame.Surface((100, 100))
    rectangle.fill((255, 255, 255))

    screen.blit(rectangle, (400, 200))
    screen.blit(masksurf, (0, 0))

    pygame.display.update()


pygame.quit()