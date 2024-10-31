import pygame

pygame.init()

font = pygame.font.Font("./assets/flappy-bird-font.ttf", 38)
screen = pygame.display.set_mode((800, 400))

run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    screen.fill(0)


    image = pygame.image.load('assets/sprites/yellowbird-upflap.png').convert_alpha()
    mymask = pygame.mask.from_surface(image)

    masksurf = mymask.to_surface()

    


    rectangle = pygame.Surface((100, 100))
    rectangle.fill((255, 255, 255))

    screen.blit(rectangle, (400, 200))
    screen.blit(masksurf, (0, 0))


    textsurface=font.render('2', True, (255, 255, 255))

    screen.blit(textsurface, (20, 20))


    pygame.display.update()


pygame.quit()