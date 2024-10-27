import pygame, random, time
from components.AllComponents import *
from pygame.locals import *
# from math import log10

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
    pos = SCREEN_WIDHT * i + 400
    pipes = get_random_pipes(pos)
    pipe_group.add(pipes[0])
    pipe_group.add(pipes[1])

    print(pipe_group)

    reward = Reward(pos + PIPE_WIDHT/2)
    reward_group.add(reward)

 

clock = pygame.time.Clock()

begin = True

while begin:

    clock.tick(32)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
        if event.type == KEYDOWN:
            if event.key == K_SPACE or event.key == K_UP:
                bird.bump()
                pygame.mixer.music.load(wing)
                pygame.mixer.music.play()
                begin = False

    screen.blit(BACKGROUND, (0, 0))
    screen.blit(BEGIN_IMAGE, (120, 150))


    if is_off_screen(ground_group.sprites()[0]):
        ground_group.remove(ground_group.sprites()[0])

        new_ground = Ground(GROUND_WIDHT - 20)
        ground_group.add(new_ground)

    bird.begin()
    ground_group.update()

    bird_group.draw(screen)
    ground_group.draw(screen)

    pygame.display.update()


while True:
    clock.tick(32)

    score = Score()
    display_score = score.update(new_val=SCORE)


    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
        if event.type == KEYDOWN:
            if event.key == K_SPACE or event.key == K_UP:
                bird.bump()
                pygame.mixer.find_channel().play(wing_sound)


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

    pygame.display.update()

    if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
        pygame.mixer.find_channel().play(hit_sound)
        time.sleep(1)
        break

    if (pygame.sprite.groupcollide(bird_group, reward_group, False, False, pygame.sprite.collide_mask)):
        reward_group.remove(reward_group.sprites()[0])
        pygame.mixer.find_channel().play(point_sound)
        SCORE += 1

pygame.quit()