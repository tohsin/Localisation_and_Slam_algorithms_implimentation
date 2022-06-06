import pygame

from EKF_SLAM_Scratch import sensors,world

worldenv = world.CreateWorld((600, 1200))

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT():
            running = False
    
    pygame.display.update()