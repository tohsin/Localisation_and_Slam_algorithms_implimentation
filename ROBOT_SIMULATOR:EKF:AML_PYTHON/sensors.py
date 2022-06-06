from statistics import covariance
from tkinter import Y
from turtle import color, distance
import numpy as np
import pygame
import math
def uncertainty_add(distance, angle, sigma):
    mean = np.array([distance, angle])
    covariance = np.diag(sigma **2)
    distance,angle = np.random.multivariate_normal(mean, covariance ) 
    distance = max(distance,0)
    angle = max(angle,0)
    return [ distance, angle ]

class LidarScanner:
    def __init__(self, range,map, noise) -> None:
        self.range = range
        self.speed = 4 # rounds per secound
        #use normal distribution for the noise
        self.sigma = np.array([noise[0], noise[1]])
        self.position = (0,0)
        self.W, self.H = pygame.display.get_surface().get_size()
        self.pointCloud  = []
    def distance(self, obstaclePosition):
        euc_dist = np.sqrt( ((obstaclePosition[1] - self.position[0])**2) + ((obstaclePosition[1] - self.position[0])**2))
        return euc_dist

    def sence_obstacle(self):
        data = []
        x1,y1 = self.position[0], self.position[1]
        
        for angle in np.linspace(0, 2 * math.pi,60, False): # loop through evenly spaced interval between 0 and 2pi like a lidar
            x2 , y2 = ( x1 + self.range * math.cos(angle) , y1 - self.range * math.sin(angle))

            for i in range(0, 100):
                u = i /100
                x =  int(x2 * u + x1* (1-u))
                y =  int(y2 * u + y1* (1-u))

                if 0<x<self.W and 0< y<self.H:
                    color = self.map.get_at((x,y))
                    if (color[0], color[1], color[2]) == (0,0,0):
                        distance = self.distance((x,y))
                        output = uncertainty_add(distance,angle,self.sigma)
                        output.append(self.position)
                        data.append(output)
                        break
        return len(data)>0

    

