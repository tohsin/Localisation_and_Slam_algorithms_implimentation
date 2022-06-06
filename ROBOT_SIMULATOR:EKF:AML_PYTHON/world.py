from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
import math
import pygame

class CreateWorld:
    def __init__(self, size) -> None:
        pygame.init()
        self.laserScans = []
        self.exMap = pygame.image.load('EKF_SLAM_Scratch/map.png')
        self.mapHeight , self.mapWidth = size
        self.windowName = 'SLAM_SIMULATOR'
        pygame.display.set_caption(self.windowName) 
        self.map  = pygame.display.set_mode((self.mapHeight,self.mapWidth))
        self.map.blit(self.exMap,(0,0)) # overlay something on a window 
        # color to be used
        self.black = (0,0,0)
        self.grey = (70, 70, 70)
        self.blue = (0 ,0 ,255)
        self.green = (0, 255 , 0)
        self.red = ( 255, 0, 0)
        self.white = (255, 255, 255)
    def AD2POS(self, distance, angle, position):
        x = distance * math.cos(angle) + position[0]
        y = - distance *math.sin(angle) + position[1]
        return (int(x),int(y))
    def data_storage(self,data):
        print(len(self.laserScans))
        for element in data:
            point = self.AD2POS(element[0],element[1], element[2])
            if point not  in self.laserScans: 
                self.laserScans.append(point)

    def show_sensor_data(self):
        self.infomap = self.map.copy()
        for point in self.laserScans:
            self.infomap.set_at((int(point[0]), int(point[1]) , int(point[2])))