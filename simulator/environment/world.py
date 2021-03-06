import math

from PIL import Image
from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np
import pygame


class World:
    """
        Physical environment simulation element
    """

    def __init__(self, map_file_name):
        self.__map_file_name = map_file_name
        self.__width = 0
        self.__height = 0
        self.__map = None
        self.read_map()

        # Initialize Hilbert curve object for generating obstacle ids
        max_side = max(self.__width, self.__height)
        side_len = math.ceil(math.log2(max_side * max_side)) // 2
        self.__hilbert_curve = HilbertCurve(p=side_len, n=2)

    def read_map(self):
        img = Image.open(self.__map_file_name)
        self.__width = img.size[0]
        self.__height = img.size[1]
        img.convert('1')  # make image binary
        self.__map = np.array(img)
        self.__map = self.__map[:, :, 0]

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    def draw(self, screen):
        transposed_map = np.transpose(self.__map)
        surf = pygame.surfarray.make_surface(transposed_map)
        screen.blit(surf, (0, 0))
        pygame.draw.circle(screen, color=(255, 0, 0), center=(self.__width // 2, self.__height // 2), radius=10)

    def allow_move(self, pos, size):
        obstacles_coords = self.get_obstacles_in_circle(pos, size)
        num = obstacles_coords.size
        return num == 0

    def get_obstacles_in_circle(self, pos, radius):
        x = self.__width // 2 + pos[1]
        y = self.__height // 2 - pos[0]
        x = np.clip(x, 0, self.__width).astype(int)
        y = np.clip(y, 0, self.__height).astype(int)

        h, w = self.__map.shape
        y_ind, x_ind = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_ind - x) ** 2 + (y_ind - y) ** 2).astype(int)
        region_coords = np.argwhere(dist_from_center <= radius)
        obstacles_mask = self.__map[region_coords[:, 0], region_coords[:, 1]] != 255
        region_coords = region_coords[obstacles_mask]
        # convert to world coordinates
        region_coords[:, 0] = h // 2 - region_coords[:, 0]
        region_coords[:, 1] = region_coords[:, 1] - w // 2
        return region_coords

    def is_obstacle(self, pos):
        x = self.__width // 2 + pos[1]
        y = self.__height // 2 - pos[0]
        if 0 <= y < self.__height and 0 <= x < self.__width:
            if self.__map[y, x] != 255:
                point_id = self.__hilbert_curve.distance_from_point([y, x])
                return True, point_id
        return False, None
