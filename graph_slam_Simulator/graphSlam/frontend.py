import numpy as np
from icp import ICP
import pygame
from util import to_screen_coords,transform_points
from backend import Backend
from skimage.draw import line_aa
from sklearn.neighbors import NearestNeighbors
from frame import RotationFrame
class Frontend():
    def __init__(self, world_height, world_width) -> None:
        self.__h = world_height
        self.__w = world_width
        # create a discrete world of size fille with white
        self.__local_map = self.intilase_map()
        self.__icp = ICP()
        self.__frames = []
        self.__knn = NearestNeighbors(n_neighbors=1)
        self.__frame_align_error = 10  # distance in pixels
        self.frames = []
    def intilase_map(self) ->np.ndarray:
        return np.full((self.__h , self.__w), 255, dtype=np.uint8)
    # get matching points and use for correspondebces
    def __find_frames_correspondences(self, frame_a : RotationFrame, frame_b : RotationFrame, min_dist : int = 5):
        ids_a = frame_a.observed_points[:, 2]
        ids_a = ids_a.reshape((-1, 1))

        ids_b = frame_b.observed_points[:, 2]
        ids_b = ids_b.reshape((-1, 1))

        estimator = self.__knn.fit(ids_b)
        distances, indices = estimator.kneighbors(ids_a, return_distance=True)
        # remove outliers
        idx_a = []
        idx_b = []
        used_ids = set()
        for d, i_b, i_a in zip(distances, indices.reshape((-1)).tolist(), range(ids_a.shape[0])):
            if i_b not in used_ids:
                if d <= min_dist:
                    idx_a.append(i_a)
                    idx_b.append(i_b)
            used_ids.add(i_b)
        return idx_a, idx_b

    def align_point_clouds(self, new_point_cloud : RotationFrame , old_global_point_cloud : RotationFrame):
        '''
         @parameters
         new_point_cloud

        '''
        rot, translation, chi = self.__icp.icp_normal( new_point_cloud.observed_points, old_global_point_cloud.observed_points)

        if chi < self.__frame_align_error: # we check if the match was successful
            # we alight new point cloud  to old one
            new_point_cloud.rotation = rot @ old_global_point_cloud.rotation
            new_point_cloud.position[:2] = translation

            new_point_cloud.position = new_point_cloud.rotation @ new_point_cloud.position
            new_point_cloud.position += old_global_point_cloud.position

            new_point_cloud.relative_icp_position = pos
            new_point_cloud.relative_icp_rotation = rot
            return True
        else:
            return False

    def create_new_frame(self, sensor):
        #  this creates a frame or pose and stores it along with it observed points
        obstacles = sensor.get_obstacles()
        if obstacles is not None:
            return RotationFrame(obstacles.copy())
        return None
    def add_key_frame(self, sensor):
        frame_candidate = self.create_new_frame(sensor)
        if frame_candidate:
            if len(self.__frames) > 0:
                # align current frame with the last one
                key_frame = self.__frames[-1]
                if not self.align_new_frame(frame_candidate, key_frame):
                    print('Failed to align frame"')
                    return False
            self.__frames.append(frame_candidate)
            return True
        else:
            return False


    def create_loop_closure(self, sensor):
         # we create a new frame upon closing the loop and align it with original frame
        frame_candidate = self.create_new_frame(sensor)
        if frame_candidate:
            if len(self.__frames) > 0:
                # align current frame with the first one
                original_frame = self.__frames[0]
                if self.align_point_clouds(frame_candidate, original_frame):
                    return frame_candidate
                else:
                    print('Failed to align frame"')
        return None

    def get_frames(self):
        return self.__frames

    def generate_local_map(self):
       # reinitialise the map or occupancy grid map
       self.__local_map = self.intilase_map()
       prev_pos = None
       frame : RotationFrame
       for frame in self.__frames:
            pixel_position = to_screen_coords(self.__h, self.__w, frame.position[:2])
           # draw position
            if not prev_pos:
                rr, cc, _ = line_aa(prev_pos[1], prev_pos[0], pixel_position[1], pixel_position[0])
                # to ensure we dont go out of screen we keep value between width
                rr = np.clip(rr, 0, self.__h - 1)
                cc = np.clip(cc, 0, self.__w - 1)
                #set pixel to black
                self.__local_map[rr, cc] = 0
            else:
                #set pixel to black
                self.__local_map[pixel_position[0], pixel_position[1]] = 0
            prev_pos = pixel_position

            points = frame.observed_points[:, :2]
            points = transform_points(points, frame.rotation, target_type=float)
            points += frame.position[:2]
            points = points.astype(int)

            # convert them into map coordinate system
            points[:, 0] = self.__h // 2 - points[:, 0]
            points[:, 1] += self.__w // 2
            points = np.clip(points, [0, 0],
                             [self.__h - 1, self.__w - 1])
            # draw
            self.__local_map[points[:, 0], points[:, 1]] = 0

    def draw(self, screen, offset):
        self.generate_local_map()
        transposed_map = np.transpose(self.__local_map)
        surf = pygame.surfarray.make_surface(transposed_map)
        screen.blit(surf, (offset, 0))

            