import pygame
import argparse
import sys 
from enum import Enum

import graph_slam_Simulator.graphSlam.backend
import graph_slam_Simulator.graphSlam.frontend
from simulator.rawsensorsview import RawSensorsView
from simulator.robot import Robot
from simulator.odometry import Odometry
from simulator.sensor import Sensor
from simulator.environment.world import World


class SimulationMode(Enum):
    RAW_SENSORS = 1,
    ICP_ADJUSTMENT = 2


def main():
    pygame.init()
    pygame.display.set_caption('ROBO World')
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', help='Enter algorithm ')
    args = parser.parse_args()

    # Create simulation objects

    # world = World(args.filename)
    world = World("assets/map.png")
    odometry = Odometry(mu=0, sigma=3)  # noised measurements
    sensor = Sensor(dist_range=350, fov=90, mu=0, sigma=1)  # noised measurements
    robot = Robot(odometry, sensor)
    sensors_view = RawSensorsView(world.height, world.width)
    slam_front_end = graph_slam_Simulator.graphSlam.frontend.Frontend(world.height, world.width)
    slam_back_end = graph_slam_Simulator.graphSlam.backend.Backend(edge_sigma=0.5, angle_sigma=0.1)

    # Initialize rendering
    screen = pygame.display.set_mode([world.width * 2, world.height])
    font = pygame.font.Font(pygame.font.get_default_font(), 24)
    sensors_text_surface = font.render('Sensors', True, (255, 0, 0))
    icp_text_surface = font.render('ICP', True, (255, 0, 0))
    text_pos = (15, 15)

    # Robot movement configuration
    rotation_step = 10  # degrees
    moving_step = 10  # points

    # make first initialization
    robot.move(0, world)
    sensors_view.take_measurements(odometry, sensor)
    slam_front_end.add_key_frame(sensor)

    # start simulation loop
    simulation_mode = SimulationMode.RAW_SENSORS
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    simulation_mode = SimulationMode.RAW_SENSORS
                if event.key == pygame.K_i:
                    simulation_mode = SimulationMode.ICP_ADJUSTMENT
                if event.key == pygame.K_s:
                    # we assume that we detect a loop so can try to optimize pose graph
                    loop_frame = slam_front_end.create_loop_closure(sensor)
                    slam_back_end.update_frames(slam_front_end.get_frames(), loop_frame)
                    break
                if event.key == pygame.K_LEFT:
                    robot.rotate(rotation_step, world)
                if event.key == pygame.K_RIGHT:
                    robot.rotate(-rotation_step, world)
                if event.key == pygame.K_UP:
                    robot.move(moving_step, world)
                if event.key == pygame.K_DOWN:
                    robot.move(-moving_step, world)

                sensors_view.take_measurements(odometry, sensor)
                slam_front_end.add_key_frame(sensor)

        world.draw(screen)
        robot.draw(screen, world.height, world.width)
        if simulation_mode == SimulationMode.RAW_SENSORS:
            sensors_view.draw(screen, offset=world.width)
            screen.blit(sensors_text_surface, dest=text_pos)
        if simulation_mode == SimulationMode.ICP_ADJUSTMENT:
            slam_front_end.draw(screen, offset=world.width)
            screen.blit(icp_text_surface, dest=text_pos)

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
