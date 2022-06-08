import time
from turtle import position
from typing import List
from graph import Graph
from frame import RotationFrame


class Backend():
    def __init__(self ,edge_sigma, angle_sigma) -> None:

        self.__graph = Graph(   edge_sigma_x=edge_sigma, 
                                edge_sigma_y=edge_sigma,
                                edge_sigma_angle=angle_sigma)


    def update_frames(self, rotationFrames : List[RotationFrame], loop_frame : RotationFrame)-> None:
        '''
        update graph and also gram the position from the simulator after loop closure
        '''
        start_time = time.perf_counter()
        print('Pose Graph optimization started...')
        # clean graph first
        self.__graph.clear()
        index_current_vertex = self.__graph.current_node_index
        # update from all the frames and process
        for frame in rotationFrames:
            position_y = frame.position[1]
            position_x = frame.position[0]
            rot_matrix = frame.rotation[:2, :2]  # slice in x axis and y axis
            self.__graph.add_node(index_current_vertex , position_x, position_y, rot_matrix.T)

            # relative frame 
            edge_ty = frame.relative_icp_position[1]
            edge_tx = frame.relative_icp_position[0]
            edge_rot = frame.relative_icp_rotation[:2, :2]
            # edge will be between the current one and previous ones
            self.__graph.add_edge(index_current_vertex-1, index_current_vertex ,edge_ty, edge_tx, edge_rot.T )
            index_current_vertex+=1

        # we do for x_i and x_j and for loop constraint we dont implment loop closure detection
        loop_ty = loop_frame.relative_icp_position[1]
        loop_tx = loop_frame.relative_icp_position[0]
        loop_rot = loop_frame.relative_icp_rotation[:2, :2]

        #join first one and last one
        self.__graph.add_edge(index_current_vertex-1,  
                                self.__graph.current_node_index + 1,
                                loop_ty, loop_tx, loop_rot.T)

        self.__graph.optimise()

        vertex_index = self.__graph.current_node_index + 1 # get first edge
        # we want to loop through each edge  and update frame which we will use 
        for frame in rotationFrames:
            tx, ty, rot = self.__graph.get_pose_at(vertex_index)
            frame.position[:2] = ty, tx
            frame.rotation = rot.T
            vertex_index += 1


        end_time = time.perf_counter()
        print(f'Pose Graph optimization finished in {end_time - start_time} seconds')
