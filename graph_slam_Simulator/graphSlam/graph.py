import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from util import t2v,v2t, id2index

class Graph(object):
    
    '''
    contains edges - nodes [ list]
    verices or constrainsts as  list
    '''
    def __init__(self, sigma_y, sigma_angle, sigma_x) -> None:
        self.__edges = []
        self.__current_node_index = -1
        self.__vertices = {} # key , node
        self.__information_matrix_omega = np.diag([sigma_y, sigma_x, sigma_angle])
        self.__information_matrix_omega = np.linalg.inv(self.__information_matrix_omega)

    @property
    def current_node_index(self):
        return self.__current_node_index


    def add_node(self, key, tx, ty, rot):
        # pose vector should be a pyhthon array
        self.__vertices[key] = np.array([tx, ty, rot])

    # add constraint or edge depends on literature
    def add_edge(self, node_key_i : int, node_key_j : int, tx, ty, rot, noise_model = None): 
        # pose vector should be a pyhthon array

        # check if nodes actually exist
        if (node_key_i in self.__nodes) and (node_key_j in self.__nodes):
            if noise_model == None:
                # set the noise parameter to information matrix omega
                noise_model = self.__information_matrix_omega
            edge = node_key_i, node_key_j, np.array(pose_vector), noise_model
            self.__edges.append(edge)

    def clear_memory(self ):
        '''
        clear graph from memory
        '''
        self.__edges.clear()
        self.__vertices.clear()

    def __get_error(self, node_key_i, node_key_j, edge_transform_z):
        '''
        computer error between nodes from odomety and loop closure
        @ is used for matrix multiplication

        parameters
            omega_information_matrix
            and e
        '''
        vertex_i , vertex_j = self.__vertices[node_key_i] , self.__vertices[node_key_j] 
        # we have to convert vector to rotational matrix using formula in paper
        #error = v2t (z_inverse * (x_i_ inverse  * xj))
        t_i = v2t(vertex_i)
        t_j = v2t(vertex_j)
        t_z = v2t(edge_transform_z)
        t_z_inv = np.linalg.inv(t_z)
        t_i_inv = np.linalg.inv(t_i)
        error = t2v(   t_z_inv @ (t_i_inv @ t_j))
        return error
    
    def __get_jacobians(self, node_key_i, node_key_j, edge_transform_z):
        vertex_i , vertex_j = self.__vertices[node_key_i] , self.__vertices[node_key_j] 
        '''
        rotation matrix can be constructed with sine and cosines of theta which is 3rd variable in pose

        '''
        sine_i = np.sin(vertex_i[2])
        cos_i = np.cos(vertex_i[2])
        partial_det_rot_i = np.array([-sine_i,cos_i], [-cos_i, -sine_i]).T #check jacobian formula
        

        diffrence_tij = np.array([vertex_j[:2] - vertex_i[:2]]).T # subtract the poses vectors t_j - t_i

        t_i = v2t(vertex_i)
        t_z = v2t(edge_transform_z)
        r_i = t_i[:2, :2]
        r_z = t_z[:2, :2]

        a_ij = np.vstack(
            (
                np.hstack((-r_z.T @ r_i.T, (r_z.T @ partial_det_rot_i.T) @ diffrence_tij)),
                          [0, 0, -1])
            )

        b_ij = np.vstack(
            (
                np.hstack((r_z.T @ r_i.T, np.zeros((2, 1)))),
                          [0, 0, 1])
                          )
        return a_ij, b_ij

    def optimise(self, tresh = 1e-5, iter = 1000):
        num_params = 3
        for _ in range(iter):
            #build linear system
            num_nodes = len(self.__vertices)

            # define the sparce matrix needed for big H
            #solve H * delx = -b 
            dim_v = num_nodes * num_params
            H = scipy.sparse.csc_matrix((dim_v ,dim_v))

            b = scipy.sparse.csc_matrix((dim_v, 1))

            for edge in self.__edges:
                node_key_i, node_key_j, edge_transform, omega = edge
                  
                error = self.__get_error(node_key_i, node_key_j, edge_transform)

                  #comute jacobian
                a_ij , b_ij = self.__get_jacobians(node_key_i, node_key_j, edge_transform)

                # from paper read for specifications to compute H
                H_ii = a_ij.T @ omega @ a_ij
                H_ij = a_ij.T @ omega @ b_ij
                H_jj = b_ij.T @ omega @ b_ij
                b_i = -a_ij.T @ omega @ error
                b_j = -b_ij.T @ omega @ error

                # now we update the actual matrix we calculate the spot
                H[id2index(node_key_i, num_params), id2index(node_key_i, num_params)] += H_ii
                H[id2index(node_key_i, num_params), id2index(node_key_j, num_params)] += H_ij
                H[id2index(node_key_j, num_params), id2index(node_key_i, num_params)] += H_ij.T
                H[id2index(node_key_j, num_params), id2index(node_key_j, num_params)] += H_jj

                b[id2index(node_key_i,num_params)] += b_i 
                b[id2index(node_key_j)] += b_j


            # The system (H b) is built only from relative constraints so H is not full rank.
            # So we fix the position of the 1st vertex
            H[:num_params, :num_params] += np.eye(3)

            verices_update = scipy.sparse.linalg.spsolve(H ,b) # solver to solve linear equation
            verices_update[np.isnan(verices_update)] = 0 # remove very small values or none
            verices_update = np.reshape(verices_update, (len(self.__vertices), num_params))

            self.__update_vertices(verices_update)

            # compute a mean error
            mean_error = 0
            for edge in self.__edges:
                value_index_i, value_index_j, edge_transform_z, _ = edge
                mean_error += self.__get_error(value_index_i, value_index_j, edge_transform_z)
            mean_error /= len(self.__edges)

            # check if we converged
            if (mean_error <= tresh).all():
                break

    def __update_vertices(self, values_update):
        # x + del_x update on every edge
         for value_id, update in enumerate(values_update):

            self.__edges[value_id] += update

    def get_pose_at(self, index):
        v = self.__vertices[index]
        v = v2t(v)
        rot = np.identity(3)
        rot[:2, :2] = v[:2, :2]
        return v[0, 2], v[1, 2], rot

    def get_vector_pose_at(self, index):
        v = self.__values[index]
        return v
