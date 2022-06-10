import math
import  numpy as np
import sys
'''
I will be implimenting the least squares solution to icp check notes and icp.ipynb for math and code 
i will be implimenting plane to point match as it converges faster
'''
class ICP():
    def __init__(self, max_iterations=20, tolerance=0.001):
        self.__max_iterations = max_iterations
        self.__tolerance = tolerance

    def get_correspondence_indices(self, P, Q) -> tuple[tuple[int,int]]:
        """For each point in P find closest one in Q."""
        p_size = P.shape[1]
        q_size = Q.shape[1]
        correspondences = []
        for i in range(p_size):
            p_point = P[:, i]
            min_dist = sys.maxsize
            chosen_idx = -1
            for j in range(q_size):
                q_point = Q[:, j]
                dist = np.linalg.norm(q_point - p_point)
                if dist < min_dist:
                    min_dist = dist
                    chosen_idx = j
            correspondences.append((i, chosen_idx))
        return correspondences
    
    def compute_normals(self, points, step=1): #pass q
        normals = [np.array([[0, 0]])]
        normals_at_points = []
        for i in range(step, points.shape[1] - step):
            prev_point = points[:, i - step]
            next_point = points[:, i + step]
            curr_point = points[:, i]
            dx = next_point[0] - prev_point[0] 
            dy = next_point[1] - prev_point[1]
            normal = np.array([[0, 0],[-dy, dx]])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal[[1], :])  
            normals_at_points.append(normal + curr_point)
        normals.append(np.array([[0, 0]]))
        return normals 

    def prepare_system_normals(self, x, P, Q, correspondences, Q_normals):
        H = np.zeros((3, 3))
        b = np.zeros((3, 1))
        chi = 0
        for i, j in correspondences:
            p_point = P[:, [i]]
            q_point = Q[:, [j]]
            normal = Q_normals[j]
            e = normal.dot(self.error(x, p_point, q_point))
            J = normal.dot(self.jacobian(x, p_point))
            H += J.T.dot(J)
            b += J.T.dot(e)
            chi += e.T * e
        return H, b, chi
    def icp_normal(self, P, Q, Q_normals, iterations=100, tolerance = 1):
        x = np.zeros((3, 1))
        # chi_values = []
        
        x_values = [x.copy()]  # Initial value for transformation.
        P_values = [P.copy()]
        P_latest = P.copy()
        corresp_values = []
        for _ in range(iterations):
            rot = self.R(x[2])
            t = x[0:2]
            correspondences = self.get_correspondence_indices(P_latest, Q)
            # corresp_values.append(correspondences)
            H, g, chi = self.prepare_system_normals(x, P, Q, correspondences, Q_normals)
            
            dx = np.linalg.lstsq(H, -g, rcond=None)[0]
            x += dx
            x[2] = math.atan2(np.sin(x[2]), np.cos(x[2])) # normalize angle
            # chi_values.append(chi.item(0)) # add error to list of errors
            x_values.append(x.copy())
            rot = self.R(x[2])
            t = x[0:2]
            if chi<=tolerance:
                break
            # P_latest = rot.dot(P.copy()) + t
            # P_values.append(P_latest)
        # corresp_values.append(corresp_values[-1])
        return rot,t, chi  

    #   def find_transform(self, points_a, points_b, iterations=100, tolerance=1):
    #     # swap x - y
    #     points_a = points_a[:, ::-1]
    #     points_b = points_b[:, ::-1]

    #     # initial values for tx, ty, angle
    #     params = np.array([0.0, 0.0, 0.0])

    #     for i in range(iterations):
    #         h_sum = np.zeros((3, 3))
    #         b_sum = np.zeros(3)

    #         # modify points with params
    #         angle = params[2]
    #         rot = create_rotation_matrix_2xy(angle)
    #         adjusted_points = points_a.dot(rot.T) #a
    #         adjusted_points += params[:2] #a

    #         # test if we can stop
    #         distances = self.__get_distances(adjusted_points, points_b) #adjusted is prediction and b is q
    #         mean_error = np.mean(distances)
    #         if mean_error < tolerance:
    #             break

    #         for pa, pb, pm in zip(points_a, points_b, adjusted_points):
    #             # Jacobian
    #             j = np.array([[1, 0, -math.sin(angle) * pa[0] - math.cos(angle) * pa[1]],
    #                           [0, 1, math.cos(angle) * pa[0] - math.sin(angle) * pa[1]]])

    #             # Hessian approximation
    #             h = j.T @ j

    #             # Right hand side
    #             e = pm - pb
    #             b = j.T @ e

    #             # accumulate
    #             h_sum += h
    #             b_sum += b

    #         params_update = -np.linalg.pinv(h_sum) @ b_sum
    #         params += params_update

    #     # Calculate an error
    #     rot = create_rotation_matrix_2xy(params[2])
    #     adjusted_points = points_a.dot(rot.T)
    #     adjusted_points += params[:2]
    #     distances = self.__get_distances(adjusted_points, points_b)
    #     mean_error = np.mean(distances)

    #     # make result
    #     rot3 = create_rotation_matrix_yx(np.degrees(params[2]))
    #     pos = params[:2][::-1]

    #     return rot3, pos, mean_error
   

    def jacobian(self, x, p_point):
        #x is the dummy approximation
        theta = x[2]
        J = np.zeros((2, 3))
        J[0:2, 0:2] = np.identity(2)
        J[0:2, [2]] = self.dR(theta).dot(p_point)
        return J

    def error(self, x, p_point, q_point):
        rotation = self.R(x[2])
        translation = x[0:2]
        prediction = rotation.dot(p_point) + translation #a
        return prediction - q_point

    def dR(self, theta):
        return np.array([[-np.sin(theta), -np.cos(theta)],
                        [np.cos(theta),  -np.sin(theta)]])

    def R(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])

    
  

    def __get_distances(self, points_a, points_b):
        assert points_a.shape == points_b.shape
        distances = np.linalg.norm(points_a - points_b, axis=1)
        return distances

        
