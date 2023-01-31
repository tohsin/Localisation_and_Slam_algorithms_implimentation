from curses import KEY_LEFT
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

#plot preferences, interactive plotting mode
fig = plt.figure()
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def plot_state(mu, sigma, landmarks, map_limits):
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean of belief as current estimate
    estimated_pose = mu

    #calculate and plot covariance ellipse
    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0],estimated_pose[1]], width=width, height=height, angle=angle/np.pi*180)
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)
    
    plt.pause(0.01)

def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 
    
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    '''your code here'''
    '''***        ***'''
    mu[0] += delta_trans + np.cos(delta_rot1 + theta)
    mu[1] += delta_trans + np.sin(delta_rot1 + theta)
    mu[2] += delta_rot1 + delta_rot2


    Q = np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2] ])

    # the jacobian of control
    # G = np.array([ [1, 0, (-1 * delta_trans * np.sin(delta_rot1 + theta))]\
    #                 [0, 1, delta_trans * np.cos(delta_rot1 + theta)],\
    #                  [0, 0 , 1] ])

    G = np.array([[1.0, 0.0, -delta_trans * np.sin(theta + delta_rot1)],\
                [0.0, 1.0, delta_trans * np.cos(theta + delta_rot1)],\
                [0.0, 0.0, 1.0]])
    sigma = np.dot(np.dot(G , sigma), np.transpose(G) ) + Q
    return mu, sigma

def correction_step(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    # 
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x = mu[0]
    y = mu[1]
    theta = mu[2]

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    '''your code here'''
    '''***        ***'''
    # range_measurement is z or h and H is jacobian which we will stack
    H = []
    Z = []
    expected_ranges = []
    for i in range(len(ids)):
        elment_id = ids[i]
        element_range = ranges[i]

        lx , ly = landmarks[elment_id][0],landmarks[elment_id][1] 

        z_ = np.sqrt( (lx - x)**2 + (ly - y)**2 )

        # see proof in notes
        H_row = [(x - lx)/z_, (y - ly)/z_, 0]
        H.append(H_row)
        Z.append(ranges[i])
        expected_ranges.append(z_)

    #np.eye returns identity like matrix 
    # np.linalg.inv
    #noise for measurement
    R =  0.5 * np.eye(len(ids)) # scalar nultiplication
    H = np.array(H)
    Z = np.array(Z)
    expected_ranges = np.array(expected_ranges)
    kalman_gain = np.dot(np.dot(H, sigma), np.transpose(H)) + R
    kalman_gain = np.linalg.inv(kalman_gain)
    kalman_gain = np.dot( np.dot(sigma, np.transpose(H)), kalman_gain)

    mu = mu  + np.dot(kalman_gain , (Z -expected_ranges ))
    sigma = np.dot(np.eye(len(sigma)) - np.dot(kalman_gain,H),sigma)
    # sigma = np.dot(( np.eye(len(ids)) - (np.dot (kalman_gain , H)) ), sigma)


    return mu, sigma

def main():
    # implementation of an extended Kalman filter for robot pose estimation

    print( "Reading landmark positions")
    landmarks = read_world("EKF_Localisation_algorithm_assesment/data/world.dat")

    print ("Reading sensor data")
    # sensor_readings = read_sensor_data("data/sensor_data.dat")
    sensor_readings = read_sensor_data("EKF_Localisation_algorithm_assesment/data/sensor_data.dat")

    #initialize belief
    mu = [0.0, 0.0, 0.0]
    sigma = np.array([[1.0, 0.0, 0.0],\
                      [0.0, 1.0, 0.0],\
                      [0.0, 0.0, 1.0]])

    map_limits = [-1, 12, -1, 10]

    #run kalman filter
    for timestep in range(len(sensor_readings)//2):

        #plot the current state
        plot_state(mu, sigma, landmarks, map_limits)

        #perform prediction step
        mu, sigma = prediction_step(sensor_readings[timestep,'odometry'], mu, sigma)

        #perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

    plt.show()

if __name__ == "__main__":
    main()