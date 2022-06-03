from cmath import cos
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data

#add random seed for generating comparable pseudo random numbers
np.random.seed(123)

#plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def plot_state(particles, landmarks, map_limits):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.
    
    xs = []
    ys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)

def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits

    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles

def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound 
    # (jump from -pi to pi). Therefore, we generate unit vectors from the 
    # angles and calculate the angle of their average 

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations 
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        #make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    #calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]

def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise 
    # (probabilistic motion models slide 27)

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]


    alpha1 = noise[0]
    alpha2 = noise[1]
    alpha3 = noise[2]
    alpha4 = noise[3]
    # generate new particle set after motion update

    std_del_rot1 = ( alpha1 * abs(delta_rot1)) + (alpha2 * delta_trans)
    std_del_trans = (alpha3 * delta_trans ) + ( alpha4 * (abs(delta_rot1) + abs(delta_rot2)))
    std_del_rot2 = ( alpha1 * abs(delta_rot2)) + (alpha2 * delta_trans)
    new_particles = []
    
    '''your code here'''
    '''***        ***'''
    for particle in particles:
        new_particle = {}
        delta_rot1_noisy = delta_rot1 + np.random.normal(0,std_del_rot1 )
        delta_trans_noisy = delta_trans + np.random.normal(0,std_del_trans )
        delta_rot2_noisy = delta_rot2 + np.random.normal(0, std_del_rot2)


        new_particle['x'] = particle['x'] + ( delta_trans_noisy * np.cos(particle['theta']  + delta_rot1_noisy ) )
        new_particle['y'] = particle['y'] + (  delta_trans_noisy * np.sin(particle['theta']  + delta_rot1_noisy ) )
        new_particle['theta'] = particle['theta'] + delta_rot1_noisy + delta_rot2_noisy
        new_particles.append(new_particle)
    return new_particles

def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    # (probabilistic sensor models slide 33)
    #
    # The employed sensor model is range only.

    sigma_r = 0.2

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []
    
    '''your code here'''
    '''***        ***'''
    for particle in particles:
        all_meas_likelihood = 1.0 #

        for i in range(len(ids)):
            i_id = ids[i]
            measurement_range = ranges[i]

            lx = landmarks[i_id][0]
            ly = landmarks[i_id][1]
            px = particle["x"]
            py = particle["y"]

            # range tracing algorithm
            measurment_range_expected = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)

            
            # evaluate sensor model (probability density function of normal distribution)
            measurement_likelihood = scipy.stats.norm.pdf(measurement_range, measurment_range_expected, sigma_r)


            # combine (independent) measurements
            all_meas_likelihood = all_meas_likelihood * measurement_likelihood

        weights.append(all_meas_likelihood)





    #normalize weights
    normalizer = sum(weights)
    weights = weights / normalizer

    return weights

def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []

    '''your code here'''
    '''***        ***'''

    step = 1.0 / len(particles)
    u = np.random.uniform(0, step) # get random pointer to a particle
    c = weights[0]

    i = 0 
    new_particles = []
    # loop over all particle weights
    for particle in particles:
        # go through the weights until you find the particle
        # to which the pointer points
        while u > c:
            i = i + 1
            c = c + weights[i]
            # add that particle
        new_particles.append(particles[i])
        # increase the threshold
        u = u + step
    return new_particles





    return new_particles

def main():
    # implementation of a particle filter for robot pose estimation

    print ("Reading landmark positions")
    landmarks = read_world("Particle_filter_localisation/pf_framework/data/world.dat")

    print ("Reading sensor data")
    sensor_readings = read_sensor_data("Particle_filter_localisation/pf_framework/data/sensor_data.dat")

    #initialize the particles
    map_limits = [-1, 12, 0, 10]
    particles = initialize_particles(1000, map_limits)

    #run particle filter
    for timestep in range(len(sensor_readings)//2):

        #plot the current state
        plot_state(particles, landmarks, map_limits)

        #predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(sensor_readings[timestep,'odometry'], particles)

        #calculate importance weights according to sensor model
        weights = eval_sensor_model(sensor_readings[timestep, 'sensor'], new_particles, landmarks)

        #resample new particle set according to their importance weights
        particles = resample_particles(new_particles, weights)

    plt.show('hold')

if __name__ == "__main__":
    main()