import evaluate_vins_accuracy as vins_acc
import bag_topic_extraction as bag_ex
import rosbag
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Parse the input arguments
parser = argparse.ArgumentParser()
parser.add_argument("bagfilename")
parser.add_argument("vins_topic")   # /vins_estimator/camera_pose
parser.add_argument("mocap_topic")  # /vrpn_client/vins/pose
args = parser.parse_args()

# Pull in the data
bag = rosbag.Bag(args.bagfilename)
vins_data = bag_ex.extractTopicWithHeader(bag, args.vins_topic, bag_ex.transformPoseToList_quat)
mocap_data = bag_ex.extractTopicWithHeader(bag, args.mocap_topic, bag_ex.transformPoseToList_quat)

def plot_data():
    ''' Function to plot data read from the bag'''
    titles = ['','translation in x','translation in y', 'translation in z','orientation in x', 'orientation in y', 'orientation in z']

    for i in range(1,6):
        plt.figure(i)
        plt.plot(vins_data[:,0],vins_data[:,i],'r',mocap_data[:,0],mocap_data[:,i],'b')
        plt.title(titles[i])
    plt.show()

def plot_error():
    ''' Function to plot the error between VINS and Mocap poses'''
    for i in range(0, 6):
        titles = ['error in translation in x','error in translation in y', 'error in translation in z','error in orientation in roll', 'error in orientation in pitch', 'error in orientation in yaw']
        plt.figure(i)
        print(np.shape(ts_mocap))
        print(np.shape(error[i,:]))
        plt.plot(ts_mocap, error[i,:], 'r')
        plt.title(titles[i])
    plt.show()

#plot_data()

ts_vins = []
ts_mocap = []
ts_vins = vins_data[:,0]
ts_mocap = mocap_data[:,0]
vins_poses = vins_data[:,1:8]
mocap_poses = mocap_data[:,1:8]
mocap_poses = np.transpose(mocap_poses)

# Find estimates
estimates = vins_acc.estimate(ts_vins, vins_poses, ts_mocap, mocap_poses)
vins_interp = vins_acc.interpolatePoses(ts_mocap, ts_vins, vins_poses)

# Find and plot error
vins_in_mocap = vins_acc.transformVINSToMocap(estimates, vins_interp)
error = vins_acc.error(mocap_poses,vins_in_mocap)
print(np.shape(error))

print(estimates)
plot_error()
