import evaluate_vins_accuracy as vins_acc
import bag_topic_extraction as bag_ex
import rosbag
import matplotlib.pyplot as plt
import numpy as np
import argparse
import rospy

# Parse the input arguments
parser = argparse.ArgumentParser()
parser.add_argument("bagfilename")
parser.add_argument("vins_topic")   # /vins_estimator/camera_pose
parser.add_argument("mocap_topic")  # /vrpn_client/vins/pose
parser.add_argument("--start",type=float,default=None)
parser.add_argument("--length",type=float,default=None)
parser.add_argument("--output")
args = parser.parse_args()

# Pull in the data
bag = rosbag.Bag(args.bagfilename)
# Time limited data
tStart = None
if args.start is not None:
    tStart = rospy.Time.from_sec(bag.get_start_time() + args.start)
vins_data = bag_ex.extractTopicWithHeader(bag, args.vins_topic, bag_ex.transformPoseToList_quat,tStart,args.length)
mocap_data = bag_ex.extractTopicWithHeader(bag, args.mocap_topic, bag_ex.transformPoseToList_quat,tStart,args.length)
# Full Data
full_vins_data = bag_ex.extractTopicWithHeader(bag, args.vins_topic, bag_ex.transformPoseToList_quat)
full_mocap_data = bag_ex.extractTopicWithHeader(bag, args.mocap_topic, bag_ex.transformPoseToList_quat)

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
        plt.plot(full_ts_mocap, error[i,:], 'r')
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

# Interpolate
full_ts_vins = full_vins_data[:,0]
full_ts_mocap = full_mocap_data[:,0]
full_vins_poses = full_vins_data[:,1:8]
full_mocap_poses = np.transpose(full_mocap_data[:,1:8])
vins_interp = vins_acc.interpolatePoses(full_ts_mocap, full_ts_vins, full_vins_poses)

# Find and plot error
vins_in_mocap = vins_acc.transformVINSToMocap(estimates, vins_interp)
error = vins_acc.error(full_mocap_poses,vins_in_mocap)

print(estimates)
if args.output is not None:
    f = open(args.output,"w")
    f.write("Bag File Name: " + args.bagfilename + "\n")
    f.write("VINS Topic Name: " + args.vins_topic + "\n")
    f.write("MOCAP Topic Name: " + args.mocap_topic + "\n")
    if args.start is not None:
        f.write("Start Time: " + str(args.start) + "\n")
    if args.length is not None:
        f.write("Length: " + str(args.length) + "\n")
    f.write("Mocap To VINS Pose: " + str(estimates[:7]) + "\n")
    f.write("IMU to Marker Pose: " + str(estimates[7:]) + "\n")
# Add additional metrics to write out here
    f.close()
plot_error()
