import evaluate_vins_accuracy as vins_acc
import bag_topic_extraction as bag_ex
import rosbag
import matplotlib.pyplot as plt
import numpy as np
import argparse
import rospy
import tf.transformations as tf

# Parse the input arguments
parser = argparse.ArgumentParser()
parser.add_argument("bagfilename",help="Bag File to use")
parser.add_argument("vins_topic",help="VINS data topic.  Use /vins_estimator/camera_pose")
parser.add_argument("mocap_topic",help="Motion Capture data topic.  Use /vrpn_client/vins/pose")
parser.add_argument("--start",type=float,default=None,help="Start calibrating after X seconds")
parser.add_argument("--length",type=float,default=None,help="Calibrate using X seconds")
parser.add_argument("--output",help="Add the results to the provided file name")
parser.add_argument("--plot",help="Save the plots with the provided file name")
parser.add_argument("--show",help="Show the plots")
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

def plot_error():
    ''' Function to plot the error between VINS and Mocap poses'''
    titles = ['error in translation in x','error in translation in y', 'error in translation in z','error in orientation in roll', 'error in orientation in pitch', 'error in orientation in yaw']
    plotsuffix = ['X','Y','Z','Roll','Pitch','Yaw']
    for i in range(0, 6):
        plt.figure(i)
        plt.plot(full_ts_mocap, error[i,:], 'r')
        plt.title(titles[i])
        plt.savefig(args.plot+plotsuffix[i]+".png")
    if args.show is not None:
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
    f = open(args.output,"a")
    formatstring = "{:>15}, {:>7}, {:>7}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n"
    output = [args.bagfilename, args.start, args.length]
    for i in range(0,3):
        output.append(estimates[i])
    rpy = tf.euler_from_quaternion(estimates[3:7])
    for i in range(0,3):
        output.append(rpy[i])
    for i in range(7,10):
        output.append(estimates[i])
    rpy = tf.euler_from_quaternion(estimates[10:14])
    for i in range(0,3):
        output.append(rpy[i])
    for i in range(0,6):
        output.append(np.mean(np.absolute(error[i,:])))
    f.write(formatstring.format(*output))
# Add additional metrics to write out here
    f.close()
if args.plot is not None:
    plot_error()
