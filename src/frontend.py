import argparse
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import bag_topic_extraction as bag_ex
import evaluate_vins_accuracy as eval_vins
#parse the input arguments
parser = argparse.ArgumentParser()
parser.add_argument("bagfilename")
parser.add_argument("vins_topic")
parser.add_argument("mocap_topic")
args = parser.parse_args()
#pull in the data
databag = rosbag.Bag(args.bagfilename)
vinsdata = bag_ex.extractTopicWithHeader(databag, args.vins_topic, bag_ex.transformPoseToList)
mocapdata = bag_ex.extractTopicWithHeader(databag, args.mocap_topic, bag_ex.transformPoseToList)
#run the estimate script
ts_vins = vinsdata(0,:)
pose_vins = vinsdata(1:,:)
ts_mocap = mocapdata(0,:)
pose_mocap = mocapdata(1:,:)
estimates = eval_vins.estimate(ts_vins,pose_vins,ts_mocap,pose_mocap)
#find the error
vins_in_mocap = eval_vins.transformVINSToMocap(estimates, pose_vins)
error = pos_mocap - vins_in_mocap
plt.subplot(711)
plt.plot(error(0,:))
plt.subplot(712)
plt.plot(error(1,:))
plt.subplot(713)
plt.plot(error(2,:))
plt.subplot(714)
plt.plot(error(3,:))
plt.subplot(715)
plt.plot(error(4,:))
plt.subplot(716)
plt.plot(error(5,:))
plt.subplot(717)
plt.plot(error(6,:))
#close the bag
databag.close()
