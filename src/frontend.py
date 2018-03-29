import evaluate_vins_accuracy as vins_acc
import bag_topic_extraction as bag_ex
import rosbag
import matplotlib.pyplot as plt

#parse the input arguments
parser = argparse.ArgumentParser()
parser.add_argument("bagfilename")
parser.add_argument("vins_topic")
parser.add_argument("mocap_topic")
args = parser.parse_args()

#pull in the data
bag = rosbag.Bag(args.bagfilename)
vins_data = bag_ex.extractTopicWithHeader(bag, args.vins_topic, bag_ex.transformPoseToList_quat)
mocap_data = bag_ex.extractTopicWithHeaderbag, args.mocap_topic, bag_ex.transformPoseToList_quat)


ts_vins = []
ts_mocap = []

def plotting():

    titles = ['','translation in x','translation in y', 'translation in z','orientation in x', 'orientation in y', 'orientation in z']

    for i in range(1,7):
        plt.figure(i)
        plt.plot(vins_data[:,0],vins_data[:,i],'r',mocap_data[:,0],mocap_data[:,i],'b')
        plt.title(titles[i])
    plt.show()

#plotting()

for i in range(0,400):
    ts_vins.append(vins_data[i, 0])
    ts_mocap.append(mocap_data[i, 0])

vins_poses = vins_data[:,1:8]
mocap_poses = mocap_data[:,1:8]

#estimates = vins_acc.estimate(ts_vins, vins_poses, ts_mocap, mocap_poses)
