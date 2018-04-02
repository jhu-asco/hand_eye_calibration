import bag_topic_extraction as mod
import unittest
import numpy.testing as np_testing
import numpy as np
import rosbag
from geometry_msgs.msg import PoseStamped

class TestBagTopicExtraction(unittest.TestCase):

    def generatePoseMsg(self,r):
        if r == 0:
            val = [1, 1, 1, "first", 1, 0, 1, 0, 0.4794, 0, 0.8776]
        elif r == 1:
            val = [10, 1, 1, "second", 1, 0, 1, 0, 0.4794, 0, 0.8776]
        msg = PoseStamped()
        msg.header.seq = val[0]
        msg.header.stamp.secs = val[1]
        msg.header.stamp.nsecs = val[2]
        msg.header.frame_id = val[3]
        msg.pose.position.x = val[4]
        msg.pose.position.y = val[5]
        msg.pose.position.z = val[6]
        msg.pose.orientation.x = val[7]
        msg.pose.orientation.y = val[8]
        msg.pose.orientation.z = val[9]
        msg.pose.orientation.w = val[10]
        return msg

    def generateRosBag(self):
        bag = rosbag.Bag('test.bag', 'w')
        try:
            pose_msg = self.generatePoseMsg(1)
            first_pose_msg = self.generatePoseMsg(0)
            bag.write('/vins_estimator/camera_pose',first_pose_msg)
            bag.write('/vins_estimator/camera_pose',pose_msg)
        finally:
            bag.close()
        return bag

    def testTransformPoseToList(self):
        pose_msg = self.generatePoseMsg(1)
        list_transformed = mod.transformPoseToList(pose_msg,'/vins_estimator/camera_pose')
        list_exp = [1,0,1,0,1,0]
        np_testing.assert_allclose(list_transformed, list_exp, atol=1e-3)

    def testExtractTopicWithHeader(self):
        pose_msg = self.generatePoseMsg(1)
        first_pose_msg = self.generatePoseMsg(0)
        tStart = first_pose_msg.header.stamp
        rel_tStart = (pose_msg.header.stamp - tStart).to_sec()
        list_transformed = mod.transformPoseToList(first_pose_msg,'/vins_estimator/camera_pose')
        TWH_exp  = np.hstack([rel_tStart, list_transformed])

        bag = rosbag.Bag('test.bag')
        topic_name = '/vins_estimator/camera_pose'

        TWH_transformed = mod.extractTopicWithHeader(bag,topic_name,mod.transformPoseToList,tStart)
        np_testing.assert_allclose(TWH_transformed[1], TWH_exp, atol=1e-3)

if __name__ == "__main__":
    unittest.main()



