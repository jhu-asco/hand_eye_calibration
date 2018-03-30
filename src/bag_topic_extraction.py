#!/usr/bin/env python
"""

Helper functions to extract topics from bag file into an array
@author: gowtham
"""

import numpy as np
import rosbag
import rospy
import tf.transformations as tf


def getRelativeTimeStamp(bag, dt):
    """
    Return the time stamp wrt to start of the bag i.e
    TimeStamp(bag.start_time + dt)
    """
    return rospy.Time.from_sec(bag.get_start_time() + dt)


def transformPoseToList(msg,topic_name):
    """
    Transform a pose message into an array of xyz, Euler rpy in ZYX format
    """
    if topic_name == '/vins_estimator/camera_pose':
        quat = tf.unit_vector(np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                        msg.pose.orientation.z, msg.pose.orientation.w]))
        yaw, pitch, roll = tf.euler_from_quaternion(quat, 'rzyx')

        return [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                roll, pitch, yaw]
    elif topic_name == '/vrpn_client/vins/pose':
        quat = tf.unit_vector(np.array([msg.transform.rotation.x, msg.transform.rotation.y,
                                        msg.transform.rotation.z, msg.transform.rotation.w]))
        yaw, pitch, roll = tf.euler_from_quaternion(quat, 'rzyx')

        return [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z,
            roll, pitch, yaw]

def transformPoseToList_quat(msg,topic_name):
    """
    Transform a pose message into an array of xyz, quaternion in xyzw format
    """
    if topic_name == '/vins_estimator/camera_pose':
        quat = tf.unit_vector(np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                        msg.pose.orientation.z, msg.pose.orientation.w]))
        return [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                quat[0], quat[1], quat[2], quat[3]]

    elif topic_name == '/vrpn_client/vins/pose':
        quat = tf.unit_vector(np.array([msg.transform.rotation.x, msg.transform.rotation.y,
                                        msg.transform.rotation.z, msg.transform.rotation.w]))

        return [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z,
                quat[0], quat[1], quat[2], quat[3]]

def extractTopicWithHeader(bag, topic_name, transform_fcn,
                           tStart=None, max_dt=None):
    """
    Extract the topic in topic_name into a numpy array. The transform_fcn
    maps a message into a list. The output array is of size [N x (tsize + 1)]
    where N is the number of messages and tsize the list size returned by
    transform_fcn. The first column of the array is the relative timestamp
    wrt tStart. The messages before tStart and after (tStart + max_dt) are
    ignored. If tStart is not given, will use the start time of the bag.
    if max_dt is not provided, all the messages in the bag file are used.
    Parameters:
         bag - Opened rosbag object
         topic_name - Name of the topic to extract
         transform_fcn - Function that maps a message into a list
         tStart - ros Timestamp from where the message recording starts
         max_dt - relative tStart when to stop recording messages
    Returns np array[N x (tsize +1)] of data in an array format
    """
    if tStart is None:
        tStart = rospy.Time.from_sec(bag.get_start_time())
    data_list = []
    for _, msg, _ in bag.read_messages([topic_name]):
        tdiff = (msg.header.stamp - tStart).to_sec()
        if max_dt is not None and tdiff > max_dt:
            break
        elif tdiff < 0:
            continue
        data_entry = transform_fcn(msg,topic_name)
        data_list.append(np.hstack([tdiff, data_entry]))
    data = np.vstack(data_list)
    return data
