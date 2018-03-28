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


def transformPoseToList(msg):
    """
    Transform a pose message into an array of xyz, Euler rpy in ZYX format
    """
    quat = tf.unit_vector(np.array([msg.orientation.x, msg.orientation.y,
                                    msg.orientation.z, msg.orientation.w]))
    yaw, pitch, roll = tf.euler_from_quaternion(quat, 'rzyx')
    return [msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]


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
        data_entry = transform_fcn(msg)
        data_list.append(np.hstack([tdiff, data_entry]))
    data = np.vstack(data_list)
    return data
