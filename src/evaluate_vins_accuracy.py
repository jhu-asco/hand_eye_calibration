#!/usr/bin/env python

import tf.transformations as tf
import numpy as np
import scipy.optimize as opt


def interpolatePoses(ts_out, ts_in, pose_in):
    """
    Map pose inputs from ts_in to ts_out
    Parameters:
        ts_out - Output time stamps
        ts_in - Input time stamps
        pose_in - Array of ts_in size containing poses-[position, quaterion]
                  size (7, N), Quat is stored as (x,y,z,w)
    Return: interpolated poses at ts_out
    """
    pos_out_list = []
    for i in range(3):
        pos_out_list.append(np.interp(ts_out, ts_in, pose_in[:,i]))
    position_out = np.vstack(pos_out_list)
    quat_out_list = []
    quat_in = pose_in[:,3:]
    # Find fraction for interpolating quaternion
    N = ts_in.size
    out = np.interp(ts_out, ts_in, np.arange(N))
    for out_i in out:
        id0 = int(np.floor(out_i))
        id1 = min(N - 1, id0 + 1)
        frac = out_i - id0
        quat_out = tf.quaternion_slerp(quat_in[id0], quat_in[id1], frac)
        quat_out_list.append(quat_out)
    quat_out_arr = np.vstack(quat_out_list).T
    return np.vstack((position_out, quat_out_arr))


def getPoseMatrix(pose):
    """
    Generate pose matrix from quaternion and position
    Parameters
        pose - [position, quaternion] quaternion in xyzw format
    Return:
        Matrix [4x4]
    """
    mat = tf.quaternion_matrix(pose[3:])
    mat[:3, -1] = pose[:3]
    return mat


def leftMultiplyPose(pose0, pose1):
    """
    Multiply pose0 with pose1 assuming pose1 is 7xn matrix. pose0 is 7
    vector.
    Returns: [7xn] pose matrix by multiplying pose0 on the left
    """
    p0 = pose0[:3]
    q0 = pose0[3:]
    q0_c = tf.quaternion_conjugate(q0)
    p1 = pose1[:3, :]
    q1 = pose1[3:,:]
    N = p1.shape[1]
    #print pose1.shape
    p1_exp = np.vstack((p1, np.zeros(N)))
    p1_rot = tf.quaternion_multiply(q0, tf.quaternion_multiply(p1_exp, q0_c))
    q_out = tf.quaternion_multiply(q0, q1)
    p_out = np.expand_dims(p0, axis=1) + p1_rot[:3, :]
    return np.vstack((p_out, q_out))


def rightMultiplyPose(pose0, pose1):
    """
    Multiply pose0 with pose1 assuming pose0 is 7xn matrix and pose1 is 7
    vector.
    Returns: [7xn] pose matrix by multiplying pose0 x pose1
    """
    p0 = pose0[:3, :]
    q0 = pose0[3:, :]
    q0_c = tf.quaternion_conjugate(q0)
    p1 = pose1[:3]
    q1 = pose1[3:]
    p1_exp = np.hstack((p1, 0))
    p1_rot = tf.quaternion_multiply(q0, tf.quaternion_multiply(p1_exp, q0_c))
    q_out = tf.quaternion_multiply(q0, q1)
    p_out = p0 + p1_rot[:3, :]
    return np.vstack((p_out, q_out))


def transformVINSToMocap(x0, vins_poses):
    """
    Transform VINS_origin to IMU to MOCAP to Marker given the poses
    MOCAP to VINS_origin and IMU to Marker.
    Parameters:
        x0 - [14] vector containing two poses (MOCAP - VINS_origin),
                (IMU to Marker)
        vins_poses - [7 x N] Sample poses containing VINS_origin to IMU
    Returns:
        Transformedposes from MOCAP to marker by multiply the given
        x0 poses
    """
    g0 = x0[:7]  # Mocap_O to VINS_O
    g1 = x0[7:]  # IMU to Marker
    mocap_vins = leftMultiplyPose(g0, vins_poses)
    mocap_marker = rightMultiplyPose(mocap_vins, g1)
    return mocap_marker


def positionDistanceSq(pos1, pos2):
    """
    Compute squared distance between two positions using Euclidean distance
    Params:
        pos1 - Vector of positions [3 x N]
        pos2 - Vector of positions [3 x N]
    Returns a distance metric vector [N] where the distance between each
    element is given by np.sum((pos1 - pos0)**2)
    """
    pdiff = pos1 - pos2
    return np.sum(np.square(pdiff), axis=0)


def quatDistance(quat1, quat2):
    """
    Compute distance metric between a vector of quaternions
    Params:
        quat1 - Vector of quaternions[4 x N]
        quat2 - Vector of quaternions[4 x N]
    Returns a distance metric vector [N] where the distance is given
    by (1 - (q0^T q1)**2)
    """
    qdot = np.sum(quat1 * quat2, axis=0)
    return (1.0 - np.square(qdot))


def cost(x0, vins_poses, mocap_poses):
    """
    Determing the distance between vins_poses and mocap_poses. Find the sum of
    mean Least squares distance between positions and mean quaternion distance
    Parameters:
        x0 - [14] vector consisting mocap_origin to vins_origin pose and
             IMU to marker pose
        vins_poses - VINS poses of size [7xN] containing position, quats
        mocap_poses - MOCAP origin to Marker poses of size [7xN]
    Return a scalar containing the mean position error and quaternion distance
    """
    mocap_vins_poses = transformVINSToMocap(x0, vins_poses)
    p_dist = positionDistanceSq(mocap_vins_poses[:3, :], mocap_poses[:3, :])
    q_dist = quatDistance(mocap_vins_poses[3:, :], mocap_poses[3:, :])
    N = mocap_poses.shape[1]
    return (1.0 / N) * np.sum(p_dist + q_dist)


def quatConstraint1(x0):
    """
    Ensure the quaternion sum of squares = 1 for mocap to VINS
    Returns: q0^T q0 - 1
    """
    q0 = x0[3:7]
    return np.sum(q0 * q0) - 1


def quatConstraint2(x0):
    """
    Ensure the quaternion sum of squares = 1 for IMU to Marker
    Returns: q1^T q1 - 1
    """
    q1 = x0[10:]
    return np.sum(q1 * q1) - 1


def estimate(ts_vins, vins_poses, ts_mocap, mocap_poses):
    """
    Given vins poses and mocap poses, interpolate and fit the transformations
    MOCAP_to_VINS_0.
    Parameters:
        ts_vins - Timestamps for vins poses[nv]
        ts_mocap - Timestamps for Mocap poses [nm]
        vins_poses - Vins poses [7xnv]
        mocap_poses - Mocap poses [7xnm]
    Return matrices [1x14] where first 7 elements are Mocap to VINS and second 7
    are IMU to Marker. Pose = [x,y,z,Qx,Qy,Qz,Qw]
    """
    vins_interp = interpolatePoses(ts_mocap, ts_vins, vins_poses)
    cons = ({'type': 'eq', 'fun': quatConstraint1},
            {'type': 'eq', 'fun': quatConstraint2})
    x0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    res = opt.minimize(cost, x0, args=(vins_interp, mocap_poses),
                       constraints=cons, method='SLSQP')
    return res.x
