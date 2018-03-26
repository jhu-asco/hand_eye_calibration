#!/usr/bin/env python2

import evaluate_vins_accuracy as mod
import unittest
import numpy.testing as np_testing
import numpy as np
import tf.transformations as tf


class TestEvaluateVINSAccuracy(unittest.TestCase):

    def testGetPoseMatrix(self):
        pose_vec = [1, 0, 0, 0.7071068, 0, 0, 0.7071068]
        pose_mat = mod.getPoseMatrix(pose_vec)
        pose_mat_exp = np.eye(4)
        pose_mat_exp[:3, -1] = pose_vec[:3]
        pose_mat_exp[:3, :3] = np.array([[1, 0, 0],
                                         [0, 0, -1],
                                         [0, 1, 0]], dtype=np.float64)
        np_testing.assert_allclose(pose_mat, pose_mat_exp, atol=1e-6)

    def generatePoseData(self, N):
        p1 = np.random.sample((7, N))
        q_norm = np.linalg.norm(p1[3:, :], axis=0)
        p1[3:, :] = p1[3:, :] / q_norm
        return p1

    def testMultiplyIdentity(self):
        p0 = np.array([0, 0, 0, 0, 0, 0, 1])
        p1 = self.generatePoseData(10)
        p1_exp = mod.leftMultiplyPose(p0, p1)
        p1_exp_2 = mod.rightMultiplyPose(p1, p0)
        np_testing.assert_allclose(p1_exp, p1, atol=1e-7)
        np_testing.assert_allclose(p1_exp_2, p1, atol=1e-7)

    def testMultiplyPoses(self):
        p0 = np.array([0, 0, 1, 0, 0, 0.7071068, 0.7071068])  # Z rot 90 deg
        p1 = np.array([[1, 0, 0, 0, 0, -0.7071068, 0.7071068],
                       [0, 1, 0, 0, 0.7071068, 0, 0.7071068]]).T
        # Left Multiply
        pout = mod.leftMultiplyPose(p0, p1)
        np_testing.assert_allclose(pout[:, 0], np.array([0, 1, 1, 0, 0, 0, 1]))
        np_testing.assert_allclose(pout[:, 1],
                                   np.array([-1, 0, 1, -0.5, 0.5, 0.5, 0.5]))
        # Right Multiply
        pout = mod.rightMultiplyPose(p1, p0)
        np_testing.assert_allclose(pout[:, 0], np.array([1, 0, 1, 0, 0, 0, 1]))
        np_testing.assert_allclose(pout[:, 1],
                                   np.array([1, 1, 0, 0.5, 0.5, 0.5, 0.5]))

    def testTransformVINSToMocap(self):
        vins_poses = self.generatePoseData(10)
        p_m_v = np.array([1, 0, 0,   0, 0, 0.7071068, 0.7071068])
        p_i_m = np.array([0, 0, 0,   0, 0, 0, 1])
        x0 = np.hstack((p_m_v, p_i_m))
        mocap_vins_poses = mod.transformVINSToMocap(x0, vins_poses)
        rot_vins_pos = vins_poses[[1, 0, 2], :]
        rot_vins_pos[0, :] = -1 * rot_vins_pos[0, :]
        trans_vins_pos = rot_vins_pos + np.array([[1], [0], [0]])
        vins_pose_rot = tf.quaternion_multiply(p_m_v[3:], vins_poses[3:, :])
        np_testing.assert_allclose(mocap_vins_poses[:3, :],
                                   trans_vins_pos, atol=1e-6)
        np_testing.assert_allclose(mocap_vins_poses[3:, :],
                                   vins_pose_rot, atol=1e-6)

    def testPositionDistance(self):
        mocap_vins_pos = np.random.sample((3, 10))
        distance = np.zeros(10)
        mocap_pos = np.copy(mocap_vins_pos)
        # Check distance keeps increasing
        for i in range(10):
            mocap_pos = mocap_vins_pos + 0.1 * (i + 1)
            p_dist = mod.positionDistanceSq(mocap_vins_pos, mocap_pos)
            np_testing.assert_array_less(distance, p_dist)
            distance = p_dist
        # Commute check
        comm_distance = mod.positionDistanceSq(mocap_pos, mocap_vins_pos)
        np_testing.assert_allclose(distance, comm_distance)

    def testQuatDistance(self):
        q0 = np.array([0, 0, 0, 1])
        axis = np.array([1, 0, 0])
        rot_q = np.empty((4, 25))
        for i in range(5):
            # Randomly perturb axis
            axis = axis + 0.1 * np.random.sample(3)
            axis = tf.unit_vector(axis)
            angle = 0
            for j in range(5):
                angle = angle + 0.1
                rot_q[:, i * 5 + j] = tf.quaternion_about_axis(angle, axis)
        q0_tile = np.tile(q0, (25, 1)).T
        q_dist = mod.quatDistance(q0_tile, rot_q)
        q_dist_r = q_dist.reshape(5, 5)
        # print(np.diff(q_dist_r))
        np_testing.assert_allclose(q_dist_r[:, 0], q_dist[0] * np.ones(5))
        np_testing.assert_array_less(np.zeros((5, 4)), np.diff(q_dist_r))

    def testCost(self):
        vins_poses = self.generatePoseData(10)
        p_m_v = np.array([0, 0, 1, 0, 0, 0.7071068, 0.7071068])  # Z rot 90 deg
        p_i_m = np.array([0, 1, 0, 0, 0.7071068, 0, 0.7071068])  # Y rot 90 deg
        mocap_poses = mod.rightMultiplyPose(
            mod.leftMultiplyPose(p_m_v, vins_poses), p_i_m)
        x0 = np.hstack((p_m_v, p_i_m))
        cost = mod.cost(x0, vins_poses, mocap_poses)
        self.assertAlmostEqual(cost, 0, places=6)
        x0[0] = x0[0] + 1
        cost_2 = mod.cost(x0, vins_poses, mocap_poses)
        self.assertGreater(np.abs(cost_2), 1e-6)

    def testQuatConstraint(self):
        p_m_v = np.array([0, 0, 1, 0, 0, 0.7071068, 0.7071068])
        p_i_m = np.array([0, 1, 0, 0, 0.7071068, 0, 0.7071068])
        x = np.hstack((p_m_v, p_i_m))
        self.assertAlmostEqual(mod.quatConstraint1(x), 0, places=6)
        self.assertAlmostEqual(mod.quatConstraint2(x), 0, places=6)
        x[4] = x[4] + 0.1
        x[13] = x[13] - 0.1
        self.assertGreater(np.abs(mod.quatConstraint1(x)), 1e-6)
        self.assertGreater(np.abs(mod.quatConstraint2(x)), 1e-6)

    def testInterpolatePosesIdentity(self):
        vins_poses = self.generatePoseData(10)
        vins_poses_out = mod.interpolatePoses(np.arange(10), np.arange(10),
                                              vins_poses)
        np_testing.assert_allclose(vins_poses, vins_poses_out)

    def testInterpolateConstantAxis(self):
        N = 20
        vins_poses = np.empty((7, N))
        diff = np.array([1, 2, 3])
        x0 = np.array([0, 0, 0])
        ts_in = np.arange(N)
        ts_out = np.arange(2 * N - 1) * 0.5
        axis = tf.unit_vector(np.random.sample(3))
        angle = 0
        adiff = 0.2
        for i in range(N):
            vins_poses[:3, i] = x0 + i * diff
            vins_poses[3:, i] = np.hstack((np.sin(angle / 2.0) * axis,
                                           np.cos(angle / 2.0)))
            angle = angle + adiff
        interp_poses = mod.interpolatePoses(ts_out, ts_in, vins_poses)
        np_testing.assert_allclose(interp_poses[:, ::2], vins_poses)
        angles_interp = 2 * np.arccos(interp_poses[6, 1::2])
        N = angles_interp.size
        expected_angles = np.arange(N) * adiff + 0.5 * adiff
        np_testing.assert_allclose(angles_interp, expected_angles)

    def testEstimate(self):
        np.random.seed(1012)
        N = 100
        vins_poses = self.generatePoseData(N)
        p_m_v = np.array([0, 0, 1, 0, 0, 0.7071068, 0.7071068])  # Z rot 90 deg
        p_i_m = np.array([0, 1, 1, 0, 0.7071068, 0, 0.7071068])  # Y rot 90 deg
        mocap_poses = mod.rightMultiplyPose(
            mod.leftMultiplyPose(p_m_v, vins_poses), p_i_m)
        ts = np.arange(N)
        x_est = mod.estimate(ts, vins_poses, ts, mocap_poses)
        np.set_printoptions(precision=3, suppress=True)
        np_testing.assert_allclose(x_est[:7], p_m_v, atol=1e-3)
        np_testing.assert_allclose(x_est[7:], p_i_m, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
