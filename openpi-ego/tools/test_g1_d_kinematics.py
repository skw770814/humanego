"""Numerical tests for the dependency-light G1-D forward kinematics."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

from tools.g1_d_kinematics import LEFT_ARM_JOINTS
from tools.g1_d_kinematics import LEFT_EEF_LINK
from tools.g1_d_kinematics import REFERENCE_LINK
from tools.g1_d_kinematics import RIGHT_ARM_JOINTS
from tools.g1_d_kinematics import RIGHT_EEF_LINK
from tools.g1_d_kinematics import G1DForwardKinematics

URDF = Path(os.environ.get("G1_D_URDF", "/home/zh/openpi/g1_d_description/g1_d.urdf"))


@unittest.skipUnless(URDF.is_file(), f"G1-D URDF not found: {URDF}")
class G1DForwardKinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = G1DForwardKinematics(URDF)

    def test_output_shape_dtype_and_grouped_rotation(self) -> None:
        samples = np.zeros((3, 14), dtype=np.float64)
        samples[1] = np.linspace(-0.4, 0.4, 14)
        samples[2] = np.linspace(0.3, -0.3, 14)
        poses = self.kinematics.forward(samples)

        assert poses.shape == (3, 18)
        assert poses.dtype == np.float32
        np.testing.assert_array_equal(self.kinematics.forward(samples[0]), poses[0])
        for offset in (3, 12):
            first_column = poses[:, offset : offset + 3]
            second_column = poses[:, offset + 3 : offset + 6]
            np.testing.assert_allclose(np.linalg.norm(first_column, axis=1), 1.0, atol=2e-7)
            np.testing.assert_allclose(np.linalg.norm(second_column, axis=1), 1.0, atol=2e-7)
            np.testing.assert_allclose(np.sum(first_column * second_column, axis=1), 0.0, atol=2e-7)

    def test_rejects_invalid_input(self) -> None:
        for values in (
            np.zeros(13),
            np.zeros((1, 2, 14)),
            np.empty((0, 14)),
            np.full(14, np.nan),
            np.full(14, np.inf),
        ):
            with self.subTest(shape=values.shape):
                try:
                    self.kinematics.forward(values)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Input {values.shape} should have been rejected")

    @unittest.skipUnless(importlib.util.find_spec("pinocchio"), "Pinocchio is not installed")
    def test_matches_pinocchio(self) -> None:
        import pinocchio as pin

        model = pin.buildModelFromUrdf(str(URDF))
        data = model.createData()
        rng = np.random.default_rng(7)
        samples = np.concatenate((np.zeros((1, 14)), rng.uniform(-0.8, 0.8, size=(31, 14))), axis=0)
        actual = self.kinematics.forward(samples)
        expected = []
        joint_names = (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)
        reference_frame = model.getFrameId(REFERENCE_LINK)
        eef_frames = (model.getFrameId(LEFT_EEF_LINK), model.getFrameId(RIGHT_EEF_LINK))

        for sample in samples:
            configuration = pin.neutral(model)
            for value, joint_name in zip(sample, joint_names, strict=True):
                joint_id = model.getJointId(joint_name)
                configuration[model.idx_qs[joint_id]] = value
            pin.forwardKinematics(model, data, configuration)
            pin.updateFramePlacements(model, data)
            row = []
            for eef_frame in eef_frames:
                reference_to_eef = data.oMf[reference_frame].inverse() * data.oMf[eef_frame]
                row.extend(reference_to_eef.translation)
                row.extend(reference_to_eef.rotation[:, 0])
                row.extend(reference_to_eef.rotation[:, 1])
            expected.append(row)

        np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-6, atol=1e-7)

    @unittest.skipUnless(importlib.util.find_spec("pinocchio"), "Pinocchio is not installed")
    def test_matches_real_robot_inference_fk(self) -> None:
        from examples.unitree_inference.g1_kinematics import G1Kinematics

        rng = np.random.default_rng(17)
        samples = np.concatenate((np.zeros((1, 14)), rng.uniform(-0.6, 0.6, size=(31, 14))), axis=0)
        converted = self.kinematics.forward(samples)
        inference = G1Kinematics(URDF, rotation_format="columns_grouped")
        inferred = np.stack([inference.forward(sample) for sample in samples])

        np.testing.assert_allclose(converted, inferred, rtol=0.0, atol=1e-7)
        assert inference.reference_frame == REFERENCE_LINK
        assert inference.eef_frames == (LEFT_EEF_LINK, RIGHT_EEF_LINK)


if __name__ == "__main__":
    unittest.main()
