from g1_kinematics import matrix_to_rot6d
from g1_kinematics import rot6d_to_matrix
import numpy as np


def test_grouped_column_rot6d_round_trip():
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    encoded = matrix_to_rot6d(rotation, "columns_grouped")
    np.testing.assert_array_equal(encoded, [0, 1, 0, -1, 0, 0])
    np.testing.assert_allclose(rot6d_to_matrix(encoded, "columns_grouped"), rotation, atol=1e-8)
