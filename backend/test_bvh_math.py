import numpy as np
from scipy.spatial.transform import Rotation as R

from utils import rotation_6d_to_matrix


def test_euler_roundtrip_xyz():
    rng = np.random.default_rng(42)
    mats = R.random(128, random_state=rng).as_matrix()
    eulers = R.from_matrix(mats).as_euler("XYZ", degrees=False)
    mats2 = R.from_euler("XYZ", eulers, degrees=False).as_matrix()
    diff = np.abs(mats - mats2).max()
    assert diff < 1e-5


def test_rot6d_to_euler_consistency():
    rng = np.random.default_rng(7)
    mats = R.random(64, random_state=rng).as_matrix()
    d6 = np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=1)
    mats2 = rotation_6d_to_matrix(d6)
    eulers = R.from_matrix(mats2).as_euler("XYZ", degrees=False)
    mats3 = R.from_euler("XYZ", eulers, degrees=False).as_matrix()
    diff = np.abs(mats - mats3).max()
    assert diff < 1e-5
