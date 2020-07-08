
from two_dimensional_interpolation import getTwoDimensionalInterpolationData
from rotation_interpolation import RotationAxis
from utils import *

import math
from scipy.optimize import fsolve

def getComponents(rot_mtx, rot_axis):
	a1 = rot_axis.axis
	a2 = rot_axis.axis.cross(rot_axis.ref_vec).norm()
	a3 = (rot_axis.ref_vec + Vec(0.0, 0.0, 0.0)).norm()

	ra1 = RotationAxis(a1)
	v1 = ra1.getVal(rot_mtx)

	ra2 = RotationAxis(a2)
	v2 = ra2.getVal(rot_mtx)

	ra3 = RotationAxis(a3)
	v3 = ra3.getVal(rot_mtx)

	return v1, v2, v3

def makeMatrixFromComponents(v1, v2, v3, rot_axis):
	m1 = rotMatrix(rot_axis.axis, v1)
	m2 = rotMatrix(rot_axis.axis.cross(rot_axis.ref_vec).norm(), v2)
	m3 = rotMatrix((rot_axis.ref_vec + Vec(0.0, 0.0, 0.0)).norm(), v3)

	return m1 * m2 * m3

if __name__ == '__main__':
	vr_data_fname = '../data/9-19/vr_data_9-19.txt'
	interpolation_data_fname = '../data/9-19/two_dimensional_interpolation_9-19.txt'

	new_vr_data_fname = '../data/9-19/denoised_vr_data_9-19.txt'

	# Get the VR data
	vr_data = getVRData(vr_data_fname)

	# Get rotation axis and vr line
	rot_axis, vr_line = getTwoDimensionalInterpolationData(interpolation_data_fname, just_axis_and_line = True)

	# Find the other rotation components
	v2s = []
	v3s = []
	difs = []
	for dp in vr_data:
		v1, v2, v3 = getComponents(dp.rot_mtx, rot_axis)

		difs.append(dp.rot_mtx.totalDifference(makeMatrixFromComponents(v1, v2, v3, rot_axis)))

		v2s.append(v2)
		v3s.append(v3)

	avg_v2 = sum(v2s) / float(len(v2s))
	avg_v3 = sum(v3s) / float(len(v3s))

	# For each point foind their (x, theta)
	new_data = []
	for dp in vr_data:
		theta = rot_axis.getVal(dp.rot_mtx)
		unrotated_tvec = rot_axis.getUnrotatedPosition(dp.rot_mtx, dp.tvec)
		x = vr_line.getVal(unrotated_tvec)

		unrotated_new_tvec = vr_line.generatePoint(x)

		new_rot_mtx = makeMatrixFromComponents(theta, avg_v2, avg_v3, rot_axis)
		new_tvec = (unrotated_new_tvec - rot_axis.ref_vec) + new_rot_mtx.mult(rot_axis.ref_vec)

		new_quat = new_rot_mtx.toQuat()

		new_dp = VRDataPoint(dp.time, new_tvec, new_rot_mtx, new_quat, dp.tx_vals, dp.rx_vals)
		new_data.append(new_dp)

	outputVRData(new_vr_data_fname, new_data)
