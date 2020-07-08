
from utils import *

from two_dimensional_interpolation import getTwoDimensionalInterpolationData, getQuad

import math

def getSimplifiedValues(vr_pos, vr_ori, vr_line, rot_axis):
	theta = rot_axis.getVal(vr_ori)

	unrotated_tvec = rot_axis.getUnrotatedPosition(vr_ori, vr_pos)

	x = vr_line.getVal(unrotated_tvec)

	return x, theta

if __name__ == '__main__':
	# Compute error between data and 2d interpolation
	vr_data_fname = '../data/12-30/vr_data_12-30.txt'
	align_data_fname = '../data/12-30/align_data_12-30.txt'

	interpolation_fname = '../data/12-30/two_dimensional_interpolation_12-30.txt'

	vr_repeats = 100
	low_dist_thresh = 0.5

	vr_data = getVRData(vr_data_fname)
	vr_data = reduceVRData(vr_data, vr_repeats)

	align_data = getVoltData(align_data_fname)

	all_data = [(vr_dp, al_dp) for vr_dp, al_dp in zip(vr_data, align_data) if al_dp.dist >= low_dist_thresh]

	# Get quads
	rot_axis, vr_line, vertexes, quads = getTwoDimensionalInterpolationData(interpolation_fname)

	quads_by_ind = {i: q for i, q in enumerate(quads)}

	for vr_dp, al_dp in all_data:
		x, theta = getSimplifiedValues(vr_dp.tvec, vr_dp.rot_mtx, vr_line, rot_axis)

		# print '%4.4f, %4.4f' % (x, theta * 180.0 / math.pi)

		this_quad = getQuad(x, theta, quads_by_ind)

		if this_quad is None:
			print('SKIPPED')
			continue

		tx1 = lambda v: v.tx1
		tx2 = lambda v: v.tx2
		rx1 = lambda v: v.rx1
		rx2 = lambda v: v.rx2

		int_tx1 = this_quad.interpolate(x, theta, tx1)
		int_tx2 = this_quad.interpolate(x, theta, tx2)
		int_rx1 = this_quad.interpolate(x, theta, rx1)
		int_rx2 = this_quad.interpolate(x, theta, rx2)

		print('----------------------')

		print(int_tx1, int_tx2, int_rx1, int_rx2)
		print(al_dp.tx_gm1, al_dp.tx_gm2, al_dp.rx_gm1, al_dp.rx_gm2)

