
from utils import *
from two_dimensional_interpolation import *

import math

def clampPointToAllQuads(this_x, this_t, quads):
	clamped_point = None
	clamped_dist  = None

	for quad in quads:
		c_x, c_t  = clampPointToQuad(this_x, this_t, quad)
		this_dist = math.sqrt(pow(c_x - this_x, 2.0) + pow(c_t - this_t, 2.0))

		if clamped_dist is None or this_dist < clamped_dist:
			clamped_point = (c_x, c_t)
			clamped_dist  = this_dist

	return clamped_point

def clampPointToQuad(this_x, this_t, quad):
	clamped_point = None
	clamped_dist  = None

	for v0, v1 in [(quad.v1, quad.v2), (quad.v2, quad.v3), (quad.v3, quad.v4), (quad.v4, quad.v1)]:
		c_x, c_t  = clampPointToLineSegment(this_x, this_t, v0, v1)
		this_dist = math.sqrt(pow(c_x - this_x, 2.0) + pow(c_t - this_t, 2.0))

		if clamped_dist is None or this_dist < clamped_dist:
			clamped_point = (c_x, c_t)
			clamped_dist  = this_dist

	return clamped_point

def clampPointToLineSegment(this_x, this_t, v0, v1):
	k = (v0.x - v1.x) * (v0.x - this_x) + (v0.theta - v1.theta) * (v0.theta - this_t) / (pow(v1.x - v0.x, 2.0) + pow(v1.theta - v0.theta, 2.0))

	# Clamp k to [0,1]
	if k > 1.0:
		k = 1.0

	if k < 0.0:
		k = 0.0

	c_x = (v1.x     - v0.x)     * k + v0.x
	c_t = (v1.theta - v0.theta) * k + v0.theta

	return c_x, c_t

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
	vr_data_fname       = '../data/1-4/vr_data_1-4.txt'
	interpolation_fname = '../data/1-4/two_dimensional_interpolation_1-4.txt'

	vr_data = getVRData(vr_data_fname)

	rot_axis, vr_line, vertexes, quads = getTwoDimensionalInterpolationData(interpolation_fname)

	x_min = min(v.x for _, v in vertexes.items())
	x_max = max(v.x for _, v in vertexes.items())

	t_min = min(v.theta for _, v in vertexes.items())
	t_max = max(v.theta for _, v in vertexes.items())

	n_x = 100
	n_t = 100

	num_pos = 0
	num_neg = 0

	num_fixed = 0
	num_double_neg = 0

	gen_points = []

	for i in range(n_x):
		this_x = (x_max - x_min) / (n_x - 1) * i + x_min

		for j in range(n_t):
			this_t = (t_max - t_min) / (n_t - 1) * j + t_min

			this_quad = getQuad(this_x, this_t, quads)

			if this_quad is None:
				num_neg += 1

				this_x, this_t = clampPointToAllQuads(this_x, this_t, quads)
				
				this_quad = getQuad(this_x, this_t, quads)

				if this_quad is None:
					raise Exception('Clamped bomb is not in any quad')

			gen_points.append((this_x, this_t, this_quad))

	v2s = []
	v3s = []
	for dp in vr_data:
		v1, v2, v3 = getComponents(dp.rot_mtx, rot_axis)

		v2s.append(v2)
		v3s.append(v3)

	print(min(v2s), max(v2s))
	print(min(v3s), max(v3s))

	avg_v2 = sum(v2s) / float(len(v2s))
	avg_v3 = sum(v3s) / float(len(v3s))

	for x, t, quad in gen_points:
		unrotated_vr_pos = vr_line.generatePoint(x)

		rm = rotMatrix(rot_axis.axis, t)

		rot_ref_vec = rm.mult(rot_axis.ref_vec)

		gen_vr_position = unrotated_vr_pos - rot_axis.ref_vec + rot_ref_vec

		gen_vr_rot_mtx = makeMatrixFromComponents(t, avg_v2, avg_v3, rot_axis)

		# print gen_vr_position
		# print gen_vr_rot_mtx