
from utils import *

from interpolation import *
from rotation_interpolation import *

import math
import random

from scipy.optimize import fsolve

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Vertex:
	def __init__(self, x, theta, tx1 = None, tx2 = None, rx1 = None, rx2 = None):
		self.x     = x
		self.theta = theta
		self.tx1, self.tx2, self.rx1, self.rx2 = tx1, tx2, rx1, rx2

	def __str__(self):
		return '(%.4f, %.4f)' % (self.x, self.theta)

def leftside(this_x, this_theta, va, vb):
	cross = (vb.x - va.x) * (this_theta - va.theta) - (vb.theta - va.theta) * (this_x - va.x)
	return cross >= -1.0e-17

class Quad:
	def __init__(self, v1, v2, v3, v4):
		self.v1, self.v2, self.v3, self.v4 = v1, v2, v3, v4

	def check(self, this_x, this_theta):
		return all(leftside(this_x, this_theta, va, vb) for va, vb in ((self.v1, self.v2), (self.v2, self.v3), (self.v3, self.v4), (self.v4, self.v1)))

	interpolation_methods = ['bilinear_horizontal', 'bilinear_vertical', 'inverse_distance']
	def interpolate(self, this_x, this_theta, val_func, quad_interpolation_method = 'bilinear_horizontal'):
		f  = lambda i, j, f: ((f(self.v1) - f(self.v2) + f(self.v3) - f(self.v4)) * i + f(self.v4) - f(self.v1)) * j + (f(self.v2) - f(self.v1)) * i + f(self.v1)
		xf = lambda i, j: f(i, j, lambda v: v.x)
		tf = lambda i, j: f(i, j, lambda v: v.theta)
		zf = lambda i, j: f(i, j, val_func)

		solve_func = lambda vs: (xf(*vs) - this_x, tf(*vs) - this_theta)

		rv = fsolve(solve_func, (0.0, 0.0))
		sp_i, sp_j = rv

		x0 = this_x
		x1 = self.v1.x
		x2 = self.v2.x
		x3 = self.v3.x
		x4 = self.v4.x

		t0 = this_theta
		t1 = self.v1.theta
		t2 = self.v2.theta
		t3 = self.v3.theta
		t4 = self.v4.theta

		p2 = (t2 - t1) * (x3 - x4) - (t3 - t4) * (x2 - x1)
		p1 = (-x1 + x2 - x3 + x4) * t0 + \
				( x0 + x3 - 2.0 * x4) * t1 + \
				(-x0 + x4) * t2 + \
				( x0 - x1) * t3 + \
				(-x0 + 2 * x1 - x2) * t4

		p0 = (x1 - x4) * t0 + (-x0 + x4) * t1 + (x0 - x1) * t4

		pos_i = (-p1 + math.sqrt(pow(p1, 2.0) - 4.0 * p2 * p0)) / (2.0 * p2)
		neg_i = (-p1 - math.sqrt(pow(p1, 2.0) - 4.0 * p2 * p0)) / (2.0 * p2)

		j_from_i = lambda i: ((x1 - x2) * i + x0 - x1) / ((x1 - x2 + x3 - x4) * i + x4 - x1)

		pos_j = j_from_i(pos_i)
		neg_j = j_from_i(neg_i)

		between_0_and_1 = lambda v: v >= 0.0 and v <= 1.0

		if between_0_and_1(neg_i) and between_0_and_1(neg_j):
			if abs(sp_i - neg_i) > 0.0001:
				raise Exception('A')

			if abs(sp_j - neg_j) > 0.0001:
				raise Exception('B')

			return zf(neg_i, neg_j)
		elif between_0_and_1(pos_i) and between_0_and_1(pos_j):
			if abs(sp_i - pos_i) > 0.0001:
				raise Exception('C')

			if abs(sp_j - pos_j) > 0.0001:
				raise Exception('D')

			return zf(pos_i, pos_j)
		else:
			print 'Scipy solution:', (sp_i, sp_j)
			print 'NEG VALS:', neg_i, neg_j
			print 'POS VALS:', pos_i, pos_j


			print 'X vs. F_x(sp_i, sp_j): ', '(%.6f, %.6f)' % (this_x,     xf(sp_i, sp_j))
			print 'T vs. F_t(sp_i, sp_j): ', '(%.6f, %.6f)' % (this_theta, tf(sp_i, sp_j))

			print 'X vs. F_x(neg_i, neg_j): ', '(%.6f, %.6f)' % (this_x,     xf(neg_i, neg_j))
			print 'T vs. F_t(neg_i, neg_j): ', '(%.6f, %.6f)' % (this_theta, tf(neg_i, neg_j))

			print 'X vs. F_x(pos_i, pos_j): ', '(%.6f, %.6f)' % (this_x,     xf(pos_i, pos_j))
			print 'T vs. F_t(pos_i, pos_j): ', '(%.6f, %.6f)' % (this_theta, tf(pos_i, pos_j))

			raise Exception('What?!?!?!?!?')

	def __str__(self):
		return ', '.join(map(str, (self.v1, self.v2, self.v3, self.v4)))

def getQuad(this_x, this_theta, quads = None, quads_by_ind = None):
	if quads_by_ind is not None:
		quads = [quad for _, quad in quads_by_ind.iteritems()]

	for quad in quads:
		if quad.check(this_x, this_theta):
			return quad

	# TODO return the nearest quad
	return None

def getTwoDimensionalInterpolationData(interpolation_data_fname, just_axis_and_line = False):
	f = open(interpolation_data_fname, 'r')

	rot_axis, ref_vec = None, None
	line_point, line_dir = None, None

	vertexes = dict()
	quads = []

	for line in f:
		spl = line.split()

		if spl[0] in ['RA', 'RV', 'VRP', 'VRD']:
			x, y, z = map(float, spl[1:])
			v = Vec(x, y, z)

			if spl[0] == 'RA':
				rot_axis = v
			if spl[0] == 'RV':
				ref_vec = v
			if spl[0] == 'VRP':
				line_point = v
			if spl[0] == 'VRD':
				line_dir = v

		elif spl[0] == 'V':
			v_id = int(spl[1])
			x, theta, tx1, tx2, rx1, rx2 = map(float, spl[2:])

			vertex = Vertex(x, theta, tx1, tx2, rx1, rx2)
			vertexes[v_id] = vertex

		elif spl[0] == 'Q':
			q_id, v1_id, v2_id, v3_id, v4_id = map(int, spl[1:])
			quad = Quad(vertexes[v1_id], vertexes[v2_id], vertexes[v3_id], vertexes[v4_id])

			quads.append(quad)

		else:
			raise Exception('What?')

	rot_axis = RotationAxis(rot_axis, ref_vec)
	vr_line = VRLine(line_point, line_dir)

	if just_axis_and_line:
		return rot_axis, vr_line

	else:
		return rot_axis, vr_line, vertexes, quads

class TwoDimensionalInterpolation:
	def __init__(self, rot_axis, line, vertexes, quads):
		self.rot_axis = rot_axis
		self.line = line
		self.vertexes = vertexes
		self.quads = quads

	def __call__(self, vr_tvec, vr_rmtx):
		this_t = self.rot_axis.getVal(vr_rmtx)
		unrotated_tvec = self.rot_axis.getUnrotatedPosition(vr_rmtx, vr_tvec)
		this_x = self.line.getVal(unrotated_tvec)

		get_t1 = lambda v: v.tx1
		get_t2 = lambda v: v.tx2
		get_r1 = lambda v: v.rx1
		get_r2 = lambda v: v.rx2

		this_quad = getQuad(this_x, this_t, self.quads)

		if this_quad is None:
			this_t1, this_t2, this_r1, this_r2 = 0.0, 0.0, 0.0, 0.0
		else:
			this_t1 = this_quad.interpolate(this_x, this_t, get_t1)
			this_t2 = this_quad.interpolate(this_x, this_t, get_t2)
			this_r1 = this_quad.interpolate(this_x, this_t, get_r1)
			this_r2 = this_quad.interpolate(this_x, this_t, get_r2)

		return map(lambda x: int(round(x)), (this_t1, this_t2, this_r1, this_r2))

if __name__ == '__main__':
	vr_data_fname = '../data/7-8/vr_data_7-8.txt'
	align_data_fname = '../data/7-8/align_data_7-8.txt'

	vr_repeats = 1000
	num_rot_per_lin =  [13, 13, 13, 1, 1, 1, 1, 1, 1, 1, 1] # [13, 13, 13, 13, 1, 1, 1, 1, 1, 1, 1,1]

	low_dist_thresh = 0.5

	out_fname = '../data/7-8/two_dimensional_interpolation_7-8_v2.txt'

	n_x = 3
	n_t = 3

	missing_indexes = []

	# Get VR data, and reduce it
	vr_data = getVRData(vr_data_fname)
	vr_data = reduceVRData(vr_data, vr_repeats)

	if sum(num_rot_per_lin) != len(vr_data):
		print sum(num_rot_per_lin), len(vr_data)
		raise Exception('Mismatch data')

	# Find each rotation axis
	i = 0
	rotation_axises = []
	for n in num_rot_per_lin:
		if n >= 3:
			this_vr_data = vr_data[i : i + n]

			ra = findAxisOfRotation(this_vr_data)

			rotation_axises.append(ra)

		i += n

	# Combine the individual rotation axises by averaging them
	avg_axis    = sum((ra.axis    for ra in rotation_axises), Vec(0.0, 0.0, 0.0)).mult(1.0 / float(len(rotation_axises)))
	avg_ref_vec = sum((ra.ref_vec for ra in rotation_axises), Vec(0.0, 0.0, 0.0)).mult(1.0 / float(len(rotation_axises)))

	print '\n'.join(map(str, [ra.ref_vec for ra in rotation_axises]))

	rotation_axis = RotationAxis(avg_axis, avg_ref_vec)

	# Find the VR Line
	unrotated_tvecs = [rotation_axis.getUnrotatedPosition(dp.rot_mtx, dp.tvec) for dp in vr_data]
	# print '\n'.join(map(str, [dp.tvec for dp in vr_data]))
	vr_line = calcVRLine(unrotated_tvecs)

	# Check angles and dists
	locs = []
	for dp in vr_data:
		theta = rotation_axis.getVal(dp.rot_mtx)
		unrotated_tvec = rotation_axis.getUnrotatedPosition(dp.rot_mtx, dp.tvec)
		x = vr_line.getVal(unrotated_tvec)

		locs.append((x, theta * 180.0 / math.pi))

	print '(x, theta [in degrees]) for all points sorted:'
	print '\n'.join(map(lambda v: '(%0.4f, %0.4f)' % v, sorted(locs)))

	# Get the align data
	align_data = getVoltData(align_data_fname)

	# Filter out data that was collected just to find the rotation axis and line
	old_vr_data, old_align_data = vr_data, align_data
	vr_data, align_data = [], []
	for vr_dp, a_dp in zip(old_vr_data, old_align_data):
		if a_dp.dist >= low_dist_thresh:
			vr_data.append(vr_dp)
			align_data.append(a_dp)

	# Get the VR Vertexes
	vr_values = []
	for dp, gm_vals in zip(vr_data, align_data):
		theta = rotation_axis.getVal(dp.rot_mtx)
		unrotated_tvec = rotation_axis.getUnrotatedPosition(dp.rot_mtx, dp.tvec)
		x = vr_line.getVal(unrotated_tvec)

		vr_values.append(Vertex(x, theta, gm_vals.tx_gm1, gm_vals.tx_gm2, gm_vals.rx_gm1, gm_vals.rx_gm2))

	# Find the points "indexes" based on the x and theta values
	value_by_index = {(i, j): None for i in range(n_x) for j in range(n_t) if (i, j) not in missing_indexes}

	# The list of valid index tuples grouped by x-index
	grouped_inds = {i: [(i, j) for j in range(n_t) if (i, j) not in missing_indexes] for i in range(n_x) if len([(i, j) for j in range(n_t) if (i, j) not in missing_indexes]) > 0}

	# Sort the values by their x-value
	sorted_vr_values = sorted(vr_values, key = lambda x: (x.x, x.theta))

	# The first num_x[0] values in sorted_vr_values should have an x_index of 0. The next num_x[1] shoudl have an x_index of 1, etc...
	cur_vals = []
	cur_ind = 0
	sorted_valid_x_inds = sorted(grouped_inds.keys())
	for v in sorted_vr_values:
		cur_vals.append(v)

		if len(cur_vals) >= len(grouped_inds[sorted_valid_x_inds[cur_ind]]):
			cur_vals.sort(key = lambda x: (x.theta, x.x))

			for v, (i, j) in zip(cur_vals, grouped_inds[sorted_valid_x_inds[cur_ind]]):

				value_by_index[(i, j)] = v

			cur_vals = []
			cur_ind += 1

	# TODO: Turn vertices into quads, make test to check if new point is in quad, do interpolation for arbitrary point in the quad
	quads_by_ind = dict()
	corners = [(0, 0), (1, 0), (1, 1), (0, 1)]

	for i in range(n_x - 1):
		for j in range(n_t - 1):
			this_corners = [(a + i, b + j) for a, b in corners]
			if any((x, y) not in value_by_index for x, y in this_corners):
				continue

			this_quad = Quad(*[value_by_index[k] for k in this_corners])
			quads_by_ind[(i, j)] = this_quad

	# TODO Write the values out to a file
	if out_fname is not None and out_fname != '':
		out_f = open(out_fname, 'w')

		# rotation_axis
		out_f.write(' '.join(map(str, ['RA', rotation_axis.axis.x,    rotation_axis.axis.y,    rotation_axis.axis.z])) + '\n')
		out_f.write(' '.join(map(str, ['RV', rotation_axis.ref_vec.x, rotation_axis.ref_vec.y, rotation_axis.ref_vec.z])) + '\n')

		# vr_line
		out_f.write(' '.join(map(str, ['VRP', vr_line.p.x, vr_line.p.y, vr_line.p.z])) + '\n')
		out_f.write(' '.join(map(str, ['VRD', vr_line.d.x, vr_line.d.y, vr_line.d.z])) + '\n')

		# vertexes
		next_id = 1
		for _, vertex in sorted(value_by_index.iteritems()):
			vertex.id = next_id
			next_id += 1

			out_f.write(' '.join(map(str, ['V', vertex.id, vertex.x, vertex.theta, vertex.tx1, vertex.tx2, vertex.rx1, vertex.rx2])) + '\n')

		# quads
		next_id = 1
		for _, quad in sorted(quads_by_ind.iteritems()):
			out_f.write(' '.join(map(str, ['Q', next_id, quad.v1.id, quad.v2.id, quad.v3.id, quad.v4.id])) + '\n')
			next_id += 1


	# Plot
	plot_interpolation = False
	if plot_interpolation:
		min_x, max_x = min(v.x     for v in vr_values), max(v.x     for v in vr_values)
		min_t, max_t = min(v.theta for v in vr_values), max(v.theta for v in vr_values)

		n_points = 100

		get_z = lambda v: v.rx1
		X, Y, Z = [], [], []

		for ix in range(n_points):
			X.append([])
			Y.append([])
			Z.append([])

			this_x = (max_x - min_x) / (n_points - 1.0) * ix + min_x

			for it in range(n_points):
				this_t = (max_t - min_t) / (n_points - 1.0) * it + min_t

				quad = getQuad(this_x, this_t, quads_by_ind)

				if quad is None:
					this_z = 0.0
				else:
					this_z = quad.interpolate(this_x, this_t, get_z)

				X[ix].append(this_x)
				Y[ix].append(this_t)
				Z[ix].append(this_z)

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		ax.plot_surface(np.array(X), np.array(Y), np.array(Z), rstride = 1, cstride = 1, cmap = cm.coolwarm,
		                       linewidth = 0, antialiased = False)

		ax.set_xlabel('Translation')
		ax.set_ylabel('Rotation')
		ax.set_zlabel('GM Val')

		plt.show()
