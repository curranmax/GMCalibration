
from utils import *

from collections import defaultdict

from scipy.optimize import least_squares
import scipy.interpolate

import matplotlib.pyplot as plt

def calcVRLine(vr_points, vr_ori = None):
	def lin_func(x):
		point     = Vec(*x[0:3])
		direction = vecFromAngle(*x[3:5])

		errs = [distanceToLine(point, direction, vr_point)[0] for vr_point in vr_points]

		return errs


	init_point  = [vr_points[0].x, vr_points[0].y, vr_points[0].z]
	# init_angles = list((vr_points[-1] - vr_points[0]).norm().getAngles())

	init_angles = list(Vec(0.0, 0.0, 1.0).norm().getAngles())

	rv = least_squares(lin_func, init_point + init_angles)

	return VRLine(Vec(*rv.x[0:3]), vecFromAngle(*rv.x[3:5]), rm = vr_ori)

class VRLine:
	def __init__(self, p, d, rm = None):
		self.p = p
		self.d = d

		# Just for checking
		self.rm = rm

	def getVal(self, new_pos):
		a = new_pos - self.p

		return a.dot(self.d)

	def generatePoint(self, v):
		return self.p + self.d.mult(v)

class NearestNeighbor:
	def __init__(self, xvs, yvs):
		self.xvs = xvs
		self.yvs = yvs

	def __call__(self, x):
		best_index, _ = min(enumerate([abs(x - xv) for xv in self.xvs]), key = lambda a: a[1])
		return self.yvs[best_index]

class Linear:
	def __init__(self, xvs, yvs):
		self.m = (yvs[-1] - yvs[0]) / (xvs[-1] - xvs[0])
		self.b = yvs[0] - self.m * xvs[0]

		def linear_err(vs):
			this_m, this_b = vs

			errs = []
			for x, y in zip(xvs, yvs):
				approx_y = this_m * x + this_b

				errs.append(y - approx_y)

			return errs

		rv = least_squares(linear_err, (self.m, self.b), **{'xtol': 2.3e-16, 'ftol': 2.3e-16, 'gtol': 2.3e-16, 'max_nfev': 1e5})

		self.m = rv.x[0]
		self.b = rv.x[1]

	def __call__(self, x):
		return self.m * x + self.b

class LinearPieceWise:
	def __init__(self, xvs, yvs):
		self.all_vals = sorted((x, y) for x, y in zip(xvs, yvs))

	def __call__(self, x):
		for i in range(len(self.all_vals) - 1):
			lx, ly = self.all_vals[i]
			hx, hy = self.all_vals[i + 1]

			if (x >= lx and x <= hx) or (i == 0 and x <= lx) or (i == len(self.all_vals) - 1 and x >= hx):
				m = (hy - ly) / (hx - lx)
				b = ly - m * lx

				return m * x + b

		raise Exception('Invalid value outside of bounds: ' + str(x))

	def outputString(self):
		return 'LinearPieceWise ' + str(len(self.all_vals)) + ' ' + ' '.join(map(lambda x: ' '.join(map(str, x)), self.all_vals))

class CubicSpline:
	def __init__(self, xvs, yvs):
		# scipy_coefs[k, i] is the coefficient for the spline between xvs[i] and xvs[i + 1] for the term with power (3 - k)
		scipy_coefs = scipy.interpolate.CubicSpline(xvs, yvs, bc_type = 'not-a-knot').c

		# coefs[i][k] is the coefficient for the spline between xvs[i] and xvs[i + 1] for the term with power k
		self.coefs = [[scipy_coefs[3 - r, i] for r in range(4)] for i in range(len(xvs) - 1)]
		self.xvs  = xvs

	def __call__(self, x):
		for i in range(len(self.coefs)):
			lx, hx = self.xvs[i], self.xvs[i + 1]
			if (x >= lx and x <= hx) or (i == 0 and x <= lx) or (i == len(self.xvs) - 1 and x >= hx):
				return sum(self.coefs[i][k] * pow(x - lx, float(k)) for k in range(4))

		raise Exception('"x" is out of range: ' + str(x))

	def outputString(self):
		return 'CubicSpline ' + str(len(self.coefs)) + ' ' + ' '.join(map(lambda x: ' '.join(map(str, [x[1]] + x[0])), zip(self.coefs, self.xvs[:-1]))) + ' ' + str(self.xvs[-1])

def plotInterpolation(func_by_name, min_x, max_x, num_x):
	xs = []
	ys_by_name = defaultdict(list)
	for i in range(num_x):
		this_x = float(max_x - min_x) / float(num_x - 1) * float(i) + min_x
		xs.append(this_x)

		for name, func in func_by_name.iteritems():
			this_y = func(this_x)
			ys_by_name[name].append(this_y)


	for name, ys in ys_by_name.iteritems():
		plt.plot(xs, ys, label = name)

	plt.legend()

	plt.show()

def getInterpolationFuncFromFile(filename):
	f = open(filename, 'r')

	vr_p = None
	vr_d = None

	vr_rm = None

	tx1_func, tx2_func, rx1_func, rx2_func = None, None, None, None 

	for line in f:
		spl = line.split()

		if spl[0] == 'VRP':
			if vr_p is not None:
				raise Exception('Multiple vr points')

			vr_p = Vec(*map(float, spl[1:]))

		if spl[0] == 'VRD':
			if vr_d is not None:
				raise Exception('Multiple vr dirs')

			vr_d = Vec(*map(float, spl[1:]))

		if spl[0] == 'VRO':
			if vr_rm is not None:
				raise Exception('Multiple VR RMs')

			vals = map(float, spl[1:])
			vr_rm = Matrix(*vals)

		if spl[0] in ['TX1', 'TX2', 'RX1', 'RX2']:
			if spl[1] == 'LinearPieceWise':
				num = int(spl[2])
				xs = map(float, spl[3::2])
				ys = map(float, spl[4::2])

				this_func = LinearPieceWise(xs, ys)

			if spl[1] == 'CubicSpline':
				raise Exception('Not implemented yet')

			if spl[0] == 'TX1':
				if tx1_func is not None:
					raise Exception('Multiple TX1')

				tx1_func = this_func

			if spl[0] == 'TX2':
				if tx2_func is not None:
					raise Exception('Multiple TX2')

				tx2_func = this_func

			if spl[0] == 'RX1':
				if rx1_func is not None:
					raise Exception('Multiple RX1')

				rx1_func = this_func

			if spl[0] == 'RX2':
				if rx2_func is not None:
					raise Exception('Multiple RX2')

				rx2_func = this_func

	return VRLine(vr_p, vr_d, rm = vr_rm), tx1_func, tx2_func, rx1_func, rx2_func

if __name__ == '__main__':
	vr_data_fname = '../data/8-21/vr_data_8-21.txt'
	vr_repeats = 100

	align_data_fname = '../data/8-21/align_data_8-21.txt'
	
	func_type = CubicSpline
	output_fname = '../data/8-21/cubic_interpolation_8-21.txt'

	vr_data = getVRData(vr_data_fname)

	if len(vr_data) % vr_repeats != 0:
		raise Exception('AAAAA')

	vr_points = []
	for i in xrange(0, len(vr_data), vr_repeats):
		this_point_data = [dp.tvec for dp in vr_data[i:i + vr_repeats]]
		sum_point = sum(this_point_data, Vec(0.0, 0.0, 0.0))
		avg_point = sum_point.mult(1.0 / float(vr_repeats))

		vr_points.append(avg_point)

	vr_line = calcVRLine(vr_points, vr_data[0].rot_mtx)

	vr_vals = [vr_line.getVal(vrp) for vrp in vr_points]
	print 'X vals:      ', vr_vals
	print 'Dist to line:', [vrp.dist(vr_line.generatePoint(vr_line.getVal(vrp))) * 1000.0 for vrp in vr_points]
	align_data = getVoltData(align_data_fname)

	print 'Calculating interpolation function, using method:', func_type
	tx1_func = func_type(vr_vals, [dp.tx_gm1 for dp in align_data])
	tx2_func = func_type(vr_vals, [dp.tx_gm2 for dp in align_data])
	rx1_func = func_type(vr_vals, [dp.rx_gm1 for dp in align_data])
	rx2_func = func_type(vr_vals, [dp.rx_gm2 for dp in align_data])

	# plotInterpolation({'lin': LinearPieceWise(vr_vals, [dp.rx_gm1 for dp in align_data]), 'cub': CubicSpline(vr_vals, [dp.rx_gm1 for dp in align_data])}, min(vr_vals), max(vr_vals), 1000)

	print 'Outputting interpolation values'
	out_f = open(output_fname, 'w')

	out_f.write(' '.join(map(str, ['VRP', vr_line.p.x, vr_line.p.y, vr_line.p.z])) + '\n')
	out_f.write(' '.join(map(str, ['VRD', vr_line.d.x, vr_line.d.y, vr_line.d.z])) + '\n')

	out_f.write(' '.join(map(str, ['VRO', vr_line.rm.vals[0][0], vr_line.rm.vals[0][1], vr_line.rm.vals[0][2], vr_line.rm.vals[1][0], vr_line.rm.vals[1][1], vr_line.rm.vals[1][2], vr_line.rm.vals[2][0], vr_line.rm.vals[2][1], vr_line.rm.vals[2][2]])) + '\n')

	out_f.write('TX1 ' + tx1_func.outputString() + '\n')
	out_f.write('TX2 ' + tx2_func.outputString() + '\n')
	out_f.write('RX1 ' + rx1_func.outputString() + '\n')
	out_f.write('RX2 ' + rx2_func.outputString() + '\n')
