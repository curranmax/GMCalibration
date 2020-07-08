
from utils import *

from two_dimensional_interpolation import *

import random
from datetime import datetime
import sys

from scipy.optimize import minimize

def randomFloat(v1, v2):
	min_val = min(v1, v2)
	max_val = max(v1, v2)

	return random.random() * (max_val - min_val) + min_val

class BruteForceTracking:
	def __init__(self, step):
		self.step = step

	def __call__(self, gm, target_point):
		best_dist = None

		best_gm1_val = None
		best_gm2_val = None
		for gm1_val in xrange(0, pow(2, 16), self.step):
			for gm2_val in xrange(0, pow(2, 16), self.step):
				p, d = gm.getOutput(gm1_val, gm2_val)
				dist, neg = distanceToLine(p, d, target_point)

				if not neg and (best_dist is None or dist < best_dist):
					best_dist = dist

					best_gm1_val = gm1_val
					best_gm2_val = gm2_val

		return best_gm1_val, best_gm2_val

class SearchTracking:
	def __init__(self, n_iter):
		self.n_iter = n_iter

	def __call__(self, gm, target_point):
		cur_gmh, cur_gmv = pow(2, 15), pow(2, 15)

		for i in range(self.n_iter):
			pc, dc = gm.getOutput(cur_gmh, cur_gmv)
			cpc = findClosestPoint(pc, dc, target_point)

			ph, dh = gm.getOutput(cur_gmh + 1, cur_gmv)
			cph = findClosestPoint(ph, dh, target_point)

			pv, dv = gm.getOutput(cur_gmh, cur_gmv + 1)
			cpv = findClosestPoint(pv, dv, target_point)

			uh = cph - cpc
			uv = cpv - cpc

			a, b = findComponentstoMinimizeDistance(uh, uv, cpc, target_point)

			a_int = int(round(a))
			b_int = int(round(b))

			if a_int == 0 and b_int == 0:
				break

			cur_gmh += a_int
			cur_gmv += b_int
		# print 'ITERS:', i
		return cur_gmh, cur_gmv

class ScipyTracking:
	def __init__(self, rt = 'int'):
		if rt not in ['int', 'float']:
			raise Exception('Invalid return type: ' + rt)

		self.rt = rt

	def __call__(self, gm, target_point):
		def func(x):
			gm1_val, gm2_val = x
			p, d = gm.getOutput(gm1_val, gm2_val)
			d, neg = distanceToLine(p, d, target_point)

			if neg:
				return float('inf')
			return d

		res = minimize(func, (pow(2, 15), pow(2, 15)))

		gm1_val, gm2_val = res.x

		if self.rt == 'float':
			return gm1_val, gm2_val
		elif self.rt == 'int':
			gm1_val, gm2_val = map(lambda x: int(round(x)), [gm1_val, gm2_val])
			return gm1_val, gm2_val

# Find a and b such that the distance between p + a*da + b*db and t 
def findComponentstoMinimizeDistance(da, db, p, t):
	v1 = db - da.mult(da.dot(db) / pow(da.mag(), 2.0))


	b = -(v1).dot(p - t + da.mult(da.dot(t - p) / pow(da.mag(), 2.0))) / pow(v1.mag(), 2.0)
	a = da.dot(t - (db.mult(b) + p)) / pow(da.mag(), 2.0)

	return a, b

# Find the value k such that the distance between (t) and (p_0 + d * k) is minimized
def findK(p_0, d, t):
	k = d.dot(t - p_0) / pow(d.mag(), 2.0)
	return k

def findClosestPoint(p_0, d, t):
	k = findK(p_0, d, t)
	return p_0 + d.mult(k)

# Returns the distance between (t) and (p_0 + d * k), and if k is negative.
def distanceToLine(p_0, d, t):
	k = findK(p_0, d, t)
	return (p_0 + d.mult(k)).dist(t), k < 0.0

# tracking must be a function that takes a GM and a Vec and returns two GM values. The goal should be that beam is close as possible to the target point.
def testTracking(gm, num_points = 1, tracking = BruteForceTracking(pow(2, 7))):
	x_range = (-10.0, 10.0)
	y_range = (-10.0, 10.0)
	z_range = (-10.0, 10.0)

	durs = []
	errors = []
	for i in range(num_points):
		# print 'Current progress', str(float(i + 1) / float(num_points) * 100.0) + '%                       \r',
		# sys.stdout.flush()

		# Generate random point
		# target = Vec(randomFloat(*x_range), randomFloat(*y_range), randomFloat(*z_range))

		target = Vec(0.307, 0.709, -1.127)

		start = datetime.now()
		gm1_val, gm2_val = tracking(gm, target)
		end = datetime.now()

		durs.append((end - start).total_seconds())

		p, d = gm.getOutput(gm1_val, gm2_val)

		dist, neg = distanceToLine(p, d, target)

		if neg:
			raise Exception('Point is behind GM')

		errors.append(dist)
	print ''
	print 'Average error:', sum(errors) / float(len(errors))
	print 'Maximum error:', max(errors)
	print ''
	print 'Average duration:', sum(durs) / float(len(durs))
	print 'Maximum duration:', max(durs)

def testTrackingAgainstDots(gm, dots, tracking = BruteForceTracking(pow(2, 10))):
	durs = []
	dist_errors = []
	gm_errors = []
	for dot in dots:
		start = datetime.now()
		gm1_val, gm2_val = tracking(gm, dot.loc)
		end = datetime.now()

		durs.append((end - start).total_seconds())

		gm_error = math.sqrt(pow(gm1_val - dot.gmh, 2.0) + pow(gm2_val - dot.gmv, 2.0))
		gm_errors.append(gm_error)

		p, d = gm.getOutput(gm1_val, gm2_val)
		dist, neg = distanceToLine(p, d, dot.loc)
		if neg:
			raise Exception('Point is behind GM')

		dist_errors.append(dist)

	print ''
	print 'Average gm_error:', sum(gm_errors) / float(len(gm_errors))
	print 'Maximum gm_error:', max(gm_errors)
	print ''
	print 'Average dist_error:', sum(dist_errors) / float(len(dist_errors))
	print 'Maximum dist_error:', max(dist_errors)
	print ''
	print 'Average duration:', sum(durs) / float(len(durs))
	print 'Maximum duration:', max(durs)

class LinkSearchTracking:
	def __init__(self, n_iters, inner_tracking):
		self.n_iters = n_iters
		self.inner_tracking = inner_tracking

	def __call__(self, tx_gm, rx_gm):
		tx_gmh, tx_gmv = pow(2, 15), pow(2, 15)
		rx_gmh, rx_gmv = pow(2, 15), pow(2, 15)

		for i in range(self.n_iters):
			tx_point, _ = tx_gm.getOutput(tx_gmh, tx_gmv)
			rx_point, _ = rx_gm.getOutput(rx_gmh, rx_gmv)
			
			prev_tx_gmh, prev_tx_gmv = tx_gmh, tx_gmv
			prev_rx_gmh, prev_rx_gmv = rx_gmh, rx_gmv

			tx_gmh, tx_gmv = self.inner_tracking(tx_gm, rx_point)
			rx_gmh, rx_gmv = self.inner_tracking(rx_gm, tx_point)

			# These values should be extremely small
			# post_tx_point, post_tx_dir = tx_gm.getOutput(tx_gmh, tx_gmv)
			# print distanceToLine(post_tx_point, post_tx_dir, rx_point)

			# post_rx_point, post_rx_dir = rx_gm.getOutput(rx_gmh, rx_gmv)
			# print distanceToLine(post_rx_point, post_rx_dir, tx_point)

			if all(map(lambda x: abs(x[0] - x[1]) < 1.0, [(prev_tx_gmh, tx_gmh), (prev_tx_gmv, tx_gmv), (prev_rx_gmh, rx_gmh), (prev_rx_gmv, rx_gmv)])):
				break

		return tx_gmh, tx_gmv, rx_gmh, rx_gmv, i

def fullLinkTracking(tx_gm, rx_rel_gm, num_points = 1, real_points = None, tracking = LinkSearchTracking(1000, SearchTracking(1000)), interpolation = None):
	x_range = (-0.5, 0.5)
	y_range = (-0.5, 0.5)
	z_range = (1.0, 3.0)

	qw_range = (1.0, 1.0)
	qx_range = (0.0, 0.0)
	qy_range = (0.0, 0.0)
	qz_range = (0.0, 0.0)

	tx_errs = []
	rx_errs = []

	durs = []

	if num_points is not None and real_points is None:
		pass
	elif num_points is None and real_points is not None:
		num_points = len(real_points)
	else:
		raise Exception('Must specify exactly one of "num_points" and "real_points"')

	for i in range(num_points):
		# print 'Current progress', str(float(i + 1) / float(num_points) * 100.0) + '%                       \r',
		# sys.stdout.flush()

		if real_points is None:
			vr_tvec = Vec(-0.0428212, 0.0499129, 1.51173) # Vec(randomFloat(*x_range), randomFloat(*y_range), randomFloat(*z_range))
			vr_rmtx = Quat(randomFloat(*qw_range), randomFloat(*qx_range), randomFloat(*qy_range), randomFloat(*qz_range)).norm().toRotMatrix()
		else:
			vr_tvec = real_points[i].tvec
			vr_rmtx = real_points[i].rot_mtx

		this_rx_gm = rx_rel_gm.move(vr_rmtx, vr_tvec)

		start = datetime.now()
		tx_gmh, tx_gmv, rx_gmh, rx_gmv, num_iters = tracking(tx_gm, this_rx_gm)
		end = datetime.now()
		durs.append((end - start).total_seconds())
		
		if interpolation is not None:
			int_tx_gmh, int_tx_gmv, int_rx_gmh, int_rx_gmv = interpolation(vr_tvec, vr_rmtx)

		if real_points is not None:
			python_vals  = tx_gmh, tx_gmv, rx_gmh, rx_gmv
			interop_vals = int_tx_gmh, int_tx_gmv, int_rx_gmh, int_rx_gmv
			cpp_vals     = real_points[i].tx_vals[0], real_points[i].tx_vals[1], real_points[i].rx_vals[0], real_points[i].rx_vals[1]

			if real_points[i].tracking_method == 'Model':
				if any(abs(cv - pv) > 1 for cv, pv in zip(cpp_vals, python_vals)):
					print 'Mismatch'

			if real_points[i].tracking_method == '2-D_Interpolation':
				if any(abs(cv - iv) > 1 for cv, iv in zip(cpp_vals, interop_vals)):
					print 'Mismatch'

		tx_point, tx_dir = tx_gm.getOutput(tx_gmh, tx_gmv)
		rx_point, rx_dir = this_rx_gm.getOutput(rx_gmh, rx_gmv)

		tx_dist, tx_neg = distanceToLine(tx_point, tx_dir, rx_point)
		rx_dist, rx_neg = distanceToLine(rx_point, rx_dir, tx_point)

		if tx_neg or rx_neg:
			raise Exception('At least one GM is facing the wrong way')

		tx_errs.append(tx_dist)
		rx_errs.append(rx_dist)


	print 'Max tx err:', max(tx_errs) * 1000.0, 'mm'
	print 'Avg tx err:', sum(tx_errs) / float(len(tx_errs)) * 1000.0, 'mm'
	print 'Max rx err:', max(rx_errs) * 1000.0, 'mm'
	print 'Avg rx err:', sum(rx_errs) / float(len(rx_errs)) * 1000.0, 'mm'

	print ''
	print 'Max dur:', max(durs) * 1000.0, 'ms'
	print 'Avg dur:', sum(durs) / float(len(durs)) * 1000.0, 'ms'

if __name__ == '__main__':
	tx_gm = getGMFromFile('../data/2-2/tx_gm_vr_2-2.txt')
	rx_gm = getGMFromFile('../data/2-2/rx_gm_vr_2-2.txt')

	real_points = getVRData('../data/2-2/live_data_2-2.txt')

	interpolation = TwoDimensionalInterpolation(*getTwoDimensionalInterpolationData('../data/2-2/two_dimensional_interpolation_2-2.txt'))

	fullLinkTracking(tx_gm, rx_gm, num_points = None, real_points = real_points, tracking = LinkSearchTracking(1000, SearchTracking(1000)), interpolation = interpolation)
