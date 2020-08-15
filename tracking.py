
from find_gm_params import *
from convert_from_wall_to_vr import *

import random
# import sys

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
		for gm1_val in range(0, pow(2, 16), self.step):
			for gm2_val in range(0, pow(2, 16), self.step):
				p, d = gm.getOutput(gm1_val, gm2_val)
				dist, neg = distanceToLine(p, d, target_point)

				if not neg and (best_dist is None or dist < best_dist):
					best_dist = dist

					best_gm1_val = gm1_val
					best_gm2_val = gm2_val

		return best_gm1_val, best_gm2_val

class SearchTracking:
	def __init__(self, n_iter, use_float = False):
		self.n_iter = n_iter
		self.use_float = use_float

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

			if uh.mag() <= 1e-10 or uv.mag() <= 1e-10:
				break
			
			a, b = findComponentstoMinimizeDistance(uh, uv, cpc, target_point)

			if self.use_float:
				if abs(a) <= 1e-4 and abs(a) <= 1e-4:
					break

				cur_gmh += a
				cur_gmv += b
			else:
				a_int = int(round(a))
				b_int = int(round(b))

				if a_int == 0 and b_int == 0:
					break

				cur_gmh += a_int
				cur_gmv += b_int
		
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
			gm1_val, gm2_val = [int(round(x)) for x in [gm1_val, gm2_val]]
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

			if all([abs(x[0] - x[1]) < 1.0 for x in [(prev_tx_gmh, tx_gmh), (prev_tx_gmv, tx_gmv), (prev_rx_gmh, rx_gmh), (prev_rx_gmv, rx_gmv)]]):
				break

		return tx_gmh, tx_gmv, rx_gmh, rx_gmv
