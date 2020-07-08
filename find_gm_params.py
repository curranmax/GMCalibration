
import numpy as np
import cv2
import math
import copy
import random

import matplotlib.pyplot as plt

from scipy.optimize import least_squares, fsolve

all_data_stopping_constraints = {'xtol': 2.3e-16, 'ftol': 2.3e-16, 'gtol': 2.3e-16, 'max_nfev': 1e4}

# Angles are in radians
def vecFromAngle(alpha, beta):
	v = Vec(math.cos(alpha) * math.cos(beta),
			math.sin(beta),
			math.sin(alpha) * math.cos(beta))

	v.alpha = alpha
	v.beta = beta

	return v

def projectToPlane(vec, norm):
	proj = vec - norm.mult(vec.dot(norm))
	proj.norm()
	return proj

def reflect(in_dir, norm):
	return in_dir - norm.mult(2.0 * in_dir.dot(norm))

def findK(p_0, d, t):
	k = d.dot(t - p_0) / pow(d.mag(), 2.0)
	return k

def distanceToLine(p_0, d, t):
	k = findK(p_0, d, t)
	return (p_0 + d.mult(k)).dist(t), k < 0.0

class Vec:
	def __init__(self, x, y, z):
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)

		# Angles used to generate the vector
		self.alpha = None
		self.beta = None

	def dist(self, v):
		return math.sqrt(pow(self.x - v.x, 2.0) + \
						 pow(self.y - v.y, 2.0) + \
						 pow(self.z - v.z, 2.0))

	def mag(self):
		return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

	def norm(self):
		mag = self.mag()
		self.x = self.x / mag
		self.y = self.y / mag
		self.z = self.z / mag

		return self

	def dot(self, vec):
		return self.x * vec.x + self.y * vec.y + self.z * vec.z

	def angle(self, vec):
		if self.dot(vec) / self.mag() / vec.mag() > 0.9999999999999999999999999999999999999:
			return 0.0
		return math.acos(self.dot(vec) / self.mag() / vec.mag())

	def signedAngle(self, vec, norm):
		theta = self.angle(vec)
		cross = self.cross(vec)

		if norm.dot(cross) < 0:
			return -theta
		return theta

	def cross(self, vec):
		return Vec(self.y * vec.z - self.z * vec.y,
					self.z * vec.x - self.x * vec.z,
					self.x * vec.y - self.y * vec.x)

	def mult(self, v):
		return Vec(self.x * v,
					self.y * v,
					self.z * v)

	def getAngles(self):
		if self.alpha != None and self.beta != None:
			return self.alpha, self.beta

		if abs(self.mag() - 1.0) > 0.0001:
			raise Exception('Trying to get angles for non-unit vector')

		self.alpha = math.atan2(self.z, self.x)
		self.beta  = math.atan2(self.y, math.sqrt(pow(self.x, 2.0) + pow(self.z, 2.0)))

		return self.alpha, self.beta		


	def __add__(self, vec):
		return Vec(self.x + vec.x,
					self.y + vec.y,
					self.z + vec.z)

	def __sub__(self, vec):
		return Vec(self.x - vec.x,
					self.y - vec.y,
					self.z - vec.z)

	def __iter__(self):
		self.i = 0
		return self

	def __next__(self):
		if self.i < 3:
			if self.i == 0:
				v = self.x
			if self.i == 1:
				v = self.y
			if self.i == 2:
				v = self.z
			self.i += 1
			return v
		else:
			return StopIteration

	def __str__(self):
		return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'

# Creates a rotation matrix of angle t about axis v
def rotMatrix(v, t):
	a = math.cos(t) + pow(v.x, 2.0) * (1.0 - math.cos(t))
	b = v.x * v.y * (1.0 - math.cos(t)) - v.z * math.sin(t)
	c = v.x * v.z * (1.0 - math.cos(t)) + v.y * math.sin(t)

	d = v.y * v.x * (1.0 - math.cos(t)) + v.z * math.sin(t)
	e = math.cos(t) + pow(v.y, 2.0) * (1.0 - math.cos(t))
	f = v.y * v.z * (1.0 - math.cos(t)) - v.x * math.sin(t)

	g = v.z * v.x * (1.0 - math.cos(t)) - v.y * math.sin(t)
	h = v.z * v.y * (1.0 - math.cos(t)) + v.x * math.sin(t)
	i = math.cos(t) + pow(v.z, 2.0) * (1.0 - math.cos(t))

	return Matrix(a, b, c, d, e, f, g, h, i)

class Matrix:
	def __init__(self, a, b, c, d, e, f, g, h, i):
		self.vals = [[float(a), float(b), float(c)],
					 [float(d), float(e), float(f)],
					 [float(g), float(h), float(i)]]

		self.a1, self.a2, self.a3 = None, None, None

	def mult(self, v):
		if isinstance(v, Vec):
			return Vec(sum(a * b for a, b in zip(v, self.vals[0])),
						sum(a * b for a, b in zip(v, self.vals[1])),
						sum(a * b for a, b in zip(v, self.vals[2])))
		elif type(v) is float or type(v) is int:
			vs = [x * v for r in self.vals for x in r]
			return Matrix(*vs)
		else:
			raise Exception('Unexpected argument: ' + str(v))



	def transpose(self):
		return Matrix(self.vals[0][0],self.vals[1][0], self.vals[2][0],
						self.vals[0][1],self.vals[1][1], self.vals[2][1],
						self.vals[0][2],self.vals[1][2], self.vals[2][2])

	def getAngles(self):
		if self.a1 is None and self.a2 is None and self.a3 is None:
			raise Exception('Cannot compute angle from rot matrix yet')

		return self.a1, self.a2, self.a3

	def __add__(self, mtx):
		vs = [v1 + v2 for r1, r2 in zip(self.vals, mtx.vals) for v1, v2 in zip(r1, r2)]
		return Matrix(*vs)

	def __mul__(self, mtx):
		new_vals = [[sum(self.vals[rn][a] * mtx.vals[b][cn] for a, b in zip(list(range(3)), list(range(3)))) for cn in range(3)] for rn in range(3)]
		new_mtx = Matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
		new_mtx.vals = new_vals

		return new_mtx

	def __str__(self):
		return '\n'.join(['[' + ', '.join(map(str, row)) + ']' for row in self.vals])

angle_algo = 'ypr'
def rotMatrixFromAngles(a1, a2, a3):
	global angle_algo
	if angle_algo == 'quat':
		return quatFromAngle(a1, a2, a3).toRotMatrix()

	if angle_algo == 'ypr':
		mtx = rotMatrix(Vec(1.0, 0.0, 0.0), a1) * rotMatrix(Vec(0.0, 1.0, 0.0), a2) * rotMatrix(Vec(0.0, 0.0, 1.0), a3)
		mtx.a1, mtx.a2, mtx.a3 = a1, a2, a3

		return mtx

def quatFromAngle(theta, alpha, beta):
	v = vecFromAngle(alpha, beta).mult(math.sin(theta))
	q = Quat(math.cos(theta), v.x, v.y, v.z)

	q.theta = theta
	q.alpha = alpha
	q.beta  = beta

	return q

# Takes a list of unit-quats and returns the average unit-quat
def computeAvgQuat(quats):
	avg_quat = sum((q for q in quats), Quat(0.0, 0.0, 0.0, 0.0)).mult(1.0 / float(len(quats)))
	return avg_quat.mult(1.0 / avg_quat.mag())

class Quat:
	def __init__(self, w, x, y, z):
		self.w = float(w)
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)

		self.theta, self.alpha, self.beta = None, None, None

	def toRotMatrix(self):
		a = 1.0 - 2.0 * pow(self.y, 2.0) - 2.0 * pow(self.z, 2.0)
		b = 2.0 * self.x * self.y - 2.0 * self.w * self.z
		c = 2.0 * self.x * self.z + 2.0 * self.w * self.y

		d = 2.0 * self.y * self.x + 2.0 * self.w * self.z
		e = 1.0 - 2.0 * pow(self.x, 2.0) - 2.0 * pow(self.z, 2.0)
		f = 2.0 * self.y * self.z - 2.0 * self.w * self.x

		g = 2.0 * self.z * self.x - 2.0 * self.w * self.y
		h = 2.0 * self.z * self.y + 2.0 * self.w * self.x
		i = 1.0 - 2.0 * pow(self.x, 2.0) - 2.0 * pow(self.y, 2.0)

		m = Matrix(a, b, c, d, e, f, g, h, i)

		m.a1, m.a2, m.a3 = self.theta, self.alpha, self.beta
		return m

	def mag(self):
		return math.sqrt(sum([pow(v, 2.0) for v in (self.w, self.x, self.y, self.z)]))

	def norm(self):
		m = self.mag()
		return Quat(*[v / m for v in (self.w, self.x, self.y, self.z)])

	def dist(self, q):
		return math.sqrt(pow(self.w - q.w, 2.0) + pow(self.x - q.x, 2.0) + pow(self.y - q.y, 2.0) + pow(self.z - q.z, 2.0))

	def mult(self, v):
		return Quat(self.w * v,
					self.x * v,
					self.y * v,
					self.z * v)

	def __add__(self, q):
		return Quat(self.w + q.w,
					self.x + q.x,
					self.y + q.y,
					self.z + q.z)

	def __str__(self):
		return '(' + ', '.join([v[0] + ': ' + str(v[1]) for v in [('W', self.w), ('X', self.x), ('Y', self.y), ('Z', self.z)]]) + ')'

class Plane:
	def __init__(self, norm, point):
		self.norm  = norm
		self.point = point

		self.norm.norm()

	def rotate(self, rot_axis, theta):
		rm = rotMatrix(rot_axis, theta)
		return Plane(rm.mult(self.norm), self.point)

	def intersect(self, p, d):
		# Find k such that (p + d.mult(k) - self.point).dot(self.norm) = 0

		a = d.dot(self.norm)
		b = (p - self.point).dot(self.norm)
		
		# If a is zero, then the line doesn't intersect the plane
		if abs(a) < 0.00000000000001:
			return None, None

		k = -b / a

		intersection_point = d.mult(k) + p
		return intersection_point, (k < 0.0)

	def __str__(self):
		return '{Normal: ' + str(self.norm) + ', Point:'  + str(self.point) + '}'

def gmValToRadian(gm_val, params = None):
	voltage = gmValToVoltage(gm_val)
	radian  = voltageToRadian(voltage, params = params)
	return radian

def gmValToVoltage(gm_val):
	voltage = 20.0 / pow(2.0, 16.0) * gm_val - 10.0
	return voltage

def voltageToRadian(voltage, params = None):
	if params is None:
		# Default linear function
		return -2.0 * math.pi / 180.0 * voltage

	if len(params) == 1:
		# Linear function
		return params[0] * voltage

	if len(params) == 2:
		return params[0] * voltage + params[1] * pow(voltage, 2.0)

	raise Exception('Unexpected value of params: ' + str(params))

def getGMFromCAD(init_dir, init_point):
	# gm = GM(init_dir, init_point,
	# 		Plane(Vec(-0.183253086052, 0.68303422983, -0.707023724721), Vec(43.3404917057, 29.865539516, 15.0005588143)),
	# 		Vec(0.965927222808, 0.258813833165, 0.0),
	# 		Plane(Vec(0.793391693847, -0.608711442422, 0.0), Vec(40.2024772148, 44.6917730708, 14.1303253915)),
	# 		Vec(0.0, 0.0, -1.0))

	gm = GM(init_dir, init_point,
			Plane(Vec(-0.707023724726, 0.683034229835, 0.183253086053), Vec(15.0005588143, 29.865539516, -43.3404917057)),
			Vec(0.0, 0.258813833165, -0.965927222808),
			Plane(Vec(0.0, -0.608711442422, -0.793391693847), Vec(14.1303253915, 44.6917730708, -40.2024772148)),
			Vec(-1.0, 0.0, 0.0))

	return gm

def outputGM(gm, fname):
	f = open(fname, 'w')

	to_string = lambda vec: str(vec.x) + ' ' + str(vec.y) + ' ' + str(vec.z)

	# Input beam
	f.write('input_dir ' + to_string(gm.init_dir) + '\n')
	f.write('input_point ' + to_string(gm.init_point) + '\n')

	# M1 (normal, point, axis)
	f.write('m1_norm ' + to_string(gm.m1.norm) + '\n')
	f.write('m1_point ' + to_string(gm.m1.point) + '\n')
	f.write('m1_axis ' + to_string(gm.a1) + '\n')

	# M2 (normal, point, axis)
	f.write('m2_norm ' + to_string(gm.m2.norm) + '\n')
	f.write('m2_point ' + to_string(gm.m2.point) + '\n')
	f.write('m2_axis ' + to_string(gm.a2) + '\n')

def getGMFromFile(fname):
	f = open(fname, 'r')

	vals = {'input_dir':None, 'input_point':None, 'm1_norm':None, 'm1_point':None, 'm1_axis':None, 'm2_norm':None, 'm2_point':None, 'm2_axis':None}

	for line in f:
		token, x, y, z = line.split()

		vec = Vec(*list(map(float, (x, y, z))))

		if token not in vals:
			raise Exception('Unexpected token: ' + str(token))

		vals[token] = vec

	m1 = Plane(vals['m1_norm'], vals['m1_point'])
	m2 = Plane(vals['m2_norm'], vals['m2_point'])

	gm = GM(vals['input_dir'], vals['input_point'], m1, vals['m1_axis'], m2, vals['m2_axis'])

	return gm

class GM:
	def __init__(self, init_dir, init_point, m1, a1, m2, a2, mirror_thickness = 0.0, val_params = None):
		# init_dir and init_point are Vecs, and are the direction and location of the initial laser beam
		self.init_dir = init_dir
		self.init_point = init_point
	
		# m1 and m2 are the Planes, and is the first and second mirror at 0 Volts
		self.m1 = m1
		self.m2 = m2

		# a1 and a2 are Vecs, and the axis of rotation
		self.a1 = a1
		self.a2 = a2

		self.init_dir.norm()
		self.a1.norm()
		self.a2.norm()

		self.mirror_thickness = mirror_thickness
		self.m1.point = self.m1.point
		self.m2.point = self.m2.point

		self.val_params = val_params

	def scale(self, v):
		self.init_point = self.init_point.mult(v)
		self.m1.point   = self.m1.point.mult(v)
		self.m2.point   = self.m2.point.mult(v)

	# Creates a new GM that is 1) rotated using the supplied rotation matrix, and 2) translated by trans_vec
	def move(self, rot_matrix, trans_vec):
		rot = lambda x: rot_matrix.mult(x)
		rot_and_trans = lambda x: rot_matrix.mult(x) + trans_vec
		return GM(rot(self.init_dir), rot_and_trans(self.init_point),
					Plane(rot(self.m1.norm), rot_and_trans(self.m1.point)), rot(self.a1),
					Plane(rot(self.m2.norm), rot_and_trans(self.m2.point)), rot(self.a2),
					mirror_thickness = self.mirror_thickness, val_params = self.val_params)

	def getOutput(self, gm1_val, gm2_val, check_intersection = False):
		# print 'GM1 angle:', gmValToRadian(gm1_val)
		# print 'GM2 angle:', gmValToRadian(gm2_val)

		this_m1 = self.m1.rotate(self.a1, gmValToRadian(gm1_val, params = self.val_params))
		this_m2 = self.m2.rotate(self.a2, gmValToRadian(gm2_val, params = self.val_params))

		this_m1.point = this_m1.point + this_m1.norm.mult(self.mirror_thickness)
		this_m2.point = this_m2.point + this_m2.norm.mult(self.mirror_thickness)

		# print 'M1:', this_m1
		# print 'M2:', this_m2

		intersect = True

		p1, neg_k = this_m1.intersect(self.init_point, self.init_dir)

		if p1 == None or neg_k:
			p1 = self.init_point
			d1 = self.init_dir
			intersect = False
		else:
			d1 = reflect(self.init_dir, this_m1.norm)

		p2, neg_k = this_m2.intersect(p1, d1)

		if p2 == None or neg_k:
			p2 = p1
			d2 = d1
			intersect = False
		else:
			d2 = reflect(d1, this_m2.norm)

		if check_intersection:
			return intersect
		else:
			return p2, d2

	def __str__(self):
		return '\tInit Loc: ' + str(self.init_point) + '\n' + \
			   '\tInit Dir: ' + str(self.init_dir) + '\n' + \
			   '\tM1 Norm:  ' + str(self.m1.norm) + '\n' + \
			   '\tM1 Point: ' + str(self.m1.point) + '\n' + \
			   '\tM1 Axis:  ' + str(self.a1) + '\n' + \
			   '\tM2 Norm:  ' + str(self.m2.norm) + '\n' + \
			   '\tM2 Point: ' + str(self.m2.point) + '\n' + \
			   '\tM2 Axis:  ' + str(self.a2)

class Dot:
	def __init__(self, gmh, gmv, loc):
		self.gmh = gmh
		self.gmv = gmv

		self.loc = loc

		self.x_ind, self.y_ind = None, None

		self.corner_ind = None

def drawDots(collected_dots, gen_dots, out_fname = 'dot.jpg'):
	min_x, max_x = min(d.loc.x for d in collected_dots), max(d.loc.x for d in collected_dots)
	min_y, max_y = min(d.loc.y for d in collected_dots), max(d.loc.y for d in collected_dots)

	width  = 4000
	buf = 0.1

	if 1.0 - 2.0 * buf < 0.0:
		raise Exception('Invalid buffer: ' + str(buf))
	
	ox = min_x - (max_x - min_x) / (1.0 - 2.0 * buf) * buf
	oy = min_y - (max_y - min_y) / (1.0 - 2.0 * buf) * buf
	origin_point = Vec(ox, oy, 0.0)

	conversion_factor = width * (1.0 - 2.0 * buf) / (max_x - min_x)
	height = int(math.ceil(conversion_factor * (max_y - min_y) / (1.0 - 2.0 * buf)))

	
	brush_size = 10
	brush_type = 'circle'
	brush = []
	for a in range(-brush_size, brush_size + 1, 1):
		for b in range(-brush_size, brush_size + 1, 1):
			valid = True
			if brush_type == 'circle':
				valid = (math.sqrt(pow(a, 2) + pow(b, 2)) <= brush_size)
			elif brush_type == 'square':
				pass
			else:
				raise Exception('Invalid brush_type: ' + str(brush_type))

			if valid:
				brush.append((a, b))

	corner_colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 255)]

	img = np.zeros((height, width, 3), np.uint8)
	for dots, channel in [(collected_dots, 0), (gen_dots, 1)]:
		if dots == None:
			continue

		for d in dots:
			# Convert 3d location to pixel
			if abs(d.loc.z) > 0.0000001:
				raise Exception('Assumed that all dots are on the plane z=0')

			yp = int((d.loc.y - origin_point.y) * conversion_factor)
			xp = int((d.loc.x - origin_point.x) * conversion_factor)

			for a, b in brush:
				if yp + a >= 0 and yp + a < height and xp + b >= 0 and xp + b < width:
					if channel == 1:
						img[yp + a][xp + b][channel] = 255

					if d.corner_ind != None:
						img[yp + a][xp + b] = corner_colors[d.corner_ind]

	cv2.imwrite(out_fname, img)

def getDotsFromFile(filename):
	f = open(filename, 'r')

	cols = None
	col_inds = {'GMH': None, 'GMV': None, 'X': None, 'Y': None, 'Z': None}
	dots = []

	corners = []
	if filename =='data/2-14/data_2-14.txt':
		corners = [Vec(0.0, 0.0, 0.0), Vec(666.0, 0.0, 0.0), Vec(0.0, 481.0, 0.0), Vec(666.0, 481.0, 0.0)]

	if filename == 'data/2-26/data_2-26.txt':
		corners = [Vec(0.0, 0.0, 0.0), Vec(662.4, 0.0, 0.0), Vec(0.0, 478.4, 0.0), Vec(662.4, 478.4, 0.0)]

	if filename == 'data/3-15/data_3-15.txt':
		corners = [Vec(0.0, 0.0, 0.0), Vec(664.2, 0.0, 0.0), Vec(0.0, 516.6, 0.0), Vec(664.2, 516.6, 0.0)]

	if filename == 'data/5-21/dot_data_5-21.txt':
		corners = [Vec(0.0, 0.0, 0.0), Vec(1061.6, 0.0, 0.0), Vec(0.0, 796.2, 0.0), Vec(1061.6, 796.2, 0.0)]

	for line in f:
		if cols == None:
			cols = line.split()

			for k in col_inds:
				col_inds[k] = cols.index(k)
		else:
			vals = line.split()
			if len(cols) != len(vals):
				raise Exception('All rows must be of length ' + str(len(cols)))

			loc = Vec(float(vals[col_inds['X']]), float(vals[col_inds['Y']]), float(vals[col_inds['Z']]))

			this_dot = Dot(int(vals[col_inds['GMH']]), int(vals[col_inds['GMV']]), loc)

			for i, corner_loc in enumerate(corners):
				if corner_loc.dist(this_dot.loc) < 0.1:
					this_dot.corner_ind = i

			dots.append(this_dot)

	f.close()

	xvals = set(d.loc.x for d in dots)
	yvals = set(d.loc.y for d in dots)

	xvals = sorted(list(xvals))
	yvals = sorted(list(yvals))

	for d in dots:
		d.x_ind = xvals.index(d.loc.x)
		d.y_ind = yvals.index(d.loc.y)

	return dots

def filterDots(dots, exclude_boundary = None, max_num_dots = None):
	# If exclude_boundary is given as an int, then the outermost number of given rows and columns are removed
	if exclude_boundary is not None:
		min_x, max_x = min(d.x_ind for d in dots), max(d.x_ind for d in dots)
		min_y, max_y = min(d.y_ind for d in dots), max(d.y_ind for d in dots)

		# Boundary values (inclusive)
		min_bx, max_bx = min_x + exclude_boundary, max_x - exclude_boundary
		min_by, max_by = min_y + exclude_boundary, max_y - exclude_boundary

		dots = [d for d in dots if d.x_ind >= min_bx and d.x_ind <= max_bx and d.y_ind >= min_by and d.y_ind <= max_by]

	if max_num_dots is not None and len(dots) > max_num_dots:
		dots = random.sample(dots, max_num_dots)

	return dots

def generateDots(gm, collected_dots, wall_plane, verbose = True):
	dots = []
	is_dot_intersection_neg = []
	behind_wall = 0
	for cdot in collected_dots:
		if not gm.getOutput(cdot.gmh, cdot.gmv, check_intersection = True):
			raise Exception('No intersection for gm values (' + str(cdot.gmh) + ', ' + str(cdot.gmv) + ')')

		p, d = gm.getOutput(cdot.gmh, cdot.gmv)

		ip, neg = wall_plane.intersect(p, d)

		if neg is None or neg is True:
			behind_wall += 1

		is_dot_intersection_neg.append(neg)

		this_dot = Dot(cdot.gmh, cdot.gmv, ip)
		this_dot.corner_ind = cdot.corner_ind
		this_dot.x_ind, this_dot.y_ind = cdot.x_ind, cdot.y_ind

		dots.append(this_dot)

	if verbose and behind_wall > 0:
		print(behind_wall, 'out of', len(collected_dots), 'are behind the wall')

	if verbose:
		return dots
	else:
		return dots, is_dot_intersection_neg

def getGridStats(dots):
	dots_by_ind = {(dot.x_ind, dot.y_ind): dot for dot in dots}

	xvecs = []
	yvecs = []
	for (x, y), dot in dots_by_ind.items():
		if (x + 1, y) in dots_by_ind:
			xvecs.append(dots_by_ind[(x + 1, y)].loc - dot.loc)
		if (x, y + 1) in dots_by_ind:
			yvecs.append(dots_by_ind[(x, y + 1)].loc - dot.loc)

	avg_xvec = sum(xvecs, Vec(0.0, 0.0, 0.0)).mult(1.0 / float(len(xvecs)))
	avg_yvec = sum(yvecs, Vec(0.0, 0.0, 0.0)).mult(1.0 / float(len(yvecs)))
	
	print(avg_xvec.dot(avg_yvec))

	prev_mag = avg_yvec.mag()
	avg_yvec = (avg_yvec - avg_xvec.mult(avg_xvec.dot(avg_yvec) / avg_xvec.mag() / avg_yvec.mag())).norm().mult(prev_mag)

	x_errs = [xv.dist(avg_xvec) for xv in xvecs]
	y_errs = [yv.dist(avg_yvec) for yv in yvecs]

	return avg_xvec, avg_yvec, x_errs, y_errs

def calculateGridError(collected_dots, gen_dots, verbose = False):
	gx, gy, gx_errs, gy_errs = getGridStats(gen_dots)

	if verbose:
		# cx, cy, cx_errs, cy_errs = getGridStats(collected_dots)
		print('Angle between axes:', gx.angle(gy) * 180.0 / math.pi)
		print('Avg x error:       ', sum(gx_errs) / float(len(gx_errs)))
		print('Max x error:       ', max(gx_errs))
		print('Avg y error:       ', sum(gy_errs) / float(len(gy_errs)))
		print('Max y error:       ', max(gy_errs))

		# print 'Dif x mag:         ', gx.mag() - cx.mag()
		# print 'Dif y mag:         ', gy.mag() - cy.mag()

	return gx_errs + gy_errs

# Find GM Param functions

# Finds the orientation and position of a given GM that minimizes the error between observed and calculated dots.
def makeRotAndTransFunction(gm, dots, wall_plane = Plane(Vec(0.0, 0.0, 1.0), Vec(0.0, 0.0, 0.0)),
							R_static = None, T_static = None, modify_init_dir = True, modify_init_point = True,
							modify_m1_norm = True, modify_m1_point = True, modify_m1_axis = True,
							modify_m2_norm = True, modify_m2_point = True, modify_m2_axis = True,
							modify_linear_val_params = False, modify_quadratic_val_params = False,
							error_type = 'distance'):
	if error_type not in ['distance', 'difference']:
		raise Exception('Invalid error_type')

	def eq(vs):
		# Get values from solver
		x = 0
		if R_static == None:
			r_a1, r_a2, r_a3 = vs[x : x+3]
			x += 3

			R = rotMatrixFromAngles(r_a1, r_a2, r_a3)
		else:
			R = R_static

		if T_static == None:
			t_x, t_y, t_z = vs[x : x+3]
			x += 3

			T = Vec(t_x, t_y, t_z)
		else:
			T = T_static

		this_gm = copy.deepcopy(gm)

		if modify_init_dir:
			id_a, id_b = vs[x : x+2]
			x += 2

			this_gm.init_dir = vecFromAngle(id_a, id_b)

		if modify_init_point:
			ip_x, ip_y, ip_z = vs[x : x+3]
			x += 3

			this_gm.init_point = Vec(ip_x, ip_y, ip_z)

		if modify_m1_norm:
			n1_a, n1_b = vs[x : x+2]
			x+=2

			this_gm.m1.norm = vecFromAngle(n1_a, n1_b)

		if modify_m1_point:
			p1_x, p1_y, p1_z = vs[x : x+3]
			x+=3

			this_gm.m1.point = Vec(p1_x, p1_y, p1_z)

		if modify_m1_axis:
			a1_a, a1_b = vs[x : x+2]
			x+=2

			this_gm.a1 = vecFromAngle(a1_a, a1_b)

		if modify_m2_norm:
			n2_a, n2_b = vs[x : x+2]
			x+=2

			this_gm.m2.norm = vecFromAngle(n2_a, n2_b)

		if modify_m2_point:
			p2_x, p2_y, p2_z = vs[x : x+3]
			x+=3

			this_gm.m2.point = Vec(p2_x, p2_y, p2_z)

		if modify_m2_axis:
			a2_a, a2_b = vs[x : x+2]
			x+=2

			this_gm.a2 = vecFromAngle(a2_a, a2_b)

		if modify_linear_val_params:
			vp = vs[x : x+1]
			x += 1

			this_gm.val_params = (vp,)

		if modify_quadratic_val_params:
			vp_a, vp_b = vs[x : x+2]
			x += 2

			this_gm.val_params = (vp_a, vp_b)

		# Move the GM based on the input from the solver
		rt_gm = this_gm.move(R, T)

		# Calculate the error between the collected and generated data
		no_intersection = any(not rt_gm.getOutput(dot.gmh, dot.gmv, check_intersection = True) for dot in dots)

		rv = []
		inf_val = pow(10.0, 20.0)
		for dot in dots:
			p, d = rt_gm.getOutput(dot.gmh, dot.gmv)
			intersection_point, neg = wall_plane.intersect(p, d)

			if no_intersection is True or neg is None or neg is True:
				if error_type == 'difference':
					rv += [inf_val, inf_val, inf_val]
				elif error_type == 'distance':
					rv.append(inf_val)
			else:
				if error_type == 'difference':
					rv += [intersection_point.x - dot.loc.x, intersection_point.y - dot.loc.y, intersection_point.z - dot.loc.z]
				elif error_type == 'distance':
					rv.append(intersection_point.dist(dot.loc))

		return rv

	return eq

def findRotAndTransParams(gm, R, T, dots, wall_plane = Plane(Vec(0.0, 0.0, 1.0), Vec(0.0, 0.0, 0.0)),
							modify_R = True, modify_T = True, modify_init_dir = True, modify_init_point = True,
							modify_m1_norm = True, modify_m1_point = True, modify_m1_axis = True,
							modify_m2_norm = True, modify_m2_point = True, modify_m2_axis = True,
							modify_linear_val_params = False, modify_quadratic_val_params = False,
							low_error_size = None):
	if all(not v for v in [modify_R, modify_T, modify_init_dir, modify_init_point,
			modify_m1_norm, modify_m1_point, modify_m1_axis,
			modify_m2_norm, modify_m2_point, modify_m2_axis,
			modify_linear_val_params, modify_quadratic_val_params]):
		return dots, R, T, gm.init_dir, gm.init_point, gm.m1.norm, gm.m1.point, gm.a1, gm.m2.norm, gm.m2.point, gm.a2, gm.val_params

	R_static = None
	if not modify_R:
		R_static = R

	T_static = None
	if not modify_T:
		T_static = T

	f = makeRotAndTransFunction(gm, dots, wall_plane,
								R_static = R_static, T_static = T_static,
								modify_init_dir = modify_init_dir, modify_init_point = modify_init_point,
								modify_m1_norm = modify_m1_norm, modify_m1_point = modify_m1_point, modify_m1_axis = modify_m1_axis,
								modify_m2_norm = modify_m2_norm, modify_m2_point = modify_m2_point, modify_m2_axis = modify_m2_axis,
								modify_linear_val_params = modify_linear_val_params, modify_quadratic_val_params = modify_quadratic_val_params,
								error_type = 'distance')

	solving_for = []

	init_guess = []
	bound_vals = []
	if modify_R:
		init_guess += list(R.getAngles())
		bound_vals += [5.0 * math.pi / 180.0] * 3

		solving_for.append('Rotation matrix')

	if modify_T:
		init_guess += [T.x, T.y, T.z]
		bound_vals += [20.0] * 3

		solving_for.append('Translation vector')

	if modify_init_dir:
		init_guess += list(gm.init_dir.getAngles())
		bound_vals += [10.0 * math.pi / 180.0] * 2

		solving_for.append('Initial direction')

	if modify_init_point:
		init_guess += [gm.init_point.x, gm.init_point.y, gm.init_point.z]
		bound_vals += [50.0] * 3

		solving_for.append('Initial point')

	if modify_m1_norm:
		init_guess += list(gm.m1.norm.getAngles())
		bound_vals += [5.0 * math.pi / 180.0] * 2

		solving_for.append('M1 normal')

	if modify_m1_point:
		init_guess += [gm.m1.point.x, gm.m1.point.y, gm.m1.point.z]
		bound_vals += [10.0] * 3

		solving_for.append('M1 point')

	if modify_m1_axis:
		init_guess += list(gm.a1.getAngles())
		bound_vals += [5.0 * math.pi / 180.0] * 2

		solving_for.append('M1 axis')

	if modify_m2_norm:
		init_guess += list(gm.m2.norm.getAngles())
		bound_vals += [5.0 * math.pi / 180.0] * 2

		solving_for.append('M2 normal')

	if modify_m2_point:
		init_guess += [gm.m2.point.x, gm.m2.point.y, gm.m2.point.z]
		bound_vals += [10.0] * 3

		solving_for.append('M2 point')

	if modify_m2_axis:
		init_guess += list(gm.a2.getAngles())
		bound_vals += [5.0 * math.pi / 180.0] * 2

		solving_for.append('M2 axis')

	if modify_linear_val_params:
		init_guess += list(gm.val_params)
		bound_vals += [0.1]

		solving_for.append('Linear Val Params')

	if modify_quadratic_val_params:
		init_guess += list(gm.val_params)
		bound_vals += [0.1] * 2

		solving_for.append('Quadratic Val Params')

	print('\nSolving For:\n' + ', '.join(solving_for) + '\n')

	min_bounds = [v - b for v, b in zip(init_guess, bound_vals)]
	max_bounds = [v + b for v, b in zip(init_guess, bound_vals)]

	print('Stopping constraints:', all_data_stopping_constraints)
	rv = least_squares(f, tuple(init_guess), **all_data_stopping_constraints)

	cost = rv.cost

	x = 0
	if R_static == None:
		r_a1, r_a2, r_a3 = rv.x[x : x+3]
		x += 3

		R = rotMatrixFromAngles(r_a1, r_a2, r_a3)
	else:
		R = R_static

	if T_static == None:
		t_x, t_y, t_z = rv.x[x : x+3]
		x += 3

		T = Vec(t_x, t_y, t_z)
	else:
		T = T_static

	if modify_init_dir:
		id_a, id_b = rv.x[x : x+2]
		x += 2

		init_dir = vecFromAngle(id_a, id_b)
	else:
		init_dir = gm.init_dir

	if modify_init_point:
		ip_x, ip_y, ip_z = rv.x[x : x+3]
		x += 3

		init_point = Vec(ip_x, ip_y, ip_z)
	else:
		init_point = gm.init_point

	if modify_m1_norm:
		n1_a, n1_b = rv.x[x : x+2]
		x += 2

		m1_norm = vecFromAngle(n1_a, n1_b)
	else:
		m1_norm = gm.m1.norm

	if modify_m1_point:
		p1_x, p1_y, p1_z = rv.x[x : x+3]
		x += 3

		m1_point = Vec(p1_x, p1_y, p1_z)
	else:
		m1_point = gm.m1.point

	if modify_m1_axis:
		a1_a, a1_b = rv.x[x : x+2]
		x += 2

		m1_axis = vecFromAngle(a1_a, a1_b)
	else:
		m1_axis = gm.a1

	if modify_m2_norm:
		n2_a, n2_b = rv.x[x : x+2]
		x += 2

		m2_norm = vecFromAngle(n2_a, n2_b)
	else:
		m2_norm = gm.m2.norm

	if modify_m2_point:
		p2_x, p2_y, p2_z = rv.x[x : x+3]
		x += 3

		m2_point = Vec(p2_x, p2_y, p2_z)
	else:
		m2_point = gm.m2.point

	if modify_m2_axis:
		a2_a, a2_b = rv.x[x : x+2]
		x += 2

		m2_axis = vecFromAngle(a2_a, a2_b)
	else:
		m2_axis = gm.a2

	if modify_linear_val_params:
		vp = rv.x[x : x+1]
		x += 1

		val_params = (vp,)
	elif modify_quadratic_val_params:
		vp_a, vp_b = rv.x[x : x+2]
		x += 2

		val_params = (vp_a, vp_b)
	else:
		val_params = gm.val_params

	if low_error_size is not None:
		if low_error_size < 0:
			low_error_size = len(dots) + low_error_size

		if low_error_size <= 0 or low_error_size >= len(dots):
			raise Exception('Invalid value fo low_error_size; must be between 1 and ' + str(len(dots) - 1))

		dot_errs = [(e, d) for e, d in zip(f(rv.x), dots)]
		dot_errs.sort()

		err_of_dots  = [e for e, _ in dot_errs[:low_error_size]]
		low_err_dots = [d for _, d in dot_errs[:low_error_size]]

		print('\nRunning regression with only low error points:')
		print('Max error of low error points in full regression:', max(err_of_dots))
		print('Avg error of low error points in full regression:', sum(err_of_dots) / float(len(err_of_dots)))
		print('')

		this_gm = GM(init_dir, init_point, Plane(m1_norm, m1_point), m1_axis, Plane(m2_norm, m2_point), m2_axis)

		_, R, T, init_dir, init_point, m1_norm, m1_point, m1_axis, m2_norm, m2_point, m2_axis, val_params = \
				findRotAndTransParams(this_gm, R, T, low_err_dots, wall_plane = wall_plane,
					modify_R = modify_R, modify_T = modify_T, modify_init_dir = modify_init_dir, modify_init_point = modify_init_point,
					modify_m1_norm = modify_m1_norm, modify_m1_point = modify_m1_point, modify_m1_axis = modify_m1_axis,
					modify_m2_norm = modify_m2_norm, modify_m2_point = modify_m2_point, modify_m2_axis = modify_m2_axis,
					modify_linear_val_params = modify_linear_val_params, modify_quadratic_val_params = modify_quadratic_val_params,
					low_error_size = None)

		return low_err_dots, R, T, init_dir, init_point, m1_norm, m1_point, m1_axis, m2_norm, m2_point, m2_axis, val_params

	return dots, R, T, init_dir, init_point, m1_norm, m1_point, m1_axis, m2_norm, m2_point, m2_axis, val_params

def calculateError(collected_dots, gen_dots):
	cdots = dict()
	for d in collected_dots:
		k = (d.gmh, d.gmv)
		if k in cdots:
			raise Exception('Repeat key: ' + str(k))

		cdots[k] = d

	gdots = dict()
	for d in gen_dots:
		k = (d.gmh, d.gmv)

		if k not in cdots:
			raise Exception('Mismatch keys: ' + str(k))
		if k in gdots:
			raise Exception('Repeat key: ' + str(k))

		gdots[k] = d

	errors = []
	for k in cdots:
		cd = cdots[k]
		gd = gdots[k]

		this_error = cd.loc.dist(gd.loc)

		errors.append(this_error)

	drawHistogram(errors, n_bins = 10000, title = 'Error in GM Model')

	print('Average Error:', sum(errors) / float(len(errors)), 'mm')
	print('Max Error:    ', max(errors), 'mm')

def differenceBetweenGM(gm1, gm2):
	print('Angle between initial beam:', gm1.init_dir.angle(gm2.init_dir))
	print('Distance between initial beam starts:', min(distanceToLine(gm1.init_point, gm1.init_dir, gm2.init_point), distanceToLine(gm2.init_point, gm2.init_dir, gm1.init_point)))
	print('Angle between m1:', gm1.m1.norm.angle(gm2.m1.norm))
	print('Angle between a1:', gm1.a1.angle(gm2.a1))
	print('Distance between m1.point:', min(distanceToLine(gm1.m1.point, gm1.a1, gm2.m1.point), distanceToLine(gm2.m1.point, gm2.a1, gm1.m1.point)))
	print('Angle between m2:', gm1.m2.norm.angle(gm2.m2.norm))
	print('Angle between a2:', gm1.a2.angle(gm2.a2))
	print('Distance between m2.point:', min(distanceToLine(gm1.m2.point, gm1.a2, gm2.m2.point), distanceToLine(gm2.m2.point, gm2.a2, gm1.m2.point)))

def rangeOfLaunchPoint(gm):
	pc, _ = gm.getOutput(pow(2, 15), pow(2, 15))

	p_nh, _ = gm.getOutput(0, pow(2, 15))
	p_ph, _ = gm.getOutput(pow(2, 16), pow(2, 15))

	p_nv, _ = gm.getOutput(pow(2, 15), 0)
	p_pv, _ = gm.getOutput(pow(2, 15), pow(2, 16))

	p_nh_nv, _ = gm.getOutput(0, 0)
	p_nh_pv, _ = gm.getOutput(0, pow(2, 16))
	p_ph_nv, _ = gm.getOutput(pow(2, 16), 0)
	p_ph_pv, _ = gm.getOutput(pow(2, 16), pow(2, 16))

	cross_dists = [p.dist(pc) for p in [p_nh, p_ph, p_nv, p_pv]]
	corner_dists = [p.dist(pc) for p in [p_nh_nv, p_nh_pv, p_ph_nv, p_ph_pv]]

	print('Max distance between center launch point and extremes:', max(cross_dists + corner_dists))

# Drawing histograms
def drawHistogram(errs, n_bins = 100, title = '', x_label = 'Error (mm)', y_label = 'CDF'):
	fig, ax = plt.subplots()

	n, bins, patches = ax.hist(errs, n_bins,
								normed = True, histtype = 'step',
								cumulative = True)

	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)

	plt.show()

def getTXInitData():
	global angle_algo
	angle_algo = 'ypr'

	print('\nSolving for TX GM\n')
	collected_dots = getDotsFromFile('data/12-2/tx_board_data_12-2.txt')
	
	# init_dir = vecFromAngle(0.0, 0.0)
	# init_point = Vec(5.0005588143, 29.865539516, -43.3404917057)

	# init_gm = getGMFromCAD(init_dir, init_point)
	# init_gm.mirror_thickness = 0.0

	# outputGM(init_gm, 'data/12-2/gm_init_12-2.txt')

	init_R = rotMatrixFromAngles(0.0, 0.0, 0.0)
	init_T = Vec(0.0, 0.0, 0.0)
	# init_T = Vec(503.85, -54.61, 1492.74)

	init_gm = getGMFromFile('data/12-2/tx_gm_board_12-2.txt')


	local_fname = None # 'data/12-2/tx_gm_local_12-2.txt'
	abs_fname   = None # 'data/12-2/tx_gm_board_12-2.txt'

	return init_gm, init_R, init_T, collected_dots, local_fname, abs_fname

def getRXInitData():
	global angle_algo
	angle_algo = 'ypr'

	print('\nSolving for RX GM\n')
	collected_dots = getDotsFromFile('data/12-2/rx_board_data_12-2.txt')

	y_filter = 53.08 * 5 + 0.1

	if y_filter is None:
		new_collected_dots = []
		for dot in collected_dots:
			if dot.loc.y <= y_filter:
				new_collected_dots.append(dot)

		collected_dots = new_collected_dots

	# init_dir = vecFromAngle(0.0, 0.0)
	# init_point = Vec(5.0005588143, 29.865539516, -43.3404917057)

	# init_gm = getGMFromCAD(init_dir, init_point)

	init_gm = getGMFromFile('data/12-5/rx_gm_board_12-5.txt')

	init_R = rotMatrixFromAngles(0.0, 0.0, 0.0)
	init_T = Vec(0.0, 0.0, 0.0)
	# init_T = Vec(403.85, -15.61, 1202.74)
	# init_T = Vec(401.594856542, 0.955179254573, 1208.9688021)
	
	local_fname = None # 'data/12-5/rx_gm_local_12-5.txt'
	abs_fname   = None # 'data/12-5/rx_gm_board_12-5.txt'

	return init_gm, init_R, init_T, collected_dots, local_fname, abs_fname

if __name__ == '__main__':
	init_gm, init_R, init_T, collected_dots, local_fname, gm_fname = getTXInitData()
	# init_gm, init_R, init_T, collected_dots, local_fname, gm_fname = getRXInitData()

	dots_used, R, T, init_dir, init_point, m1_norm, m1_point, m1_axis, m2_norm, m2_point, m2_axis, val_params = \
			findRotAndTransParams(init_gm, init_R, init_T, collected_dots,
									modify_R = False, modify_T = False,
									modify_init_dir = False, modify_init_point = False,
									modify_m1_norm = False, modify_m1_point = False, modify_m1_axis = False,
									modify_m2_norm = False, modify_m2_point = False, modify_m2_axis = False,
									modify_linear_val_params = False, modify_quadratic_val_params = False,
									low_error_size = None)

	print('Rotation Matrix:')
	print(R)
	print('Rotation Angles:   ', R.getAngles())
	print('Translation Vector:', T)
	print('Initial direction: ', init_dir)
	print('Initial dir angles:', init_dir.getAngles())
	print('Initial point:     ', init_point)
	print('M1 norm:           ', m1_norm)
	print('M1 norm angles:    ', m1_norm.getAngles())
	print('M1 point:          ', m1_point)
	print('M1 axis:           ', m1_axis)
	print('M1 axis angles:    ', m1_axis.getAngles())
	print('M2 norm:           ', m2_norm)
	print('M2 norm angles:    ', m2_norm.getAngles())
	print('M2 point:          ', m2_point)
	print('M2 axis:           ', m2_axis)
	print('M2 axis angles:    ', m2_axis.getAngles())
	print('Convert val params:', val_params)
	print('')

	# Find rotation and translation that best matches data
	opt_gm = GM(init_dir, init_point, Plane(m1_norm, m1_point), m1_axis, Plane(m2_norm, m2_point), m2_axis, mirror_thickness = init_gm.mirror_thickness, val_params = val_params)

	if local_fname is not None:
		outputGM(opt_gm, local_fname)

	opt_gm = opt_gm.move(R, T)
	if gm_fname is not None:
		outputGM(opt_gm, gm_fname)

	# Generate dots
	gen_dots = generateDots(opt_gm, dots_used, Plane(Vec(0.0, 0.0, 1.0), Vec(0.0, 0.0, 0.0)))

	# Calculate the error between the dots
	calculateError(dots_used, gen_dots)

	# Draw collected dots and the generated dots
	# drawDots(dots_used, gen_dots)

	rangeOfLaunchPoint(opt_gm)
