
import numpy as np
import cv2
import math
import copy
import random

from scipy.optimize import least_squares, fsolve

# TODO make everything in meters

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
		return math.acos(self.dot(vec) / self.mag() / vec.mag())

	def signedAngle(self, vec, norm):
		theta = self.angle(vec)
		cross = self.cross(vec)

		if norm.dot(cross) < 0:
			return -theta
		return theta

	def cross(self, vec):
		return Vec(self.y * vec.z - self.z * vec.y, self.z * vec.x - self.x * vec.z, self.x * vec.y - self.y * vec.x)

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

	def next(self):
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

		self.theta, self.alpha, self.beta = None, None, None

	def mult(self, vec):
		return Vec(sum(a * b for a, b in zip(vec, self.vals[0])),
					sum(a * b for a, b in zip(vec, self.vals[1])),
					sum(a * b for a, b in zip(vec, self.vals[2])))

	def getAngles(self):
		if self.theta is None and self.alpha is None and self.beta is None:
			raise Exception('Cannot compute angle from rot matrix yet')

		return self.theta, self.alpha, self.beta

	def __str__(self):
		return '\n'.join(map(lambda row: '[' + ', '.join(map(str, row)) + ']', self.vals))

def quatFromAngle(theta, alpha, beta):
	v = vecFromAngle(alpha, beta).mult(math.sin(theta))
	q = Quat(math.cos(theta), v.x, v.y, v.z)

	q.theta = theta
	q.alpha = alpha
	q.beta  = beta

	return q

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

		m.theta, m.alpha, m.beta = self.theta, self.alpha, self.beta
		return m

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

def gmValToRadian(gm_val):
	return (40.0 / pow(2.0, 16.0) * gm_val - 20.0) * math.pi / 180.0

def getGMFromCAD(init_dir, init_point):
	# gm = GM(init_dir, init_point,
	# 		Plane(Vec(-0.183253086052, 0.68303422983, -0.707023724721), Vec(4.33404917057, 2.9865539516, 1.50005588143)),
	# 		Vec(0.965927222809, 0.258813833165, 0.0),
	# 		Plane(Vec(0.793391693847, -0.608711442422, 0.0), Vec(4.02024772148, 4.46917730708, 1.41303253915)),
	# 		Vec(0.0, 0.0, -1.0))

	gm = GM(init_dir, init_point,
			Plane(Vec(-0.68303422983, -0.707023724721,  0.183253086052), Vec(-2.9865539516, 1.50005588143, -4.33404917057)),
			Vec(-0.258813833165, 0.0, -0.965927222809),
			Plane(Vec(0.608711442422, 0.0, -0.793391693847), Vec(-4.46917730708, 1.41303253915, -4.02024772148)),
			Vec(0.0, 1.0, 0.0))

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

		vec = Vec(*map(float, (x, y, z)))

		if token not in vals:
			raise Exception('Unexpected token: ' + str(token))

		vals[token] = vec

	m1 = Plane(vals['m1_norm'], vals['m1_point'])
	m2 = Plane(vals['m2_norm'], vals['m2_point'])

	gm = GM(vals['input_dir'], vals['input_point'], m1, vals['m1_axis'], m2, vals['m2_axis'])

	return gm

class GM:
	def __init__(self, init_dir, init_point, m1, a1, m2, a2):
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

	# Creates a new GM that is 1) rotated using the supplied rotation matrix, and 2) translated by trans_vec
	def move(self, rot_matrix, trans_vec):
		rot = lambda x: rot_matrix.mult(x)
		rot_and_trans = lambda x: rot_matrix.mult(x) + trans_vec
		return GM(rot(self.init_dir), rot_and_trans(self.init_point),
					Plane(rot(self.m1.norm), rot_and_trans(self.m1.point)), rot(self.a1),
					Plane(rot(self.m2.norm), rot_and_trans(self.m2.point)), rot(self.a2))

	def getOutput(self, gm1_val, gm2_val, check_intersection = False):
		this_m1 = self.m1.rotate(self.a1, gmValToRadian(gm1_val))
		this_m2 = self.m2.rotate(self.a2, gmValToRadian(gm2_val))

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
	for a in xrange(-brush_size, brush_size + 1, 1):
		for b in xrange(-brush_size, brush_size + 1, 1):
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
	if filename == 'data/2-14/data_2-14.txt':
		corners = [Vec(0.0, 0.0, 0.0), Vec(666.0, 0.0, 0.0), Vec(0.0, 481.0, 0.0), Vec(666.0, 481.0, 0.0)]

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

		dots.append(this_dot)

	if verbose and behind_wall > 0:
		print behind_wall, 'out of', len(collected_dots), 'are behind the wall'

	if verbose:
		return dots
	else:
		return dots, is_dot_intersection_neg

# Find GM Param functions

# Finds the orientation and position of a given GM that minimizes the error between observed and calculated dots.
def makeRotAndTransFunction(gm, dots, wall_plane = Plane(Vec(0.0, 0.0, 1.0), Vec(0.0, 0.0, 0.0)),
							R_static = None, T_static = None, modify_init_dir = True, modify_init_point = True,
							modify_m1_norm = True, modify_m1_point = True, modify_m1_axis = True,
							modify_m2_norm = True, modify_m2_point = True, modify_m2_axis = True,
							error_type = 'distance'):
	if error_type not in ['distance', 'difference']:
		raise Exception('Invalid error_type')

	def eq(vs):
		# Get values from solver
		x = 0
		if R_static == None:
			r_theta, r_alpha, r_beta = vs[x : x+3]
			x += 3

			R = quatFromAngle(r_theta, r_alpha, r_beta).toRotMatrix()
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

		# Move the GM based on the input from the solver
		rt_gm = this_gm.move(R, T)

		# Calculate the error between the collected and generated data
		no_intersection = any(not rt_gm.getOutput(dot.gmh, dot.gmv, check_intersection = True) for dot in dots)

		rv = []
		for dot in dots:
			p, d = rt_gm.getOutput(dot.gmh, dot.gmv)
			intersection_point, neg = wall_plane.intersect(p, d)

			if no_intersection is True or neg is None or neg is True:
				if error_type == 'difference':
					rv += [float('inf'), float('inf'), float('inf')]
				elif error_type == 'distance':
					rv.append(float('inf'))
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
							modify_m2_norm = True, modify_m2_point = True, modify_m2_axis = True):
	if all(not v for v in [modify_R, modify_T, modify_init_dir, modify_init_point,
			modify_m1_norm, modify_m1_point, modify_m1_axis,
			modify_m2_norm, modify_m2_point, modify_m2_axis]):
		return R, T, gm.init_dir, gm.init_point, gm.m1.norm, gm.m1.point, gm.a1, gm.m2.norm, gm.m2.point, gm.a2

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
								error_type = 'distance')

	init_guess = []
	if modify_R:
		init_guess += list(R.getAngles())

	if modify_T:
		init_guess += [T.x, T.y, T.z]

	if modify_init_dir:
		init_guess += list(gm.init_dir.getAngles())
	if modify_init_point:
		init_guess += [gm.init_point.x, gm.init_point.y, gm.init_point.z]

	if modify_m1_norm:
		init_guess += list(gm.m1.norm.getAngles())
	if modify_m1_point:
		init_guess += [gm.m1.point.x, gm.m1.point.y, gm.m1.point.z]
	if modify_m1_axis:
		init_guess += list(gm.a1.getAngles())

	if modify_m2_norm:
		init_guess += list(gm.m2.norm.getAngles())
	if modify_m2_point:
		init_guess += [gm.m2.point.x, gm.m2.point.y, gm.m2.point.z]
	if modify_m2_axis:
		init_guess += list(gm.a2.getAngles())

	rv = least_squares(f, tuple(init_guess))

	cost = rv.cost

	x = 0
	if R_static == None:
		r_theta, r_alpha, r_beta = rv.x[x : x+3]
		x += 3

		print 'Rotation matrix angles:', r_theta, r_alpha, r_beta

		R = quatFromAngle(r_theta, r_alpha, r_beta).toRotMatrix()
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

		print 'Init dir angles:', id_a, id_b
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

	return R, T, init_dir, init_point, m1_norm, m1_point, m1_axis, m2_norm, m2_point, m2_axis

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

	print 'Average Error:', sum(errors) / float(len(errors))
	print 'Max Error:    ', max(errors)

if __name__ == '__main__':
	collected_dots = getDotsFromFile('data/2-14/data_2-14.txt')

	# Initialize GM
	init_gm = getGMFromCAD(vecFromAngle(-0.437406371192, 1.56966805448), Vec(-2.98952866863, 0.500058946231, -4.3355745444))

	init_R = quatFromAngle(0.010526284225, -0.728423588802, 0.279838973767).toRotMatrix()
	init_T = Vec(378.122802504, -128.186853293, 858.533878011)

	R, T, init_dir, init_point, m1_norm, m1_point, m1_axis, m2_norm, m2_point, m2_axis = \
			findRotAndTransParams(init_gm, init_R, init_T, collected_dots,
									modify_R = True, modify_T = True,
									modify_init_dir = True, modify_init_point = True,
									modify_m1_norm = True, modify_m1_point = True, modify_m1_axis = True,
									modify_m2_norm = True, modify_m2_point = True, modify_m2_axis = True)

	print 'Rotation Matrix:'
	print R
	print 'Rotation Angles:   ', R.getAngles()
	print 'Translation Vector:', T
	print 'Initial direction: ', init_dir
	print 'Initial dir angles:', init_dir.getAngles()
	print 'Initial point:     ', init_point
	print 'M1 norm:           ', m1_norm
	print 'M1 norm angles:    ', m1_norm.getAngles()
	print 'M1 point:          ', m1_point
	print 'M1 axis:           ', m1_axis
	print 'M1 axis angles:    ', m1_axis.getAngles()
	print 'M2 norm:           ', m2_norm
	print 'M2 norm angles:    ', m2_norm.getAngles()
	print 'M2 point:          ', m2_point
	print 'M2 axis:           ', m2_axis
	print 'M2 axis angles:    ', m2_axis.getAngles()

	# Find rotation and translation that best matches data
	opt_gm = GM(init_dir, init_point, Plane(m1_norm, m1_point), m1_axis, Plane(m2_norm, m2_point), m2_axis)
	opt_gm = opt_gm.move(R, T)

	outputGM(opt_gm, 'gm_2-19.txt')

	# Generate dots
	gen_dots = generateDots(opt_gm, collected_dots, Plane(Vec(0.0, 0.0, 1.0), Vec(0.0, 0.0, 0.0)))

	# Calculate the error between the dots
	calculateError(collected_dots, gen_dots)

	# Draw collected dots and the generated dots
	drawDots(collected_dots, gen_dots)
