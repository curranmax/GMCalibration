
import numpy as np
import cv2
import math

from scipy.optimize import least_squares

# TODO make everything in meters

# Angles are in radians
def vecFromAngle(alpha, beta):
	return Vec(math.cos(alpha) * math.cos(beta),
			   math.sin(beta),
			   math.sin(alpha) * math.cos(beta))

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

	def mult(self, vec):
		return Vec(sum(a * b for a, b in zip(vec, self.vals[0])),
					sum(a * b for a, b in zip(vec, self.vals[1])),
					sum(a * b for a, b in zip(vec, self.vals[2])))

	def __str__(self):
		return '\n'.join(map(lambda row: '[' + ', '.join(map(str, row)) + ']', self.vals))

def quatFromAngle(theta, alpha, beta):
	v = vecFromAngle(alpha, beta).mult(math.sin(theta))

	return Quat(math.cos(theta), v.x, v.y, v.z)

class Quat:
	def __init__(self, w, x, y, z):
		self.w = float(w)
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)

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

		return Matrix(a, b, c, d, e, f, g, h, i)

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
			Plane(Vec(-0.707023724721, 0.68303422983, 0.183253086052), Vec(1.50005588143, 2.9865539516, -4.33404917057)),
			Vec(0.0, 0.258813833165, -0.965927222809),
			Plane(Vec(0.0, -0.608711442422, -0.793391693847), Vec(1.41303253915, 4.46917730708, -4.02024772148)),
			Vec(-1.0, 0.0, 0.0))

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

	img = np.zeros((height, width, 3), np.uint8)

	for dots, channel in [(collected_dots, 0), (gen_dots, 1)]:
		if dots == None:
			continue

		print min(d.loc.x for d in dots), max(d.loc.x for d in dots)
		print min(d.loc.y for d in dots), max(d.loc.y for d in dots)

		for d in dots:
			# Convert 3d location to pixel
			if abs(d.loc.z) > 0.0000001:
				raise Exception('Assumed that all dots are on the plane z=0')

			yp = int((d.loc.y - origin_point.y) * conversion_factor)
			xp = int((d.loc.x - origin_point.x) * conversion_factor)

			for a, b in brush:
				if yp + a >= 0 and yp + a < height and xp + b >= 0 and xp + b < width:
					img[yp + a][xp + b][channel] = 255

	cv2.imwrite(out_fname, img)

def getDotsFromFile(filename):
	f = open(filename, 'r')

	cols = None
	col_inds = {'GMH': None, 'GMV': None, 'X': None, 'Y': None, 'Z': None}
	dots = []
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
			dots.append(this_dot)
	return dots

def generateDots(gm, gm_vals, wall_plane):
	dots = []
	for gm1_val, gm2_val in gm_vals:
		if not gm.getOutput(gm1_val, gm2_val, check_intersection = True):
			raise Exception('No intersection for gm values (' + str(gm1_val) + ', ' + str(gm2_val) + ')')

		p, d = gm.getOutput(gm1_val, gm2_val)

		ip, neg = wall_plane.intersect(p, d)

		# if neg is None or neg is False:
		# 	print 'Error GM behind wall:', p, d, ip, neg

		dots.append(Dot(gm1_val, gm2_val, ip))

	return dots

# Find GM Param functions

# Finds the orientation and position of a given GM that minimizes the error between observed and calculated dots.
def makeRotAndTransFunction(gm, dots, wall_plane = Plane(Vec(0.0, 0.0, 1.0), Vec(0.0, 0.0, 0.0))):
	def eq(vs):
		r_theta, r_alpha, r_beta, t_x, t_y, t_z = vs

		rt_gm = gm.move(quatFromAngle(r_theta, r_alpha, r_beta).toRotMatrix(), Vec(t_x, t_y, t_z))

		rv = []
		for dot in dots:
			p, d = rt_gm.getOutput(dot.gmh, dot.gmv)

			calc_loc, neg = wall_plane.intersect(p, d)

			if neg:
				rv += [float('inf'), float('inf'), float('inf')]
			else:
				rv += [calc_loc.x - dot.loc.x, calc_loc.y - dot.loc.y, calc_loc.z - dot.loc.z]

		return rv

	return eq

def findRotAndTransParams(gm, dots, wall_plane = Plane(Vec(0.0, 0.0, 1.0), Vec(0.0, 0.0, 0.0))):
	f = makeRotAndTransFunction(gm, dots, wall_plane)

	rv = least_squares(f, (0.0, 0.0, 0.0, 0.0, 0.0, 100.0))

	cost = rv.cost
	r_theta, r_alpha, r_beta, t_x, t_y, t_z = rv.x

	print r_theta, r_alpha, r_beta

	R = quatFromAngle(r_theta, r_alpha, r_beta).toRotMatrix()
	T = Vec(t_x, t_y, t_z)

	return R, T, cost

if __name__ == '__main__':
	collected_dots = getDotsFromFile('data/2-14/data_2-14.txt')

	# Initialize GM
	init_gm = getGMFromCAD(Vec(1.0, 0.0, 0.0), Vec(0.50005588143, 2.9865539516, -4.33404917057))

	R, T, cost = findRotAndTransParams(init_gm, collected_dots)

	print 'Rotation Matrix:',
	print R
	print 'Translation Vector:', T
	print 'Scipy least_squares cost:', cost

	# Find rotation and translation that best matches data
	opt_gm = init_gm.move(R, T)

	# Generate dots
	gen_dots = generateDots(opt_gm, [(d.gmh, d.gmv) for d in collected_dots], Plane(Vec(0.0, 0.0, 1.0), Vec(0.0, 0.0, 0.0)))

	# Draw collected dots and the generated dots
	drawDots(collected_dots, gen_dots)
