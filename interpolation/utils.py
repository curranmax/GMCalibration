import math

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
					self.x * vec.y - self.y * vec.x).mult(1.0 / self.mag() / vec.mag())

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

	def totalDifference(self, rm):
		return sum(abs(a - b) for r1, r2 in zip(self.vals, rm.vals) for a, b in zip(r1, r2))

	def toQuat(self):
		qw = math.sqrt(1.0 + self.vals[0][0] + self.vals[1][1] + self.vals[2][2]) / 2.0
		qx = (self.vals[2][1] - self.vals[1][2]) / (4.0 * qw)
		qy = (self.vals[0][2] - self.vals[2][0]) / (4.0 * qw)
		qz = (self.vals[1][0] - self.vals[0][1]) / (4.0 * qw)

		return Quat(qw, qx, qy, qz)

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

	def mult(self, v):
		return Quat(self.w * v, self.x * v, self.y * v, self.z * v)

	def __add__(self, quat):
		return Quat(self.w + quat.w,
					self.x + quat.x,
					self.y + quat.y,
					self.z + quat.z)

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

	def dist(self, v):
		return self.norm.dot(v - self.point)

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

class VRDataPoint:
	def __init__(self, time, tvec, rot_mtx, quat, tracking_method = None, tx_vals = None, rx_vals = None):
		self.time = time

		self.tvec = tvec
		self.rot_mtx = rot_mtx
		self.quat = quat

		self.tracking_method = tracking_method

		self.tx_vals = tx_vals
		self.rx_vals = rx_vals

# Returns the raw data from the VR headset
def getVRData(fname):
	f = open(fname, 'r')

	data = []
	for line in f:
		spl = line.split()

		time = int(spl[0])

		tvec = Vec(*list(map(float, spl[2:5])))

		if spl[5] == 'qWXYZ:':
			w, x, y, z = list(map(float, spl[6:10]))
		elif spl[5] == 'qXYZW:':
			x, y, z, w = list(map(float, spl[6:10]))
		else:
			raise Exception('Invalid file format')

		tracking_method = spl[11]

		tx_vals = list(map(int, spl[13:15]))
		rx_vals = list(map(int, spl[16:18]))

		quat = Quat(w, x, y, z)

		rot_mtx = quat.toRotMatrix()

		data.append(VRDataPoint(time, tvec, rot_mtx, quat, tracking_method = tracking_method, tx_vals = tx_vals, rx_vals = rx_vals))

	return data


def outputVRData(fname, vr_data):
	f = open(fname, 'w')

	for dp in vr_data:
		vals = [dp.time, 'XYZ:', dp.tvec.x, dp.tvec.y, dp.tvec.z, 'qWXYZ:', dp.quat.w, dp.quat.x, dp.quat.y, dp.quat.z]
		f.write(' '.join(map(str, vals)) + '\n')

	f.close()

def reduceVRData(vr_data, k):
	if len(vr_data) % k != 0:
		raise Exception('Can\'t reduce given data by given k: len(vr_data) = ' + str(len(vr_data)) + ', k = ' + str(k))

	reduced_data = []
	for i in range(0, len(vr_data), k):
		this_data = vr_data[i : i+k]

		this_time = min(dp.time for dp in this_data)
		this_tvec = sum((dp.tvec for dp in this_data), Vec(0.0, 0.0, 0.0)).mult(1.0 / float(len(this_data)))
		this_quat = computeAvgQuat([dp.quat for dp in this_data])

		reduced_data.append(VRDataPoint(this_time, this_tvec, this_quat.toRotMatrix(), this_quat, None, None))

	return reduced_data

def computeAvgQuat(quats):
	avg_quat = sum((q for q in quats), Quat(0.0, 0.0, 0.0, 0.0)).mult(1.0 / float(len(quats)))
	return avg_quat.mult(1.0 / avg_quat.mag())

class VoltDataPoint:
	def __init__(self, tx_gm1, tx_gm2, rx_gm1, rx_gm2, dist = None):
		self.tx_gm1 = tx_gm1
		self.tx_gm2 = tx_gm2
		self.rx_gm1 = rx_gm1
		self.rx_gm2 = rx_gm2

		self.dist = dist

def getVoltData(fname):
	f = open(fname, 'r')

	first = True
	cols = {'TX1': None, 'TX2': None, 'RX1': None, 'RX2': None, 'Dist': None}

	data = []
	for line in f:
		spl = line.split()
		if first:
			first = False

			if len(spl) not in [4, 5] or (len(spl) == 4 and any(cname not in spl for cname in cols if cname != 'Dist')) or (len(spl) == 5 and any(cname not in spl for cname in cols)):
				raise Exception('Invalid column names: ' + ', '.join(spl))

			for cname in cols:
				cols[cname] = spl.index(cname)
		else:
			tx1, tx2, rx1, rx2 = list(map(int, (spl[cols['TX1']], spl[cols['TX2']], spl[cols['RX1']], spl[cols['RX2']])))

			if cols['Dist'] is None:
				dist = None
			else:
				dist = float(spl[cols['Dist']])

			data.append(VoltDataPoint(tx1, tx2, rx1, rx2, dist = dist))

	return data

