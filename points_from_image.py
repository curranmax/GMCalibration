
import numpy as np
from scipy.optimize import fsolve
import cv2
import math
import os
from collections import defaultdict
import itertools

class CameraCalibration:
	def __init__(self, fx, fy, cx, cy, k1, k2, k3, p1, p2):
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy

		self.k1 = k1
		self.k2 = k2
		self.k3 = k3

		self.p1 = p1
		self.p2 = p2

	def npArrays(self):
		mtx = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
		dist = np.array([[self.k1, self.k2, self.p1, self.p2, self.k3]])

		return mtx, dist

	def __str__(self):
		return '(' + ', '.join(map(lambda x: x[0] + ': ' + str(x[1]), (('fx', self.fx), ('fy', self.fy), ('cx', self.cx), ('cy', self.cy), ('k1', self.k1), ('k2', self.k2), ('k3', self.k3), ('p1', self.p1), ('p2', self.p2)))) + ')'

def outputCalibration(cal, image_pairs, fname):
	f = open(fname, 'w')

	f.write('fx ' + str(cal.fx) + '\n')
	f.write('fy ' + str(cal.fy) + '\n')
	f.write('cx ' + str(cal.cx) + '\n')
	f.write('cy ' + str(cal.cy) + '\n')
	f.write('k1 ' + str(cal.k1) + '\n')
	f.write('k2 ' + str(cal.k2) + '\n')
	f.write('k3 ' + str(cal.k3) + '\n')
	f.write('p1 ' + str(cal.p1) + '\n')
	f.write('p2 ' + str(cal.p2) + '\n')

	for i, ip in enumerate(image_pairs):
		f.write('img:' + str(i + 1))

		# cal_fname
		f.write(' cal_fname:' + str(ip.cal_fname))

		# dot_fname
		f.write(' dot_fname:' + str(ip.dot_fname))

		f.write(' filter_fname:' + str(ip.filter_fname))

		# rvec
		f.write(' rvec:')
		if ip.rvec is None:
			f.write('None')
		else:
			f.write(','.join(map(str, ip.rvec.flatten())))

		# tvec
		f.write(' tvec:')
		if ip.tvec is None:
			f.write('None')
		else:
			f.write(','.join(map(str, ip.tvec.flatten())))

		if len(ip.missing_dots) > 0:
			f.write(' missing_dots:' + ';'.join(map(lambda x: str(x[0]) + ',' + str(x[1]), ip.missing_dots)))

		f.write('\n')

	f.close()

def inputCalibration(fname):
	f = open(fname, 'r')

	vals = {'fx': None, 'fy': None, 'cx': None, 'cy': None, 'k1': None, 'k2': None, 'k3': None, 'p1': None, 'p2': None}
	imgs = []

	for line in f:
		spl = line.split()
		token = spl[0]

		if token in vals and vals[token] == None:
			val = float(spl[1])
			vals[token] = val
		elif token[:3] == 'img':
			cal_fname = None
			dot_fname = None
			filter_fname = None
			tvec = None
			rvec = None
			missing_dots = []
			for k, v in map(lambda x: x.split(':'), spl):
				if k == 'cal_fname':
					cal_fname = v
				if k == 'dot_fname':
					dot_fname = v
				if k == 'filter_fname' and v != 'None':
					filter_fname = v
				if k == 'tvec':
					if v != 'None':
						tvec = np.array([[x] for x in map(float, v.split(','))])
				if k == 'rvec':
					if v != 'None':
						rvec = np.array([[x] for x in map(float, v.split(','))])
				if k == 'missing_dots':
					missing_dots = map(lambda x: tuple(map(int, x.split(','))) ,v.split(';'))
			
			img = ImagePair(cal_fname, dot_fname)
			img.filter_fname = filter_fname
			img.tvec = tvec
			img.rvec = rvec
			img.missing_dots = missing_dots

			imgs.append(img)
		else:
			if token not in vals:
				raise Exception('Unexpected token: ' + token)
			if vals[token] != None:
				raise Exception('Duplicate token: ' + token)

	if any(v == None for k, v in vals.iteritems()):
		raise Exception('Missing values: ' + ', '.join(k for k, v in vals.iteritems() if v == None))

	return CameraCalibration(**vals), imgs

class ImagePair:
	def __init__(self, cal_fname, dot_fname):
		self.cal_fname = cal_fname
		self.dot_fname = dot_fname

		self.filter_fname = None

		self.tvec = None
		self.rvec = None

		self.missing_dots = []

def getImagePairs(folder):
	fnames = defaultdict(list)
	for f in os.listdir(folder):
		token = f[:f.find('_')]
		n = int(f[f.find('_') + 1 : f.find('.')])

		if token in ['cal', 'dot']:
			fnames[n].append(f)

	image_pairs = []
	for k in fnames:
		if len(fnames[k]) != 2:
			continue

		cal_fname, dot_fname = None, None
		for fname in fnames[k]:
			if 'cal' == fname[:3]:
				cal_fname = os.path.join(folder, fname)
			if 'dot' == fname[:3]:
				dot_fname = os.path.join(folder, fname)

		if cal_fname == None or dot_fname == None:
			continue
		image_pairs.append(ImagePair(cal_fname, dot_fname))

	return image_pairs

sample_image_fnames = ['data/sample/left01.jpg', 'data/sample/left02.jpg', 'data/sample/left03.jpg',
					   'data/sample/left04.jpg', 'data/sample/left05.jpg', 'data/sample/left06.jpg',
					   'data/sample/left07.jpg', 'data/sample/left08.jpg', 'data/sample/left12.jpg',
					   'data/sample/left13.jpg', 'data/sample/left14.jpg']

def calibrateCamera(image_pairs, save_cal_images = True):
	# Termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	objp = np.zeros((6 * 6, 3), np.float32)
	objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)

	obj_points = [] # 3d points of the plane
	img_points = [] # 2d points in the image

	working_inds = []

	for ip_ind, ip in enumerate(image_pairs):
		print 'Running image', ip_ind + 1, 'of', len(image_pairs)

		img = cv2.imread(ip.cal_fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the corners
		ret, corners = cv2.findChessboardCorners(gray, (6, 6), None)

		# Changes the order of the corners for these images to match the others
		if ip.cal_fname in ['data/1-18/cal_4.JPG', 'data/1-18/cal_5.JPG', 'data/1-18/cal_8.JPG','data/1-18/cal_9.JPG', 'data/1-18/cal_10.JPG',
				'data/1-21/cal_15.JPG', 'data/1-21/cal_7.JPG', 'data/1-21/cal_8.JPG']:
			grid_corners = dict()
			for i, c in enumerate(corners):
				x = i % 6
				y = int(i / 6)

				grid_corners[(x, y)] = c

			new_corners = []
			for x in xrange(0, 6, 1):
				for y in xrange(5, -1, -1):
					new_corners.append(grid_corners[(x, y)])

			corners = np.array(new_corners)

		if ret == True:
			corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

			working_inds.append(ip_ind)
			
			obj_points.append(objp)
			img_points.append(corners2)

			if save_cal_images:
				img = cv2.drawChessboardCorners(img, (6, 6), corners2, ret)

				d = os.path.dirname(ip.cal_fname)

				bn = os.path.basename(ip.cal_fname)
				n = int(bn[bn.find('_') + 1:bn.find('.')])

				cal_img_name = os.path.join(d, 'calimg_' + str(n) + '.jpg')

				cv2.imwrite(cal_img_name, img)

	# ret - whethere the calibration was sucessful
	# mtx - intrinsic camera matrix
	# dist - distortion parameters
	# rvecs - rotation vectors for each image
	# tvecs - translation vectors for each image
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

	fx = mtx[0][0]
	fy = mtx[1][1]
	cx = mtx[0][2]
	cy = mtx[1][2]

	k1, k2, p1, p2, k3 = dist[0]

	print fx, fy
	print cx, cy

	print k1, k2, k3
	print p1, p2

	cal = CameraCalibration(fx, fy, cx, cy, k1, k2, k3, p1, p2)

	print rvecs
	print tvecs

	for j, i in enumerate(working_inds):
		image_pairs[i].rvec = rvecs[j]
		image_pairs[i].tvec = tvecs[j]

	return cal

def tmp():
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	for fname in ['data/sample/chessboard-2.jpg']:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the corners
		ret, corners = cv2.findChessboardCorners(gray, (4, 7), None)

		if ret == True:
			corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

			img = cv2.drawChessboardCorners(img, (4, 7), corners2, ret)
			cv2.imshow('img',img)
			cv2.waitKey(5000)
		else:
			print 'Can\'t find corners'
			cv2.imshow('img',img)
			cv2.waitKey(500)

	cv2.destroyAllWindows()

def projectLib(real_points, cal, rvec, tvec, show_image = True):
	mtx, dist = cal.npArrays()

	points_proj, _ = cv2.projectPoints(real_points, rvec, tvec, mtx, dist)
	points_proj = np.squeeze(points_proj, axis = 1)

	print points_proj

	if show_image:
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		img = cv2.imread(sample_image_fnames[0])
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the corners
		ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

		if ret == True:
			corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

			img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)


			# endpoint = np.array([[points_proj[0][0] + 0.0, points_proj[0][1] + 10.0],])
			img = cv2.line(img, tuple(map(int, points_proj.ravel())), tuple(map(int, points_proj.ravel())), (0, 0, 255), 5)

			cv2.imshow('img',img)
			cv2.waitKey()
		cv2.destroyAllWindows()

	return points_proj

def projectSimp(real_points, cal, rvec, tvec):
	rot_mtx, _ = cv2.Rodrigues(rvec)

	points_proj = []
	for rp in real_points:
		rt_point = np.matmul(rot_mtx, rp) + tvec.flatten()

		xp = rt_point[0] / rt_point[2]
		yp = rt_point[1] / rt_point[2]

		r = math.sqrt(pow(xp, 2.0) + pow(yp, 2.0))

		xpp = xp * (1.0 + cal.k1 * pow(r, 2.0) + cal.k2 * pow(r, 4.0) + cal.k3 * pow(r, 6.0)) + 2.0 * cal.p1 * xp * yp + cal.p2 * (pow(r, 2.0) + 2.0 * pow(xp, 2.0))
		ypp = yp * (1.0 + cal.k1 * pow(r, 2.0) + cal.k2 * pow(r, 4.0) + cal.k3 * pow(r, 6.0)) + cal.p1 * (pow(r, 2.0) + 2.0 * pow(yp, 2.0)) + 2.0 * cal.p2 * xp * yp

		u = cal.fx * xpp + cal.cx
		v = cal.fy * ypp + cal.cy

		points_proj.append([u, v])

	return np.array(points_proj)

def unproject(image_points, z_values, cal, rvec, tvec):
	real_points = []
	for ip, z in zip(image_points, z_values):
		def eq(vs):
			x, y = vs

			rps = np.array([[x, y, z]])

			rv = projectSimp(rps, cal, rvec, tvec)
			calc_ix, calc_iy = rv[0]

			return [ip[0] - calc_ix, ip[1] - calc_iy]

		rx, ry = fsolve(eq, (0.0, 0.0))

		real_points.append([rx, ry, z])

	return np.array(real_points)

def getRedFilter(img_pair):
	if img_pair.tvec is None or img_pair.rvec is None:
		return

	dot_img = cv2.imread(img_pair.dot_fname)

	height, width, _ = dot_img.shape

	filter_img = np.zeros((height, width), np.uint8)

	low_val = 10
	high_val = 100

	for x in range(height):
		for y in range(width):
			b, g, r = dot_img[x][y]

			if b <= low_val and g <= low_val and r >= high_val:
				filter_img[x][y] = 255

	d = os.path.dirname(img_pair.dot_fname)

	bn = os.path.basename(img_pair.dot_fname)
	n = int(bn[bn.find('_') + 1:bn.find('.')])

	img_pair.filter_fname = os.path.join(d, 'filter_' + str(n) + '.jpg')

	print img_pair.filter_fname

	cv2.imwrite(img_pair.filter_fname, filter_img)

class Dot:
	def __init__(self):
		self.pixels = set()
		self.center = None

		self.coord = None
		self.gm_vals = None

	def forceAdd(self, x, y):
		self.pixels.add((x, y))

	def add(self, x, y):
		if any((x + a, y + b) in self.pixels for a, b in itertools.product([-2, -1, 0, 1, 2], repeat = 2)):
			self.pixels.add((x, y))
			return True
		return False

	def merge(self, dot):
		self.pixels = self.pixels.union(dot.pixels)

	def size(self):
		return len(self.pixels)

# Finds the the value t s.t. the distance between f(t) = (xm, ym) * t + (xb, yb) and (x, y) is minimized
# Returns the value of t, f(t), and the distance between f(t) and (x, y)
def minDistToLine(x, y, xm, ym, xb, yb):
	t = (xm * (x - xb) + ym * (y - yb)) / (pow(xm, 2.0) + pow(ym, 2.0))

	fx = xm * t + xb
	fy = ym * t + yb

	d = math.sqrt(pow(fx - x, 2.0) + pow(fy - y, 2.0))

	return t, fx, fy, d

def getRedDots(img_pair):
	if img_pair.tvec is None or img_pair.rvec is None:
		return

	dot_img = cv2.imread(img_pair.dot_fname)
	filter_img = cv2.imread(img_pair.filter_fname, cv2.IMREAD_GRAYSCALE)

	height, width, _ = dot_img.shape

	filter_pixels = []

	for x in range(height):
		for y in range(width):
			if filter_img[x][y] > 200:
				filter_pixels.append((x, y))

	# Find the connected components ("dots")
	dots = []
	for x, y in filter_pixels:

		this_dots = [d for d in dots if d.add(x, y)]

		if len(this_dots) == 0:
			new_dot = Dot()
			new_dot.forceAdd(x, y)
			dots.append(new_dot)
		if len(this_dots) == 1:
			pass
		if len(this_dots) > 1:

			this_dot = this_dots[0]
			for dot in this_dots[1:]:
				this_dot.merge(dot)
				dots.remove(dot)


	# Find the center of each dot. The center is defined as the brightest pixel
	for dot in dots:
		bright_value = None
		bright_pixel = None

		for x, y in dot.pixels:
			v = math.sqrt(sum(pow(v, 2.0) for v in dot_img[x][y]))
			if bright_value == None or bright_value < v:
				bright_value = v
				bright_pixel = (x, y)

		dot.center = map(float, bright_pixel)

	missing_dots = img_pair.missing_dots

	min_h, max_h = -12, 8
	min_v, max_v = -14, 4

	valid_dot_coords = []
	num_center_line = 0
	for h in xrange(min_h, max_h + 1):
		for v in xrange(min_v, max_v + 1):

			if (h, v) not in missing_dots:
				valid_dot_coords.append((h, v))

				if h == 0 or v == 0:
					num_center_line += 1

	if len(valid_dot_coords) != len(dots):
		raise Exception('Mismatching number of dots and coords')

	# Find the "median" dots to form the axis
	sorted_dots = list(dots)
	sorted_dots.sort(key = lambda x: x.size(), reverse = True)

	cl_dots = sorted_dots[:num_center_line]
	other_dots = sorted_dots[num_center_line:]

	# Separate into two groups with exactly one dot in both, s.t. each group are close to co-linear
	_, min_v_cl_dot = min((d.center[0], d) for d in cl_dots)
	_, max_v_cl_dot = max((d.center[0], d) for d in cl_dots)
	_, min_h_cl_dot = min((d.center[1], d) for d in cl_dots)
	_, max_h_cl_dot = max((d.center[1], d) for d in cl_dots)
	
	# Find "origin" dot, by finding the intersection between the lines form between the min and max points of each direction.
	hm = (max_h_cl_dot.center[0] - min_h_cl_dot.center[0], max_h_cl_dot.center[1] - min_h_cl_dot.center[1])
	hb = (min_h_cl_dot.center[0], min_h_cl_dot.center[1])

	vm = (max_v_cl_dot.center[0] - min_v_cl_dot.center[0], max_v_cl_dot.center[1] - min_v_cl_dot.center[1])
	vb = (min_v_cl_dot.center[0], min_v_cl_dot.center[1])

	a = [[hm[0], -vm[0]],
			[hm[1], -vm[1]]]
	b = [vb[0] - hb[0], vb[1] - hb[1]]

	c_ht, c_vt = np.linalg.solve(a, b)

	orign_point = (hm[0] * c_ht + hb[0],
					hm[1] * c_ht + hb[1])

	min_dist = None
	origin_dot  = None

	for d in cl_dots:
		this_dist = math.sqrt(pow(orign_point[0] - d.center[0], 2.0) + pow(orign_point[1] - d.center[1], 2.0))
		if min_dist == None or min_dist > this_dist:
			min_dist = this_dist
			origin_dot  = d

	# Find the vertical and horizontal axises
	vdots = []
	hdots = []

	for d in cl_dots:
		if d == origin_dot:
			hdots.append((d, c_ht))
			vdots.append((d, c_vt))
			continue

		ht, hx, hy, hd = minDistToLine(d.center[0], d.center[1], hm[0], hm[1], hb[0], hb[1])
		vt, vx, vy, vd = minDistToLine(d.center[0], d.center[1], vm[0], vm[1], vb[0], vb[1])

		if hd <= vd:
			hdots.append((d, ht))
		elif vd < hd:
			vdots.append((d, vt))

	# Assign coordinates to vdots and hdots
	hdots.sort(key = lambda x: x[1])
	hdots = [hd for hd, _ in hdots]

	vdots.sort(key = lambda x: x[1])
	vdots = [vd for vd, _ in vdots]

	h_origin_index = hdots.index(origin_dot)
	v_origin_index = vdots.index(origin_dot)

	for i, hd in enumerate(hdots):
		hd.coord = (i - h_origin_index, 0)

	for i, vd in enumerate(vdots):
		vd.coord = (0, i - v_origin_index)

	# Find the coordinates of the other dots

	dots_by_coord = {(x, y): None for x, y in valid_dot_coords}
	for d in hdots + vdots:
		dots_by_coord[d.coord] = d

	spiral_search =  sorted(dots_by_coord.keys(), key = lambda c: pow(c[0], 2) + pow(c[1], 2))

	for x, y in spiral_search:
		if (x, y) not in dots_by_coord or dots_by_coord[(x, y)] != None:
			continue

		for ax, ay in spiral_search:
			bx = x - ax
			by = y - ay

			if (ax, ay) in dots_by_coord and dots_by_coord[(ax, ay)] != None and (bx, by) in dots_by_coord and dots_by_coord[(bx, by)] != None:
				break

		approx_px = origin_dot.center[0] + (dots_by_coord[(ax, ay)].center[0] - origin_dot.center[0]) + (dots_by_coord[(bx, by)].center[0] - origin_dot.center[0])
		approx_py = origin_dot.center[1] + (dots_by_coord[(ax, ay)].center[1] - origin_dot.center[1]) + (dots_by_coord[(bx, by)].center[1] - origin_dot.center[1])

		if len(other_dots) == 0:
			raise Exception('Ran out of dots')

		min_dist = None
		min_dot  = None

		for d in other_dots:
			this_dist = math.sqrt(pow(d.center[0] - approx_px, 2) + pow(d.center[1] - approx_py, 2))
			if min_dist == None or this_dist < min_dist:
				min_dist = this_dist
				min_dot = d

		min_dot.coord = (x, y)
		dots_by_coord[(x, y)] = min_dot
		other_dots.remove(min_dot)

	# # Calculates the gm_vals for each dot
	# for d in dots:
	# 	d.gm_vals = (d.coord[0] * 2048 + 32768, d.coord[1] * 2048 + 32768)

	# # Checks that all dots have coord and gm_vals set
	# for d in dots:
	# 	if d.gm_vals is None or d.coord is None:
	# 		raise Exception('Didn\'t set the coordinates/gm_vals of a dot')

	# Temporarily draw image for debugging purposes
	tmp_img = np.zeros((height, width, 3), np.uint8)
	for (x, y) in dots_by_coord:
		d = dots_by_coord[(x, y)]
		if d == None:
			print 'Skipping: (' + str(x) + ', ' + str(y) + ')'
			continue
		if y == 3:
			c = [0, 0, 255]
		else:
			c = [0, 0, 0]
		for x, y in d.pixels:
			tmp_img[x][y] = c

	for hd in hdots:
		for x, y in hd.pixels:
			tmp_img[x][y] = [255, 0, 0]

	for vd in vdots:
		for x, y in vd.pixels:
			tmp_img[x][y] = [0, 255, 0]

	for x, y in origin_dot.pixels:
		tmp_img[x][y] = [255, 255, 255]

	cv2.imwrite('tmp.jpg', tmp_img)

	raise Exception('Quit')

	return dots

def computeAndSaveCalibration(image_folder, output_file):
	image_pairs = getImagePairs(image_folder)

	# Calibration
	cal = calibrateCamera(image_pairs)

	outputCalibration(cal, image_pairs, output_file)

def computeRedFilter(cal_file):
	cal, image_pairs = inputCalibration(cal_file)
	for img_pair in image_pairs:
		if img_pair.filter_fname == None:
			getRedFilter(img_pair)

	outputCalibration(cal, image_pairs, cal_file)

def computeRedDots(cal_file):
	cal, image_pairs = inputCalibration(cal_file)

	all_dots = []
	for img_pair in image_pairs:
		if img_pair.filter_fname != None and img_pair.filter_fname not in ['data/1-11/filter_6.jpg', 'data/1-11/filter_15.jpg']:
			print 'Getting Red Dots for image', img_pair.dot_fname
			this_dots = getRedDots(img_pair)
			all_dots.append((this_dots, img_pair))

	dots_by_coord = defaultdict(list)
	for dots, img_pair in all_dots:
		for d in dots:
			dots_by_coord[d.coord].append((d, img_pair.rvec, img_pair.tvec))

	all_dists = []
	dists_by_coord = defaultdict(list)
	for x, y in sorted(dots_by_coord):
		real_points = []
		for d, rvec, tvec in dots_by_coord[(x, y)]:
			image_points = np.array([list(d.center)])
			z_values = np.array([0.0])

			real_point = unproject(image_points, z_values, cal, rvec, tvec)[0]
			real_points.append(real_point)

		# Calculate "average" point

		avg_point = sum(real_points) / len(real_points)
		dists = [math.sqrt(sum(pow(rv - av, 2.0) for rv, av in zip(rp, avg_point))) for rp in real_points]
		all_dists += dists
		dists_by_coord[(x, y)] = sum(dists) / len(dists)

	print 'Max dist:', max(all_dists)
	print 'Avg dist:', sum(all_dists) / len(all_dists)

	print 'Worsts Coords:', sorted(((x, y, dists_by_coord[(x, y)]) for x, y in dists_by_coord), key = lambda v: v[2], reverse = True)[:5]

if __name__ == '__main__':
	data_folder = 'data/1-21/'
	save_file = 'cal_data_1-21.txt'

	# 1) Compute the camera caliibration, rotation vectors, and translation vectors from images in data_folder and save it to save_file
	# computeAndSaveCalibration(data_folder, save_file)

	# 2) Computes the filter images.
	# computeRedFilter(save_file)

	# After computing the filter images, you may have to manually:
	#	a) Remove anything that isn't a laser dot
	#	b) Add "missing_dots" entry to save_file. This is any dot that isn't visible in the filter image. See 'cal_test.txt' for details.
	#	c) Increase the size of the "median" dots

	# 3) Find the Red Dots.
	computeRedDots(save_file)
