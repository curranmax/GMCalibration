
import numpy as np
from scipy.optimize import fsolve
import cv2
import math
import os
from collections import defaultdict

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

def outputCalibration(cal, fname):
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

	f.close()

def inputCalibration(cal, fname):
	f = open(fname, 'r')

	vals = {'fx': None, 'fy': None, 'cx': None, 'cy': None, 'k1': None, 'k2': None, 'k3': None, 'p1': None, 'p2': None}

	for line in f:
		token, val = line.split()
		val = float(val)

		if token in vals and vals[token] == None:
			vals[token] = val
		else:
			if token not in vals:
				raise Exception('Unexpected token: ' + token)
			if vals[token] != None:
				raise Exception('Duplicate token: ' + token)

	if any(v == None for k, v in vals.iteritems()):
		raise Exception('Missing values: ' + ', '.join(k for k, v in vals.iteritems() if v == None))

	return CameraCalibration(**vals)

class ImagePair:
	def __init__(self, cal_fname, dot_fname):
		self.cal_fname = cal_fname
		self.dot_fname = dot_fname

		self.tvec = None
		self.rvec = None

def getImagePairs(folder):
	fnames = defaultdict(list)
	for f in os.listdir(folder):
		n = int(f[f.find('_') + 1 : f.find('.')])

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

def calibrateCamera(image_pairs, show_images = False):
	# Termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	objp = np.zeros((6 * 6, 3), np.float32)
	objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)

	obj_points = [] # 3d points of the plane
	img_points = [] # 2d points in the image

	working_inds = []

	for i, ip in enumerate(image_pairs):
		print 'Running image', i + 1, 'of', len(image_pairs)

		img = cv2.imread(ip.cal_fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the corners
		ret, corners = cv2.findChessboardCorners(gray, (6, 6), None)

		if ret == True:
			working_inds.append(i)

			corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

			obj_points.append(objp)
			img_points.append(corners2)

			if show_images:
				img = cv2.drawChessboardCorners(img, (6, 6), corners2, ret)

				x, y, z = img.shape
				img = cv2.resize(img, (int(x / 10), int(y / 10)))

				cv2.imshow('img', img)
				cv2.waitKey(500)

	if show_images:
		cv2.destroyAllWindows()

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
		rt_point = np.matmul(rot_mtx, rp) + tvec

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

			rv = projectSimp([[x, y, z]], cal, rvec, tvec)
			calc_ix, calc_iy = rv[0]

			return [ip[0] - calc_ix, ip[1] - calc_iy]

		rx, ry = fsolve(eq, (0.0, 0.0))

		real_points.append([rx, ry, z])

	return np.array(real_points)

if __name__ == '__main__':
	image_pairs = getImagePairs('data/1-11/')


	# Calibration
	cal = calibrateCamera(image_pairs)

else:

	# TMP
	rvec = np.array([-0.43239601, 0.25603401, -3.08832021])
	tvec = np.array([3.79739158, 0.89895019, 14.85930553])

	init_points = np.array([[0.5, 0.75, 1.0],])

	image_points = projectSimp(init_points, cal, rvec, tvec)

	# print image_points

	z_values = np.array([p[2] for p in init_points])
	real_points = unproject(image_points, z_values, cal, rvec, tvec)

	print init_points[0]
	print real_points[0]


