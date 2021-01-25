
from utils import *

import math
import numpy as np
import cv2
from collections import defaultdict

class LatticeLine:
	def __init__(self, points = None):
		if points is None:
			self.points = []
		else:
			self.points = points

class Vec2:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def dist(self, v):
		return math.sqrt(pow(self.x - v.x, 2.0) + pow(self.y - v.y, 2.0))

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

def findComponentstoMinimizeDistance(da, db, p, t):
	v1 = db - da.mult(da.dot(db) / pow(da.mag(), 2.0))

	b = -(v1).dot(p - t + da.mult(da.dot(t - p) / pow(da.mag(), 2.0))) / pow(v1.mag(), 2.0)
	a = da.dot(t - (db.mult(b) + p)) / pow(da.mag(), 2.0)

	return a, b

def convertToImagePoint(v, origin_point, x_dir, y_dir):
	xv, yv = findComponentstoMinimizeDistance(x_dir, y_dir, origin_point, v)

	if v.dist(origin_point + x_dir.mult(xv) + y_dir.mult(yv)) > 0.1:
		raise Exception('Point, x_dir, or y_dir are not on the image plane')

	return Vec2(xv, yv)
 
def drawDots(lattice_points, lattice_lines, out_fname = None, lp_color = None, lp_inner_radius = None, lp_outer_radius = None, ll_color = None, ll_inner_radius = None, ll_outer_radius = None):
	min_x, max_x = min(p.x for p in lattice_points), max(p.x for p in lattice_points)
	min_y, max_y = min(p.y for p in lattice_points), max(p.y for p in lattice_points)

	height = 1000
	buf = 0.05

	if 1.0 - 2.0 * buf < 0.0:
		raise Exception('Invalid buffer: ' + str(buf))
	
	ox = min_x - (max_x - min_x) / (1.0 - 2.0 * buf) * buf
	oy = min_y - (max_y - min_y) / (1.0 - 2.0 * buf) * buf
	origin_point = Vec2(ox, oy)

	conversion_factor = height * (1.0 - 2.0 * buf) / (max_x - min_x)
	width = int(math.ceil(conversion_factor * (max_y - min_y) / (1.0 - 2.0 * buf)))

	img = np.zeros((height, width, 4), np.uint8)

	lp_brush = []
	ll_brush = []
	outer_radius = max(lp_outer_radius, ll_outer_radius)
	for a in range(-outer_radius, outer_radius + 1, 1):
		for b in range(-outer_radius, outer_radius + 1, 1):
			mag = math.sqrt(pow(a, 2.0) + pow(b, 2.0))
			if mag <= lp_inner_radius:
				lp_weight = 1.0
			elif mag > lp_inner_radius and mag <= lp_outer_radius:
				lp_weight = (mag - lp_outer_radius) / (lp_inner_radius - lp_outer_radius)
			else:
				lp_weight = None

			if lp_weight is not None:
				lp_brush.append((a, b, lp_weight))

			if mag <= ll_inner_radius:
				ll_weight = 1.0
			elif mag > ll_inner_radius and mag <= ll_outer_radius:
				ll_weight = (mag - ll_outer_radius) / (ll_inner_radius - ll_outer_radius)
			else:
				ll_weight = None

			if ll_weight is not None:
				ll_brush.append((a, b, ll_weight))

	ll_pixels = defaultdict(list)
	for ll in lattice_lines:
		for p1, p2 in zip(ll.points[:-1], ll.points[1:]):
			p1x = int((p1.x - origin_point.x) * conversion_factor)
			p1y = int((p1.y - origin_point.y) * conversion_factor)
			p2x = int((p2.x - origin_point.x) * conversion_factor)
			p2y = int((p2.y - origin_point.y) * conversion_factor)

			min_x = int(math.floor(min(p1x - ll_outer_radius, p2x - ll_outer_radius)))
			max_x = int(math.ceil( max(p1x + ll_outer_radius, p2x + ll_outer_radius)))
			min_y = int(math.floor(min(p1y - ll_outer_radius, p2y - ll_outer_radius)))
			max_y = int(math.ceil( max(p1y + ll_outer_radius, p2y + ll_outer_radius)))

			for x in range(min_x, max_x + 1, 1):
				for y in range(min_y, max_y + 1, 1):
					if (pow(p2x - p1x, 2.0) + pow(p2y - p1y, 2.0)) <= 0.0000001:
						k = 0.0
					else:
						k = ((x - p1x) * (p2x - p1x) + (y - p1y) * (p2y - p1y)) / (pow(p2x - p1x, 2.0) + pow(p2y - p1y, 2.0))

					if k < 0.0:
						xt = p1x
						yt = p1y
					elif k > 1.0:
						xt = p2x
						yt = p2y
					else:
						xt = (p2x - p1x) * k + p1x
						yt = (p2y - p1y) * k + p1y

					this_mag = math.sqrt(pow(x - xt, 2.0) + pow(y - yt, 2.0))

					if this_mag <= ll_inner_radius:
						this_weight = 1.0
					elif this_mag > ll_inner_radius and this_mag <= ll_outer_radius:
						this_weight = (this_mag - ll_outer_radius) / (ll_inner_radius - ll_outer_radius)
					else:
						this_weight = None

					if this_weight is not None:
						# prev_color = img[x][y]

						# ll_pixels[(x, y)].append((prev_color, (1.0 - this_weight)))
						ll_pixels[(x, y)].append((ll_color, this_weight))

	for (x, y), vals in ll_pixels.items():
		prev_color = img[x][y]

		max_weight     = None
		weighted_color = None
		for col, w in vals:
			if max_weight is None or w > max_weight:
				max_weight     = w
				weighted_color = col

		new_color = tuple(pc * (1.0 - max_weight) + wc * max_weight for pc, wc in zip(prev_color, weighted_color))

		img[x][y] = new_color

	for lp in lattice_points:
		xp = int((lp.x - origin_point.x) * conversion_factor)
		yp = int((lp.y - origin_point.y) * conversion_factor)

		for a, b, w in lp_brush:
			x = xp + a
			y = yp + b

			prev_color = img[x][y]

			new_color = tuple(pv * (1.0 - w) + nv * w for pv, nv in zip(prev_color, lp_color))

			img[x][y] = new_color

	cv2.imwrite(out_fname, img)

if __name__ == '__main__':
	# Constants
	# Lattice Points
	lp_num_steps = pow(2, 3)
	lp_step_size = pow(2, 16) / lp_num_steps

	lp_lattice_values = [i * lp_step_size for i in range(lp_num_steps + 1)]

	# Lattice lines
	ll_num_steps = pow(2, 10)
	ll_step_size = pow(2, 16) / ll_num_steps

	ll_lattice_values = [i * ll_step_size for i in range(ll_num_steps + 1)]

	# In mm
	dist_to_plane = 1000.0

	# Output params
	filename = 'plots/3-27/distortion_3_27.png'

	# BGRA
	lp_color = (82, 78, 196, 255)
	lp_inner_radius = 10
	lp_outer_radius = 12

	ll_color = (176, 114, 76, 255)
	ll_inner_radius = 2
	ll_outer_radius = 6

	# Create GM
	init_dir = vecFromAngle(0.0, 0.0)
	init_point = Vec(5.0005588143, 29.865539516, -43.3404917057)

	gm = getGMFromCAD(init_dir, init_point)

	p, d = gm.getOutput(pow(2, 15), pow(2, 15))
	
	# Create Wall plane
	plane_norm = d.mult(-1)
	plane_point = p + d.mult(dist_to_plane)

	image_plane = Plane(plane_norm, plane_point)

	# Generate lattice poitns on image_plane
	lattice_points = []
	origin_point = None
	for xv in lp_lattice_values:
		for yv in lp_lattice_values:
			sp, d = gm.getOutput(xv, yv)

			ip, is_neg = image_plane.intersect(sp, d)

			if is_neg:
				raise Exception('Invalid value for: (%d, %d)' % (xv, yv))

			lattice_points.append(ip)

			if xv == pow(2, 15) and yv == pow(2, 15):
				origin_point = ip

	# Generate lattice lines on image_plane
	lattice_lines = []

	# X-lines
	x_dir = None
	for xv in lp_lattice_values:
		this_line = LatticeLine()
		for yv in ll_lattice_values:
			sp, d = gm.getOutput(xv, yv)

			ip, is_neg = image_plane.intersect(sp, d)

			if is_neg:
				raise Exception('Invalid value for: (%d, %d)' % (xv, yv))

			this_line.points.append(ip)

			if xv == pow(2, 15) and yv == pow(2, 15) + ll_step_size:
				x_dir = (origin_point - ip).norm()

		lattice_lines.append(this_line)

	# Y-lines
	y_dir = None
	for yv in lp_lattice_values:
		this_line = LatticeLine()
		for xv in ll_lattice_values:
			sp, d = gm.getOutput(xv, yv)

			ip, is_neg = image_plane.intersect(sp, d)

			if is_neg:
				raise Exception('Invalid value for: (%d, %d)' % (xv, yv))

			this_line.points.append(ip)
			if yv == pow(2, 15) and xv == pow(2, 15) + ll_step_size:
				y_dir = (origin_point - ip).norm()

		lattice_lines.append(this_line)

	# Convert all the points to the Image Plane
	lp_image = [convertToImagePoint(lp, origin_point, x_dir, y_dir) for lp in lattice_points]
	ll_image = [LatticeLine(points = [convertToImagePoint(p, origin_point, x_dir, y_dir) for p in ll.points]) for ll in lattice_lines]

	# Generate image
	drawDots(lp_image, ll_image, out_fname = filename, lp_color = lp_color, lp_inner_radius = lp_inner_radius, lp_outer_radius = lp_outer_radius, ll_color = ll_color, ll_inner_radius = ll_inner_radius, ll_outer_radius = ll_outer_radius)

