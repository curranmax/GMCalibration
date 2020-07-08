
from find_gm_params import *

from scipy.optimize import minimize, least_squares

from collections import defaultdict
import copy
import random
import sys

all_data_stopping_constraints = {'xtol': 2.3e-16, 'ftol': 2.3e-16, 'gtol': 2.3e-16, 'max_nfev': 1e20}
ransac_stopping_constraints = {'xtol': 2.3e-16, 'ftol': 2.3e-16, 'gtol': 2.3e-16, 'max_nfev': 1e20}

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

def getVRRefPointData(fname):
	return getVRData(fname)

def findVRRefPointRansac(data, set_size, num_iters):
	data = [dp for i, dp in enumerate(data) if i % 4 == 0]

	def makeFunc(func_data):
		def func(vs, all_errs = True):
			x, y, z = vs

			ref_vec = Vec(x, y, z)

			ps = []
			sum_ps = Vec(0.0, 0.0, 0.0)
			for dp in func_data:
				p = dp.rot_mtx.mult(ref_vec) + dp.tvec
				ps.append(p)

				sum_ps = sum_ps + p

			average_ps = sum_ps.mult(1.0 / float(len(ps)))

			all_dists = [p.dist(average_ps) for p in ps]
			if all_errs:
				return all_dists
			else:
				avg_dist = sum(all_dists) / float(len(all_dists))

				return avg_dist

		return func

	all_func = makeFunc(data)

	# reduce_func = lambda x: sum(x) / float(len(x))
	reduce_func = max

	best_all_res = None
	best_all_err = None

	best_sel_res = None
	best_sel_err = None

	for i in range(num_iters):
		print('Completed', str(float(i + 1) / float(num_iters) * 100.0) + '%', '                                     \r', end=' ')

		this_data_set = random.sample(data, set_size)

		this_func = makeFunc(this_data_set)
		this_res  = least_squares(this_func, (0.0, 0.0, 0.0), **ransac_stopping_constraints)
		
		this_all_errs = all_func(this_res.x, all_errs = True)
		this_all_err  = reduce_func(this_all_errs)

		this_sel_errs = this_func(this_res.x, all_errs = True)
		this_sel_err  = reduce_func(this_sel_errs)

		if best_all_err is None or this_all_err < best_all_err:
			best_all_err = this_all_err
			best_all_res = this_res.x

		if best_sel_err is None or this_sel_err < best_sel_err:
			best_sel_err = this_sel_err
			best_sel_res = this_res.x
	print('')
	print('Best all error:', best_all_err * 1000.0, 'mm')
	print('Best sel error:', best_sel_err * 1000.0, 'mm')

	return Vec(*best_sel_res)

def findVRRefPoint(data, error_limit = None):
	def makeFunc(func_data):
		def func(vs, all_errs = True):
			x, y, z = vs

			ref_vec = Vec(x, y, z)

			ps = []
			sum_ps = Vec(0.0, 0.0, 0.0)
			for dp in func_data:
				p = dp.rot_mtx.mult(ref_vec) + dp.tvec
				ps.append(p)

				sum_ps = sum_ps + p

			average_ps = sum_ps.mult(1.0 / float(len(ps)))

			all_dists = [p.dist(average_ps) for p in ps]
			if all_errs:
				return all_dists
			else:
				avg_dist = sum(all_dists) / float(len(all_dists))

				return avg_dist

		return func

	all_func = makeFunc(data)
	res = least_squares(all_func, (0.0, 0.0, 0.0), **all_data_stopping_constraints)

	errs = all_func(res.x, all_errs = True)

	if error_limit != None:

		limited_data = []
		for dp, err in zip(data, errs):
			if err <= error_limit:
				print('Using dp with error:', err)
				limited_data.append(dp)

		limited_func = makeFunc(limited_data)
		res = least_squares(all_func, (0.0, 0.0, 0.0), **all_data_stopping_constraints)

		errs = limited_func(res.x, all_errs = True)

	print('')
	print('Max ref error:', max(errs) * 1000.0, 'mm')
	print('Avg ref error:', sum(errs) / float(len(errs)) * 1000.0, 'mm')

	# print 'All ref error:\n', '\n'.join(map(str, sorted(errs)))

	return Vec(*res.x)

class NumEntrieDataPoint:
	def __init__(self, x, y, ne):
		self.x  = x
		self.y  = y
		self.ne = ne

def getConversionData(fname, num_entries_fname, ref_vec):
	f = open(num_entries_fname, 'r')

	ne_data = []
	for line in f:
		if line.strip() == 'BX BY N':
			continue

		x, y, ne = list(map(int, line.split()))
		ne_data.append(NumEntrieDataPoint(x, y, ne))

	data = getVRData(fname)

	i = 0
	data_by_coord = defaultdict(list)
	for nep in ne_data:
		data_by_coord[(nep.x, nep.y)] += list([d.rot_mtx.mult(ref_vec) + d.tvec for d in data[i : i + nep.ne]])
		i += nep.ne

	# if i != len(data):
	# 	raise Exception('Mismatch data')

	return data_by_coord

def reduceConversionData(data, ignore_error = None):
	avg_dists = []
	new_data = dict()
	for k, points in data.items():

		sum_point = Vec(0.0, 0.0, 0.0)
		for p in points:
			sum_point = sum_point + p

		avg_point = sum_point.mult(1.0 / float(len(points)))

		avg_dist = sum(p.dist(avg_point) for p in points) / float(len(points))

		if avg_dist >= ignore_error:
			print(k, avg_dist * 1000.0)

		if ignore_error is None or avg_dist < ignore_error:
			avg_dists.append(avg_dist)
			new_data[k] = avg_point


	print('Reduction stats')
	print('Max Error:', max(avg_dists) * 1000.0, 'mm')
	print('Avg Error:', sum(avg_dists) / float(len(avg_dists)) * 1000.0, 'mm')

	return new_data

GLOBAL_EDGE_LENGTH = None
GLOBAL_INIT_TVEC = None
GLOBAL_INIT_ROT_ANGLES = None

# Find the rotatation matrix R and translation vector T, s.t. given a point X in wall space, R * X + T is the point in VR space
def findConversionValues(data, flip_x_dimension = False, flip_y_dimension = False, limit_data = False, ransac_set_size = None, ransac_num_iters = None, low_error_size = None):
	
	if GLOBAL_EDGE_LENGTH is None:
		raise Exception('GLOBAL_EDGE_LENGTH not set')
	edge_length = GLOBAL_EDGE_LENGTH

	xv = (-1.0 if flip_x_dimension else 1.0)
	yv = (-1.0 if flip_y_dimension else 1.0)

	reflection_matrix = Matrix( xv, 0.0, 0.0,
							   0.0,  yv, 0.0,
							   0.0, 0.0, 1.0)

	def makeFunc(func_data):
		def func(vals, raw = True, verbose = False, just_rail = limit_data, just_hand = False):
			if raw:
				x, y, z, theta, alpha, beta = vals

				tvec = Vec(x, y, z)
				rot_mtx = quatFromAngle(theta, alpha, beta).toRotMatrix()
			else:
				tvec, rot_mtx = vals

			result = []
			for (ix, iy), p in func_data.items():
				if just_rail and (ix > 12 or iy > 12):
					continue

				if just_hand and not (ix > 12 or iy > 12):
					continue

				wall_point = Vec(float(ix) * edge_length, float(iy) * edge_length, 0.0)

				reflected_wall_point = reflection_matrix.mult(wall_point)

				converted_point = rot_mtx.mult(reflected_wall_point) + tvec
				result.append(converted_point.dist(p))

				if verbose:
					print(reflected_wall_point, converted_point, p, converted_point.dist(p))

			return result
		return func

	all_func = makeFunc(data)

	init_edge_length = [3.72 / 100.0]

	if GLOBAL_INIT_TVEC is None:
		raise Exception('GLOBAL_INIT_TVEC not set')

	if GLOBAL_INIT_ROT_ANGLES is None:
		raise Exception('GLOBAL_INIT_ROT_ANGLES not set')

	init_tvec = GLOBAL_INIT_TVEC
	init_rot_mtx = GLOBAL_INIT_ROT_ANGLES

	# res = least_squares(all_func, init_tvec + init_rot_mtx, **all_data_stopping_constraints)

	# x, y, z, theta, alpha, beta = res.x

	# tvec = Vec(0.390437037079, 1.63642427355, -1.04906989547)
	# rot_mtx = Matrix(-0.998500745687, 0.0368210904233, -0.0405026932777,
						# 0.0366011989211, 0.999310981304, 0.00615750627084,
						# 0.0407015122599, 0.00466582746951, -0.99916045606)

	# tvec = Vec(x, y, z)
	# rot_mtx = quatFromAngle(theta, alpha, beta).toRotMatrix()

	tvec = Vec(-0.250705299603, 1.29973565267, 0.341761404037)

	rot_mtx = Matrix(-0.0464761969007, 0.0090332291895, 0.998878553124,
						0.0566646731451, 0.998372803796, -0.00639214029261,
						-0.997310923403, 0.0563040443536, -0.046912435989)


	# all_errs = all_func((x, y, z, theta, alpha, beta), verbose = False, just_rail = False, just_hand = False)
	all_errs = all_func((tvec, rot_mtx), raw = False)

	# drawHistogram(map(lambda x: x *1000.0, all_errs), title = 'Error in Conversion')

	print('')
	print('Convert from Wall To VR coordinate values')

	print('Edge Length:', edge_length * 1000.0, 'mm')
	print('Tvec:       ', tvec)
	print('Rot mtx:\n', rot_mtx)

	print('')
	print('Max conversion all_error:', max(all_errs) * 1000.0, 'mm')
	print('Avg conversion all_error:', sum(all_errs) / float(len(all_errs)) * 1000.0, 'mm')

	if ransac_set_size is not None and ransac_num_iters is not None:
		print('Starting RANSAC for conversion values')

		reduce_func = max

		best_sel_err = None
		best_sel_res = None

		best_all_err = None
		best_all_res = None

		for i in range(ransac_num_iters):
			print('Completed', str(float(i + 1) / float(ransac_num_iters) * 100.0) + '%', '                                     \r', end=' ')
			sys.stdout.flush()

			this_data_keys = random.sample(list(data.keys()), ransac_set_size)
			this_data = {key: data[key] for key in this_data_keys}

			this_func = makeFunc(this_data)

			this_res = least_squares(this_func, init_tvec + init_rot_mtx, **ransac_stopping_constraints)

			this_sel_errs = this_func(res.x)
			this_sel_err  = reduce_func(this_sel_errs)

			this_all_errs = all_func(res.x)
			this_all_err  = reduce_func(this_all_errs)

			if best_sel_err is None or this_sel_err < best_sel_err:
				best_sel_err = this_sel_err
				best_sel_res = this_res

			if best_all_err is None or this_sel_err < best_all_err:
				best_all_err = this_all_err
				best_all_res = this_res

		print('')
		print('')
		print('Best all error:', best_all_err * 1000.0, 'mm')
		print('Best sel error:', best_sel_err * 1000.0, 'mm')

	if low_error_size is not None:
		all_errs_with_data = [(e, k, data[k]) for e, k in zip(all_errs, data)]
		all_errs_with_data.sort()

		print('')
		print('Max conversion low_error:', max(e for e, _, _ in all_errs_with_data[:low_error_size]) * 1000.0, 'mm')
		print('Avg conversion low_error:', sum(e for e, _, _ in all_errs_with_data[:low_error_size]) / float(low_error_size) * 1000.0, 'mm')

		low_err_data = {k:p for _,k, p in all_errs_with_data[:low_error_size]}

		low_err_func = makeFunc(low_err_data)

		print('')
		print('Starting low error regression with', low_error_size, 'data points')
		this_res = least_squares(low_err_func, init_tvec + init_rot_mtx, **all_data_stopping_constraints)
		print('Finished low error regression')

		x, y, z, theta, alpha, beta = this_res.x

		this_tvec = Vec(x, y, z)
		this_rot_mtx = quatFromAngle(theta, alpha, beta).toRotMatrix()

		this_errs = low_err_func(this_res.x)

		print('')
		print('Lower error only regression results:')

		print('Tvec:', this_tvec)
		print('Rot mtx:\n', this_rot_mtx)

		print('')
		print('Max conversion low_error:', max(this_errs) * 1000.0, 'mm')
		print('Avg conversion low_error:', sum(this_errs) / float(len(this_errs)) * 1000.0, 'mm')

		return this_tvec, this_rot_mtx

	return tvec, rot_mtx

def calcConversionError(data, rot_mtx, tvec):
	edge_length = 3.7 / 100.0

	reflection_matrix = Matrix(-1.0, 0.0, 0.0,
								0.0, 1.0, 0.0,
								0.0, 0.0, 1.0)

	errors = []
	for (ix, iy), p in data.items():
		wall_point = Vec(float(ix) * edge_length, float(iy) * edge_length, 0.0)
		refelcted_wall_point = reflection_matrix.mult(wall_point)

		converted_point = rot_mtx.mult(refelcted_wall_point) + tvec
		errors.append(converted_point.dist(p))

	print('Max Error:', max(errors))
	print('Avg Error:', sum(errors) / float(len(errors)))

def outputBoardTestPointS(board_fname, vr_fname, rot_mtx, tvec):
	b_f = open(board_fname, 'w')
	vr_f = open(vr_fname, 'w')

	for x in range(20):
		for y in range(14):
			board_point = Vec(x * 37.0, y * 37.0, 0.0)

			vr_point = rot_mtx.mult(board_point) + tvec

			b_f.write(str(board_point.x) + ' ' + str(board_point.y) + ' ' + str(board_point.z) + '\n')
			vr_f.write(str(vr_point.x) + ' ' + str(vr_point.y) + ' ' + str(vr_point.z) + '\n')

	b_f.close()
	vr_f.close()

def outputVRTestPoints(dot_fname, vr_point_fname, vr_dot_fname, rot_mtx, tvec):
	dots = getDotsFromFile(dot_fname)

	vr_point_f = open(vr_point_fname, 'w')

	for dot in dots:
		vr_loc = rot_mtx.mult(dot.loc) + tvec

		vr_point_f.write(str(vr_loc.x) + ' ' + str(vr_loc.y) + ' ' + str(vr_loc.z) + '\n')

	vr_point_f.close()

	vr_dot_f = open(vr_dot_fname, 'w')

	vr_dot_f.write('GMH GMV X Y Z')

	for dot in dots:
		vr_loc = rot_mtx.mult(dot.loc) + tvec

		vr_dot_f.write(' '.join(map(str, [dot.gmh, dot.gmv, vr_loc.x, vr_loc.y, vr_loc.z])) + '\n')

def distanceToPlane(plane, point):
	k = (plane.point - point).dot(plane.norm) / plane.norm.dot(plane.norm)
	return abs(k * plane.norm.mag())

def checkConversionDataAgainstPlane(wall_plane, data):
	# Check data against wall_plane
	dists = []
	rail_dists = []
	hand_dists = []
	for (ix, iy), p in data.items():
		this_dist = distanceToPlane(wall_plane, p)
		dists.append(this_dist)

		if ix > 12 or iy > 12:
			hand_dists.append(this_dist)
		else:
			rail_dists.append(this_dist)

	print('')
	print('Max dist:', max(dists))
	print('Avg dist:', sum(dists) / float(len(dists)))

	# print ''
	# print 'Max hand_dist:', max(hand_dists)
	# print 'Avg hand_dist:', sum(hand_dists) / float(len(hand_dists))

	# print ''
	# print 'Max rail_dist:', max(rail_dists)
	# print 'Avg rail_dist:', sum(rail_dists) / float(len(rail_dists))

	# Find the best plane that goes through data
	def func(vs, just_rail = True, just_hand = False):
		alpha, beta, x, y, z = vs
		plane = Plane(vecFromAngle(alpha, beta), Vec(x, y, z))

		if just_hand:
			return [distanceToPlane(plane, p) for (ix, iy), p in data.items() if ix > 12 or iy > 12]

		if just_rail:
			return [distanceToPlane(plane, p) for (ix, iy), p in data.items() if not (ix > 12 or iy > 12)]

		return [distanceToPlane(plane, p) for _, p in data.items()]


	init_alpha, init_beta = wall_plane.norm.getAngles()
	init_x, init_y, init_z = wall_plane.point.x, wall_plane.point.y, wall_plane.point.z
	res = least_squares(func, (init_alpha, init_beta, init_x, init_y, init_z))

	alpha, beta, x, y, z = res.x

	fitted_plane = Plane(vecFromAngle(alpha, beta), Vec(x, y, z))
	fitted_error = func((alpha, beta, x, y, z))
	fitted_rail_error = func((alpha, beta, x, y, z), just_rail = True)
	fitted_hand_error = func((alpha, beta, x, y, z), just_hand = True)

	print('')
	print('Max fitted error:', max(fitted_error) * 1000.0, 'mm')
	print('Avg fitted error:', sum(fitted_error) / float(len(fitted_error)) * 1000.0, 'mm')

	# print ''
	# print 'Max fitted hand error:', max(fitted_hand_error)
	# print 'Avg fitted hand error:', sum(fitted_hand_error) / float(len(fitted_hand_error))

	# print ''
	# print 'Max fitted rail error:', max(fitted_rail_error)
	# print 'Avg fitted rail error:', sum(fitted_rail_error) / float(len(fitted_rail_error))

def calculateGridSize(data, out_fname, sorted_out_fname):
	reg_f = open(out_fname, 'w')
	sorted_f = open(sorted_out_fname, 'w')
	col_header = 'X1 Y1 X2 Y2 Dist_VR_mm\n'
	reg_f.write(col_header)
	sorted_f.write(col_header)
	
	dists = []
	all_vals = []
	for (ix1, iy1), p1 in sorted(data.items()):
		for xd, yd in [(1, 0), (0, 1)]:
			ix2 = ix1 + xd
			iy2 = iy1 + yd
			if (ix2, iy2) in data:
				p2 = data[(ix2, iy2)]

				this_dist = p1.dist(p2)
				dists.append(this_dist)

				vals = (ix1, iy1, ix2, iy2, this_dist * 1000.0)
				reg_f.write(' '.join(map(str, vals)) + '\n')
				all_vals.append(vals)
	reg_f.close()

	all_vals.sort(key = lambda x: x[4])
	for vals in all_vals:
		sorted_f.write(' '.join(map(str, vals)) + '\n')
	sorted_f.close()

	print('')
	print('Max grid size:', max(dists) * 1000.0, 'mm')
	print('Min grid size:', min(dists) * 1000.0, 'mm')
	print('Avg grid size:', sum(dists) / float(len(dists)) * 1000.0, 'mm')

def getTXInitValues():
	global GLOBAL_EDGE_LENGTH, GLOBAL_INIT_TVEC, GLOBAL_INIT_ROT_ANGLES

	print('\nRunning with TX Values\n')

	conversion_data_fname = 'data/4-16/vr_to_board_data_4-16.txt'
	num_point_fname = 'data/4-16/vr_num_4-16.txt'

	gm_board_fname = 'data/4-25/gm_board_4-25.txt'
	gm_vr_fname = 'data/4-25/gm_vr_4-25.txt'

	GLOBAL_EDGE_LENGTH = 3.72 / 100.0
	GLOBAL_INIT_TVEC = [0.391034833952, 1.63543827235, -1.04940144591]
	GLOBAL_INIT_ROT_ANGLES = [math.pi / 2.0, math.pi / 2.0, 0.0]

	return conversion_data_fname, num_point_fname, gm_board_fname, gm_vr_fname, None

def getRXInitValues():
	global GLOBAL_EDGE_LENGTH, GLOBAL_INIT_TVEC, GLOBAL_INIT_ROT_ANGLES

	print('\nRunning with RX Values\n')

	conversion_data_fname = 'data/5-6/board_to_vr_5-6.txt'
	num_point_fname = 'data/5-6/btv_num_5-6.txt'

	gm_board_fname = 'data/6-21/gm_board_6-21.txt'
	gm_vr_fname = 'data/6-26/gm_vr_rel_6-26.txt'

	vr_pos_rot_fname = 'data/6-26/rx_vr_pos_rot_6-26.txt'

	GLOBAL_EDGE_LENGTH = 53.08 / 1000.0
	GLOBAL_INIT_TVEC = [-0.22850624415, 1.31810481426, 0.305000694493]
	GLOBAL_INIT_ROT_ANGLES = [math.pi / 4.0, math.pi / 2.0, 0.0]

	return conversion_data_fname, num_point_fname, gm_board_fname, gm_vr_fname, vr_pos_rot_fname

def findRXRelativePosition(abs_gm, vr_data):
	rel_gm = copy.deepcopy(abs_gm)

	vr_tvec = Vec(0.0, 0.0, 0.0)
	vr_rot_mtx = Matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	for dp in vr_data:
		vr_tvec = vr_tvec + dp.tvec
		vr_rot_mtx = vr_rot_mtx + dp.rot_mtx

	vr_tvec = vr_tvec.mult(1.0 / len(vr_data))
	vr_rot_mtx = vr_rot_mtx.mult(1.0 / len(vr_data))

	print('\nVR Position')
	print('Tvec:', vr_tvec)
	print('Rot mtx:')
	print(vr_rot_mtx)
	print('')

	# Since this is a rotation matrix, the inverse is just the transpose.
	inv_vr_rot_mtx = vr_rot_mtx.transpose()

	print('Inverse Rotation matrix:')
	print(inv_vr_rot_mtx)

	loc_func = lambda l: inv_vr_rot_mtx.mult(l - vr_tvec)
	dir_func = lambda d: inv_vr_rot_mtx.mult(d)

	rel_gm = GM(dir_func(abs_gm.init_dir), loc_func(abs_gm.init_point),
				Plane(dir_func(abs_gm.m1.norm), loc_func(abs_gm.m1.point)), dir_func(abs_gm.a1),
				Plane(dir_func(abs_gm.m2.norm), loc_func(abs_gm.m2.point)), dir_func(abs_gm.a2))

	return rel_gm

if __name__ == '__main__':
	calculate_ref_vec = False

	if calculate_ref_vec:
		ref_data = getVRRefPointData('data/4-16/vr_ori_ref_4-15.txt')

		# ref_vec = findVRRefPointRansac(ref_data, 5, 10000)
		ref_vec = findVRRefPoint(ref_data)

	else:
		ref_vec = Vec(-0.0146541189049, 0.0361052143921, 0.237722417595)

	print('')
	print('Ref vec:', ref_vec)
	print('')

	conversion_data_fname, num_point_fname, gm_board_fname, gm_vr_fname, vr_pos_rot_fname = getRXInitValues()

	data = getConversionData(conversion_data_fname, num_point_fname, ref_vec)

	data = reduceConversionData(data, ignore_error = 1.0 / 1000.0)

	# calculateGridSize(data, 'data/4-16/vr_grid_size_4-16.txt', 'data/4-16/vr_grid_size_sorted_4-16.txt')

	# tvec, rot_mtx = findConversionValues(data, ransac_set_size = None, ransac_num_iters = None, low_error_size = 50)

	tvec = Vec(-0.252647108837, 1.32179836801, 0.350554865532)
	rot_mtx = Matrix(-0.0423259936468, 0.00522985360934, 0.999090165547,
						0.0265911743282, 0.999637958755, -0.00410619827161,
						-0.998749928515, 0.0263931818397, -0.0424497378404)

	gm_model = getGMFromFile(gm_board_fname)

	# Make the rot_mtx convert the direction of the x-axis and the units from mm to m
	new_mtx = []
	for row in rot_mtx.vals:
		for v in row:
			new_mtx.append(v / 1000.0)

	new_rot_mtx = Matrix(*new_mtx)

	# print 'Wall plane:'
	# print 'norm: ', new_rot_mtx.mult(Vec(0.0, 0.0, 1.0)).norm()
	# print 'point:', tvec

	# wall_plane = Plane(new_rot_mtx.mult(Vec(0.0, 0.0, 1.0)).norm(), tvec)

	# checkConversionDataAgainstPlane(wall_plane, data)

	# print ''
	# for i in range(5):
	# 	board_point = Vec(i * 36.9, 0.0, 0.0)
	# 	converted_point = new_rot_mtx.mult(board_point) + tvec
	# 	print board_point, data[(i, 0)], converted_point, data[(i, 0)].dist(converted_point)

	# print ''
	# for i in range(5):
	# 	converted_point = new_rot_mtx.mult(Vec(0.0, i * 36.9, 0.0)) + tvec
	# 	print data[(0, i)], converted_point, data[(0, i)].dist(converted_point)

	gm_vr_model = gm_model.move(new_rot_mtx, tvec)

	if vr_pos_rot_fname is not None:
		# TODO average values (R and T) from this filename then given x=gm_vr_model find y s.t. x = R * y + T

		print('Converting to relative GM model')

		vr_pos_rot_data = getVRData(vr_pos_rot_fname)

		gm_vr_model = findRXRelativePosition(gm_vr_model, vr_pos_rot_data)

	outputGM(gm_vr_model, gm_vr_fname)

	# outputVRTestPoints('data/3-15/data_3-15.txt', 'data/3-26/vr_points_3-26.txt', 'data/3-26/vr_dots_3-26.txt', new_rot_mtx, tvec)
