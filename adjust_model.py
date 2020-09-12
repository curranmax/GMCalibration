
from find_gm_params import *
from convert_from_wall_to_vr import *
from tracking import *

from scipy.optimize import least_squares

from copy import deepcopy
import math

# ======================================================================
# ======================================================================
# Starting Models. Once you get something that works, you can use those results here

TX_MODEL_FNAME = 'data/8-10/tx_gm_vr_8-10_2.txt'
RX_MODEL_FNAME = 'data/8-10/rx_gm_vr_8-10_2.txt'

# The two data files
VR_DATA_FNAME    = 'data/8-12/vr_data_8-12.txt'
ALIGN_DATA_FNAME = 'data/8-12/align_data_8-12.txt'

# The output files
NEW_TX_MODEL_FNAME = 'data/8-12/tx_gm_vr_8-12_tmp.txt'
NEW_RX_MODEL_FNAME = 'data/8-12/rx_gm_vr_8-12_tmp.txt'

# Adjust Initial Values
FIX_MODEL_POSITIONS = True
INIT_TX_POSITION = Vec(0.024742774, 1.311485219, 1.601749131)
INIT_RX_POSITION = Vec(-0.07342621, 0.197472645, -0.273909988) # Vec(0.273909988, 0.197472645, -0.07342621)
ADJUST_TX_MATRIX = rotMatrixFromAngles(0.0, 0.0, 0.0)
ADJUST_RX_MATRIX = rotMatrixFromAngles(0.0, math.pi, 0.0)

VR_TO_TX = Vec(0.0, 0.0, 1.0)
UP_DIR   = Vec(0.0, 1.0, 0.0)

TX_LEFT_DIR = Vec(-1.0, 0.0, 0.0)
RX_LEFT_DIR = Vec( 1.0, 0.0, 0.0)

# Error type to use (must be either 'distance' or 'gm_values')
ERROR_TYPE = 'gm_values'

# Parameters of the Search
LINEAR_BOUNDS = 1.0 # In meters
ANGULAR_BOUNDS = 3.5 # In radians

# Max search iterations
MAX_ITERS = 1e4

# Controls what values are adjustModel
ADJUST_TX_POSITION = False
ADJUST_TX_ORIENTATION = True
ADJUST_RX_POSITION = False
ADJUST_RX_ORIENTATION = True

# ======================================================================
# ======================================================================

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

def processData(vr_data, volt_data, n = 1, low_dist_thresh = 0.0):
	if len(vr_data) != len(volt_data) * n:
		raise Exception('Mismtaching data: VR --> ' + str(len(vr_data)) + ', Volt --> ' + str(len(volt_data)))

	full_data = []
	for i in range(0, len(vr_data), n):
		if volt_data[int(i / n)].dist < low_dist_thresh:
			continue

		avg_tvec = Vec(0.0, 0.0, 0.0)
		# avg_rot_mtx = Matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
		all_quats = []
		for x in range(i, i + n, 1):
			avg_tvec    = avg_tvec    + vr_data[x].tvec
			# avg_rot_mtx = avg_rot_mtx + vr_data[x].rot_mtx
			all_quats.append(vr_data[x].quat)

		avg_tvec    = avg_tvec.mult(1.0 / float(n))
		avg_rot_mtx = computeAvgQuat(all_quats).toRotMatrix()

		full_data.append((VRDataPoint(None, avg_tvec, avg_rot_mtx, None, None, None), volt_data[int(i / n)]))
	return full_data

def getLocalModel(gm_model):
	init_tvec    = gm_model.m1.point
	init_rot_mtx = rotMatrixFromAngles(0.0, 0.0, 0.0)

	local_gm_model = gm_model.move(init_rot_mtx, init_tvec.mult(-1))

	return init_tvec, init_rot_mtx, local_gm_model

def distanceToLineWithDist(p_0, d, t, exp_dist, dist_err):
	k = findK(p_0, d, t)

	if k > exp_dist + dist_err:
		new_k = exp_dist + dist_err
	elif k < exp_dist - dist_err:
		new_k = exp_dist - dist_err
	else:
		new_k = k

	return (p_0 + d.mult(new_k)).dist(t), k < 0.0

def adjustModel(tx_gm_model, rx_gm_model, full_data, use_dist = False, dist_err = 0.0, adjust_tx_input_beam = False, outer_error_type = 'distance', adjust_tx_position = True, adjust_tx_orientation = True, adjust_rx_position = True, adjust_rx_orientation = True):
	init_tx_tvec, init_tx_rot_mtx, local_tx_gm_model = getLocalModel(tx_gm_model)
	init_rx_tvec, init_rx_rot_mtx, local_rx_gm_model = getLocalModel(rx_gm_model)

	def func(vals, split = False, error_type = outer_error_type):
		x = 0

		if adjust_tx_position:
			tx_tvec = Vec(*vals[x : x+3])
			x += 3
		else:
			tx_tvec = init_tx_tvec

		if adjust_tx_orientation:
			tx_rot_mtx = rotMatrixFromAngles(*vals[x : x+3])
			x += 3
		else:
			tx_rot_mtx = init_tx_rot_mtx

		if adjust_rx_position:
			rx_tvec = Vec(*vals[x : x+3])
			x += 3
		else:
			rx_tvec = init_rx_tvec

		if adjust_rx_orientation:
			rx_rot_mtx = rotMatrixFromAngles(*vals[x : x+3])
			x += 3
		else:
			rx_rot_mtx = init_rx_rot_mtx

		this_tx_gm_model = deepcopy(local_tx_gm_model)
		this_rx_gm_model = deepcopy(local_rx_gm_model)

		if adjust_tx_input_beam:
			dif_init_point_y, dif_init_point_z = vals[x : x+2]
			x += 2

			this_tx_gm_model.init_point.y += dif_init_point_y
			this_tx_gm_model.init_point.z += dif_init_point_z

			old_alpha, old_beta = local_tx_gm_model.init_dir.getAngles()
			dif_alpha, dif_beta = vals[x : x+2]
			x += 2

			this_tx_gm_model.init_dir = vecFromAngle(old_alpha + dif_alpha, old_beta + dif_beta)

		this_tx_gm_model = this_tx_gm_model.move(tx_rot_mtx, tx_tvec)
		this_rx_gm_model = this_rx_gm_model.move(rx_rot_mtx, rx_tvec)

		if split:
			tx_errs = []
			rx_errs = []
		else:
			errs = []

		for vr_dp, volt_dp in full_data:
			print('-' * 80)
			print(vr_dp.tvec)
			print(vr_dp.rot_mtx)
			abs_rx_gm_model = this_rx_gm_model.move(vr_dp.rot_mtx, vr_dp.tvec)

			if error_type == 'distance':
				tx_p, tx_d = this_tx_gm_model.getOutput(volt_dp.tx_gm1, volt_dp.tx_gm2)
				rx_p, rx_d = abs_rx_gm_model.getOutput(volt_dp.rx_gm1, volt_dp.rx_gm2)

				if use_dist:
					if volt_dp.dist is None:
						raise Exception('Trying to use distance when none is given')

					tx_err, _ = distanceToLineWithDist(tx_p, tx_d, rx_p, volt_dp.dist, dist_err)
					rx_err, _ = distanceToLineWithDist(rx_p, rx_d, tx_p, volt_dp.dist, dist_err)

				else:
					tx_err, _ = distanceToLine(tx_p, tx_d, rx_p)
					rx_err, _ = distanceToLine(rx_p, rx_d, tx_p)

				tx_err = (tx_err,)
				rx_err = (rx_err,)
			elif error_type == 'gm_values':
				tracking = LinkSearchTracking(100, SearchTracking(100, use_float = True))
				tx1, tx2, rx1, rx2 = tracking(this_tx_gm_model, abs_rx_gm_model)

				err_func = lambda x: x

				tx_err = (err_func(tx1 - volt_dp.tx_gm1), err_func(tx2 - volt_dp.tx_gm2))
				rx_err = (err_func(rx1 - volt_dp.rx_gm1), err_func(rx2 - volt_dp.rx_gm2))
			else:
				raise Exception('Unexpected error_type = %s' % error_type)

			if split:
				for v in tx_err:
					tx_errs.append(v)
				for v in rx_err:
					rx_errs.append(v)
			else:
				for v in tx_err:
					errs.append(v)
				for v in rx_err:
					errs.append(v)
			print('-' * 80)

		if split:
			return tx_errs, rx_errs
		else:
			return errs

	linear_bound = LINEAR_BOUNDS
	angular_bound = ANGULAR_BOUNDS
	print('Using linear bounds of:', linear_bound * 1000.0, 'mm')
	print('Using angular bounds of:', angular_bound * 1000.0, 'mrad')
	print('')

	init_guess = []
	bounds = []
	if adjust_tx_position:
		init_guess += [init_tx_tvec.x, init_tx_tvec.y, init_tx_tvec.z]
		bounds += [linear_bound] * 3

	if adjust_tx_orientation:
		init_guess += list(init_tx_rot_mtx.getAngles())
		bounds += [angular_bound] * 3

	if adjust_rx_position:
		init_guess += [init_rx_tvec.x, init_rx_tvec.y, init_rx_tvec.z]
		bounds += [linear_bound] * 3
	
	if adjust_rx_orientation:
		init_guess += list(init_rx_rot_mtx.getAngles())
		bounds += [angular_bound] * 3

	min_bounds = [v - b for v, b in zip(init_guess, bounds)]
	max_bounds = [v + b for v, b in zip(init_guess, bounds)]

	if adjust_tx_input_beam:
		init_guess += [0.0] * 4
		min_bounds += [-0.01] * 4
		max_bounds += [ 0.01] * 4

	init_tx_errs, init_rx_errs = func(init_guess, split = True, error_type = 'distance')
	print('Distance based Error:')
	print('Init Avg TX error:', sum(init_tx_errs) / float(len(init_tx_errs)) * 1000.0, 'mm')
	print('Init Max TX error:', max(init_tx_errs) * 1000.0, 'mm')
	print('')
	print('Init Avg RX error:', sum(init_rx_errs) / float(len(init_rx_errs)) * 1000.0, 'mm')
	print('Init Max RX error:', max(init_rx_errs) * 1000.0, 'mm')
	print('')
	print('Init Avg all error:', sum(init_tx_errs + init_rx_errs) / float(len(init_tx_errs + init_rx_errs)) * 1000.0, 'mm')
	print('Init Max all error:', max(init_tx_errs + init_rx_errs) * 1000.0, 'mm')
	print('')

	angle_conversion_factor = 40.0 / 65536.0 * math.pi / 180.0

	init_tx_errs, init_rx_errs = func(init_guess, split = True, error_type = 'gm_values')
	init_tx_errs, init_rx_errs = list(map(abs, init_tx_errs)), list(map(abs, init_rx_errs))
	print('Mirror-Angle based Error:')
	print('Init Avg TX error:', sum(init_tx_errs) / float(len(init_tx_errs)) * angle_conversion_factor * 1000.0, 'mrad')
	print('Init Max TX error:', max(init_tx_errs) * angle_conversion_factor * 1000.0, 'mrad')
	print('')
	print('Init Avg RX error:', sum(init_rx_errs) / float(len(init_rx_errs)) * angle_conversion_factor * 1000.0, 'mrad')
	print('Init Max RX error:', max(init_rx_errs) * angle_conversion_factor * 1000.0, 'mrad')
	print('')
	print('Init Avg all error:', sum(init_tx_errs + init_rx_errs) / float(len(init_tx_errs + init_rx_errs)) * angle_conversion_factor * 1000.0, 'mrad')
	print('Init Max all error:', max(init_tx_errs + init_rx_errs) * angle_conversion_factor * 1000.0, 'mrad')
	print('')

	quit()

	stopping_constraints = {'xtol': 2.3e-16, 'ftol': 2.3e-16, 'gtol': 2.3e-16, 'max_nfev': MAX_ITERS}
	
	print('Starting model optimization')
	rv = least_squares(func, init_guess,  bounds = (min_bounds, max_bounds), **stopping_constraints)
	print('Finished model optimization')
	print('')

	vals = rv.x
	print('Raw output:', vals)
	print('')

	x = 0

	if adjust_tx_position:
		tx_tvec = Vec(*vals[x : x+3])
		x += 3
	else:
		tx_tvec = init_tx_tvec

	if adjust_tx_orientation:
		tx_rot_mtx = rotMatrixFromAngles(*vals[x : x+3])
		x += 3
	else:
		tx_rot_mtx = init_tx_rot_mtx

	if adjust_rx_position:
		rx_tvec = Vec(*vals[x : x+3])
		x += 3
	else:
		rx_tvec = init_rx_tvec

	if adjust_rx_orientation:
		rx_rot_mtx = rotMatrixFromAngles(*vals[x : x+3])
		x += 3
	else:
		rx_rot_mtx = init_rx_rot_mtx

	final_input = list(vals)

	if adjust_tx_input_beam:
		dif_init_point_y, dif_init_point_z = vals[x : x+2]
		x += 2

		dif_alpha, dif_beta = vals[x : x+2]
		x += 2

	print('TX tvec:', tx_tvec - init_tx_tvec)
	print('TX rm angles:', '(' + ', '.join(map(str, tx_rot_mtx.getAngles()))  +')')
	print('TX rot mtx:')
	print(tx_rot_mtx)
	print('')
	print('RX tvec:', rx_tvec - init_rx_tvec)
	print('RX rm angles:', '(' + ', '.join(map(str, rx_rot_mtx.getAngles()))  +')')
	print('RX rot mtx:')
	print(rx_rot_mtx)
	print('')

	if adjust_tx_input_beam:
		print('Dif TX init point:', dif_init_point_y, dif_init_point_z)
		print('Dif TX init dir:  ', dif_alpha, dif_beta)

	final_tx_errs, final_rx_errs = func(final_input, split = True, error_type = 'distance')
	print('Distance based Error:')
	print('Avg TX error:', sum(final_tx_errs) / float(len(final_tx_errs)) * 1000.0, 'mm')
	print('Max TX error:', max(final_tx_errs) * 1000.0, 'mm')
	print('')
	print('Avg RX error:', sum(final_rx_errs) / float(len(final_rx_errs)) * 1000.0, 'mm')
	print('Max RX error:', max(final_rx_errs) * 1000.0, 'mm')
	print('')
	print('Avg total error:', sum(final_tx_errs + final_rx_errs) / float(len(final_tx_errs + final_rx_errs)) * 1000.0, 'mm')
	print('Max total error:', max(final_tx_errs + final_rx_errs) * 1000.0, 'mm')
	print('')

	final_tx_errs, final_rx_errs = func(final_input, split = True, error_type = 'gm_values')
	final_tx_errs, final_rx_errs = list(map(abs, final_tx_errs)), list(map(abs, final_rx_errs))
	print('Mirror-Angle based Error:')
	print('Avg TX error:', sum(final_tx_errs) / float(len(final_tx_errs)) * angle_conversion_factor * 1000.0, 'mrad')
	print('Max TX error:', max(final_tx_errs) * angle_conversion_factor * 1000.0, 'mrad')
	print('')
	print('Avg RX error:', sum(final_rx_errs) / float(len(final_rx_errs)) * angle_conversion_factor * 1000.0, 'mrad')
	print('Max RX error:', max(final_rx_errs) * angle_conversion_factor * 1000.0, 'mrad')
	print('')
	print('Avg all error:', sum(final_tx_errs + final_rx_errs) / float(len(final_tx_errs + final_rx_errs)) * angle_conversion_factor * 1000.0, 'mrad')
	print('Max all error:', max(final_tx_errs + final_rx_errs) * angle_conversion_factor * 1000.0, 'mrad')
	print('')

	if adjust_tx_input_beam:
		local_tx_gm_model.init_point.y += dif_init_point_y
		local_tx_gm_model.init_point.z += dif_init_point_z

		old_alpha, old_beta = local_tx_gm_model.init_dir.getAngles()
		local_tx_gm_model.init_dir = vecFromAngle(old_alpha + dif_alpha, old_beta + dif_beta)

	return local_tx_gm_model.move(tx_rot_mtx, tx_tvec), local_rx_gm_model.move(rx_rot_mtx, rx_tvec)

def testStuff(tx_gm_model, rx_gm_model, full_data):
	print('\nStart Sanity Check:')
	p, d   = tx_gm_model.getOutput(pow(2, 15), pow(2, 15))
	p2, d2 = tx_gm_model.getOutput(pow(2, 15), pow(2, 16))
	p3, d3 = tx_gm_model.getOutput(pow(2, 16), pow(2, 15))
	print('TX')
	print('Beam Launch Point: ', p)
	print('Should be close to:', INIT_TX_POSITION)
	print('Beam Launch Dir:   ', d)
	print('Should be close to:', VR_TO_TX.mult(-1.0))
	print('Vertical beam dir change:  ', (d2 - d.mult(d.dot(d2))).norm())
	print('Should be close to:        ', UP_DIR)
	print('Horizontal beam dir change:', (d3 - d.mult(d.dot(d3))).norm())
	print('Should be close to:        ', TX_LEFT_DIR, '\n')

	def_vr_p = full_data[1][0]

	def_rx_gm_model = rx_gm_model.move(def_vr_p.rot_mtx, def_vr_p.tvec)

	p, d   = def_rx_gm_model.getOutput(pow(2, 15), pow(2, 15))
	p2, d2 = def_rx_gm_model.getOutput(pow(2, 15), pow(2, 16))
	p3, d3 = def_rx_gm_model.getOutput(pow(2, 16), pow(2, 15))
	print('RX')
	print('Beam Launch Point: ', p)
	print('Should be close to:', def_vr_p.tvec + def_vr_p.rot_mtx.mult(INIT_RX_POSITION))
	print('Beam Launch Dir:   ', d)
	print('Should be close to:', VR_TO_TX)
	print('Vertical beam dir change:', (d2 - d.mult(d.dot(d2))).norm())
	print('Should be close to:      ', UP_DIR)
	print('Horizontal beam dir change:', (d3 - d.mult(d.dot(d3))).norm())
	print('Should be close to:        ', RX_LEFT_DIR, '\n')

	print('End Sanity Check\n')

def findConversionBetweenVRSpaces():
	old_vr_data = getVRData('data/8-8/vr_data_8-8.txt')
	new_vr_data = getVRData('data/8-17/vr_data_8-17.txt')

	volt_data = getVoltData('data/8-17/align_data_8-17.txt')

	old_avg_data = processData(old_vr_data, volt_data, n = 100)
	new_avg_data = processData(new_vr_data, volt_data, n = 100)

	sum_vec = Vec(0.0, 0.0, 0.0)
	for (old_dp, _), (new_dp, _) in zip(old_avg_data, new_avg_data):
		# print new_dp.tvec - old_dp.tvec
		sum_vec += new_dp.tvec - old_dp.tvec

	avg_vec = sum_vec.mult(1.0 / float(len(old_avg_data)))

	rot_mtx = rotMatrixFromAngles(0.0, 0.0, 0.0)

	tx_gm_model = getGMFromFile('data/4-25/gm_vr_4-25.txt')

	print(tx_gm_model.init_point + avg_vec)

	converted_vr_pos = [rot_mtx.mult(old_dp.tvec) + avg_vec for old_dp, _ in old_avg_data]

	# print sum([con_pos.dist(new_dp.tvec) for con_pos, (new_dp, _) in zip(converted_vr_pos, new_avg_data)]) / float(len(converted_vr_pos)) * 1000.0, 'mm'

if __name__ == '__main__':
	tx_gm_model_fname = TX_MODEL_FNAME
	rx_gm_model_fname = RX_MODEL_FNAME

	vr_data_fnames   = [VR_DATA_FNAME]
	volt_data_fnames = [ALIGN_DATA_FNAME]

	num_vr_per_volt = 1000

	low_dist_thresh = 0.5

	use_dist = False
	dist_err = 0.5

	new_tx_gm_model_fname = NEW_TX_MODEL_FNAME
	new_rx_gm_model_fname = NEW_RX_MODEL_FNAME

	print('TX Model:', tx_gm_model_fname)
	print('RX Model:', rx_gm_model_fname)
	print('')
	print('VR Data:  ', vr_data_fnames)
	print('Volt Data:', volt_data_fnames)
	print('')
	print('VR per Volt:', num_vr_per_volt)
	print('LD Thresh:  ', low_dist_thresh, 'm')
	print('Use dist:   ', use_dist)
	print('Dist err:   ', dist_err * 1000.0, 'mm')
	print('')
	print('New TX Model:', new_tx_gm_model_fname)
	print('New RX Model:', new_rx_gm_model_fname)
	print('')

	# Get GM Models
	tx_gm_model = getGMFromFile(tx_gm_model_fname)
	rx_gm_model = getGMFromFile(rx_gm_model_fname)

	if FIX_MODEL_POSITIONS:
		tx_gm_model = tx_gm_model.move(rotMatrixFromAngles(0.0, 0.0, 0.0), tx_gm_model.init_point.mult(-1.0))
		rx_gm_model = rx_gm_model.move(rotMatrixFromAngles(0.0, 0.0, 0.0), rx_gm_model.init_point.mult(-1.0))
		tx_gm_model = tx_gm_model.move(ADJUST_TX_MATRIX, INIT_TX_POSITION)
		rx_gm_model = rx_gm_model.move(ADJUST_RX_MATRIX, INIT_RX_POSITION)

	# Get Data
	full_data = []
	for vr_data_fname, volt_data_fname in zip(vr_data_fnames, volt_data_fnames):
		vr_data   = getVRData(vr_data_fname)
		volt_data = getVoltData(volt_data_fname)

		full_data += processData(vr_data, volt_data, n = num_vr_per_volt, low_dist_thresh = low_dist_thresh)

	testStuff(tx_gm_model, rx_gm_model, full_data)

	# Adjust Models
	new_tx_gm_model, new_rx_gm_model = adjustModel(tx_gm_model, rx_gm_model, full_data, use_dist = use_dist, dist_err = dist_err, adjust_tx_input_beam = False, outer_error_type = ERROR_TYPE, adjust_tx_position = ADJUST_TX_POSITION, adjust_tx_orientation = ADJUST_TX_ORIENTATION, adjust_rx_position = ADJUST_RX_POSITION, adjust_rx_orientation = ADJUST_RX_ORIENTATION)

	if new_tx_gm_model_fname is not None:
		outputGM(new_tx_gm_model, new_tx_gm_model_fname)
	if new_rx_gm_model_fname is not None:
		outputGM(new_rx_gm_model, new_rx_gm_model_fname)

	testStuff(new_tx_gm_model, new_rx_gm_model, full_data)
