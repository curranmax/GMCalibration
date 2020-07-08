
from find_gm_params import *
from convert_from_wall_to_vr import *

from scipy.optimize import least_squares

from copy import deepcopy

# ======================================================================
# ======================================================================
# Starting Models. Once you get somethingn that works, you can use those results here
TX_MODEL_FNAME = 'data/3-4/tx_gm_vr_3-4.txt'
RX_MODEL_FNAME = 'data/3-4/rx_gm_vr_3-4.txt'

FIX_TX_POSITION = False
INIT_TX_POSITIONO = Vec(0.40069934, 0.98062328, 0.36487804)

# The two data files
VR_DATA_FNAME    = 'data/3-6/vr_data_3-6.txt'
ALIGN_DATA_FNAME = 'data/3-6/align_data_3-6.txt'

# THe output files
NEW_TX_MODEL_FNAME = 'data/3-6/tx_gm_vr_3-6.txt'
NEW_RX_MODEL_FNAME = 'data/3-6/rx_gm_vr_3-6.txt'

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
			tx1, tx2, rx1, rx2 = map(int, (spl[cols['TX1']], spl[cols['TX2']], spl[cols['RX1']], spl[cols['RX2']]))

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
	for i in xrange(0, len(vr_data), n):
		if volt_data[i / n].dist < low_dist_thresh:
			continue

		avg_tvec = Vec(0.0, 0.0, 0.0)
		# avg_rot_mtx = Matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
		all_quats = []
		for x in xrange(i, i + n, 1):
			avg_tvec    = avg_tvec    + vr_data[x].tvec
			# avg_rot_mtx = avg_rot_mtx + vr_data[x].rot_mtx
			all_quats.append(vr_data[x].quat)

		avg_tvec    = avg_tvec.mult(1.0 / float(n))
		avg_rot_mtx = computeAvgQuat(all_quats).toRotMatrix()

		full_data.append((VRDataPoint(None, avg_tvec, avg_rot_mtx, None, None, None), volt_data[i / n]))
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

def adjustModel(tx_gm_model, rx_gm_model, full_data, use_dist = False, dist_err = 0.0, adjust_tx_input_beam = False):
	init_tx_tvec, init_tx_rot_mtx, local_tx_gm_model = getLocalModel(tx_gm_model)
	init_rx_tvec, init_rx_rot_mtx, local_rx_gm_model = getLocalModel(rx_gm_model)

	def func(vals, split = False):
		tx_tvec    = Vec(*vals[0:3])
		tx_rot_mtx = rotMatrixFromAngles(*vals[3:6])

		rx_tvec    = Vec(*vals[6:9])
		rx_rot_mtx = rotMatrixFromAngles(*vals[9:12])

		this_tx_gm_model = deepcopy(local_tx_gm_model)
		this_rx_gm_model = deepcopy(local_rx_gm_model)

		if adjust_tx_input_beam:
			dif_init_point_y, dif_init_point_z = vals[12:14]

			this_tx_gm_model.init_point.y += dif_init_point_y
			this_tx_gm_model.init_point.z += dif_init_point_z

			old_alpha, old_beta = local_tx_gm_model.init_dir.getAngles()
			dif_alpha, dif_beta = vals[14:16]

			this_tx_gm_model.init_dir = vecFromAngle(old_alpha + dif_alpha, old_beta + dif_beta)

		this_tx_gm_model = this_tx_gm_model.move(tx_rot_mtx, tx_tvec)
		this_rx_gm_model = this_rx_gm_model.move(rx_rot_mtx, rx_tvec)

		if split:
			tx_errs = []
			rx_errs = []
		else:
			errs = []

		for vr_dp, volt_dp in full_data:
			abs_rx_gm_model = this_rx_gm_model.move(vr_dp.rot_mtx, vr_dp.tvec)

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

			if split:
				tx_errs.append(tx_err)
				rx_errs.append(rx_err)
			else:
				errs.append(tx_err)
				errs.append(rx_err)

		if split:
			return tx_errs, rx_errs
		else:
			return errs

	init_guess = [init_tx_tvec.x, init_tx_tvec.y, init_tx_tvec.z] + \
					list(init_tx_rot_mtx.getAngles()) + \
					[init_rx_tvec.x, init_rx_tvec.y, init_rx_tvec.z] + \
					list(init_rx_rot_mtx.getAngles())

	bound = 10.0
	print 'Using bounds of:', bound * 1000.0, 'mm/mrad'
	print ''

	min_bounds = [v - bound for v in init_guess]
	max_bounds = [v + bound for v in init_guess]

	if adjust_tx_input_beam:
		init_guess += [0.0] * 4
		min_bounds += [-0.01] * 4
		max_bounds += [ 0.01] * 4

	init_tx_errs, init_rx_errs = func(init_guess, split = True)

	print 'Init Avg TX error:', sum(init_tx_errs) / float(len(init_tx_errs)) * 1000.0, 'mm'
	print 'Init Max TX error:', max(init_tx_errs) * 1000.0, 'mm'
	print ''
	print 'Init Avg RX error:', sum(init_rx_errs) / float(len(init_rx_errs)) * 1000.0, 'mm'
	print 'Init Max RX error:', max(init_rx_errs) * 1000.0, 'mm'
	print ''
	print 'Init Avg all error:', sum(init_tx_errs + init_rx_errs) / float(len(init_tx_errs + init_rx_errs)) * 1000.0, 'mm'
	print 'Init Max all error:', max(init_tx_errs + init_rx_errs) * 1000.0, 'mm'
	print ''

	stopping_constraints = {'xtol': 2.3e-16, 'ftol': 2.3e-16, 'gtol': 2.3e-16, 'max_nfev': 1e4}
	
	rv = least_squares(func, init_guess,  bounds = (min_bounds, max_bounds), **stopping_constraints)

	print rv.x
	vals = rv.x
	
	tx_tvec    = Vec(*vals[0:3])
	tx_rot_mtx = rotMatrixFromAngles(*vals[3:6])
	rx_tvec    = Vec(*vals[6:9])
	rx_rot_mtx = rotMatrixFromAngles(*vals[9:12])

	final_input = [tx_tvec.x, tx_tvec.y, tx_tvec.z] + list(tx_rot_mtx.getAngles()) + [rx_tvec.x, rx_tvec.y, rx_tvec.z] + list(rx_rot_mtx.getAngles())

	if adjust_tx_input_beam:
		dif_init_point_y, dif_init_point_z = vals[12:14]
		dif_alpha, dif_beta = vals[14:16]

		final_input += [dif_init_point_y, dif_init_point_z, dif_alpha, dif_beta]

	final_tx_errs, final_rx_errs = func(final_input, split = True)

	print 'TX tvec:', tx_tvec - init_tx_tvec
	print 'TX rm angles:', '(' + ', '.join(map(str, vals[3:6]))  +')'
	print 'TX rot mtx:'
	print tx_rot_mtx
	print ''
	print 'RX tvec:', rx_tvec - init_rx_tvec
	print 'RX rm angles:', '(' + ', '.join(map(str, vals[9:12]))  +')'
	print 'RX rot mtx:'
	print rx_rot_mtx
	print ''

	if adjust_tx_input_beam:
		print 'Dif TX init point:', dif_init_point_y, dif_init_point_z
		print 'Dif TX init dir:  ', dif_alpha, dif_beta

	print 'Avg TX error:', sum(final_tx_errs) / float(len(final_tx_errs)) * 1000.0, 'mm'
	print 'Max TX error:', max(final_tx_errs) * 1000.0, 'mm'
	print ''
	print 'Avg RX error:', sum(final_rx_errs) / float(len(final_rx_errs)) * 1000.0, 'mm'
	print 'Max RX error:', max(final_rx_errs) * 1000.0, 'mm'
	print ''

	print 'Avg total error:', sum(final_tx_errs + final_rx_errs) / float(len(final_tx_errs + final_rx_errs)) * 1000.0, 'mm'
	print 'Max total error:', max(final_tx_errs + final_rx_errs) * 1000.0, 'mm'
	print ''

	if adjust_tx_input_beam:
		local_tx_gm_model.init_point.y += dif_init_point_y
		local_tx_gm_model.init_point.z += dif_init_point_z

		old_alpha, old_beta = local_tx_gm_model.init_dir.getAngles()
		local_tx_gm_model.init_dir = vecFromAngle(old_alpha + dif_alpha, old_beta + dif_beta)

	return local_tx_gm_model.move(tx_rot_mtx, tx_tvec), local_rx_gm_model.move(rx_rot_mtx, rx_tvec)

def testStuff(tx_gm_model, rx_gm_model, full_data):
	p, d = tx_gm_model.getOutput(pow(2, 15), pow(2, 15))
	print 'TX:', p, d
	print 'Second Vec should be approx:', Vec(0.0, 0.0, 1.0)

	def_vr_p = full_data[1][0]

	print def_vr_p.rot_mtx

	def_rx_gm_model = rx_gm_model.move(def_vr_p.rot_mtx, def_vr_p.tvec)

	p, d = def_rx_gm_model.getOutput(pow(2, 15), pow(2, 15))
	print 'RX:', p, d
	print 'Second Vec should be approx:', Vec(0.0, 0.0, -1.0)

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

	print tx_gm_model.init_point + avg_vec

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

	new_tx_gm_model_fname = NEW_TX_FNAME
	new_rx_gm_model_fname = NEW_RX_FNAME

	print 'TX Model:', tx_gm_model_fname
	print 'RX Model:', rx_gm_model_fname
	print ''
	print 'VR Data:  ', vr_data_fnames
	print 'Volt Data:', volt_data_fnames
	print ''
	print 'VR per Volt:', num_vr_per_volt
	print 'LD Thresh:  ', low_dist_thresh, 'm'
	print 'Use dist:   ', use_dist
	print 'Dist err:   ', dist_err * 1000.0, 'mm'
	print ''
	print 'New TX Model:', new_tx_gm_model_fname
	print 'New RX Model:', new_rx_gm_model_fname
	print ''

	# Get GM Models
	tx_gm_model = getGMFromFile(tx_gm_model_fname)
	rx_gm_model = getGMFromFile(rx_gm_model_fname)

  print tx_gm_model

  quit()

	# tx_gm_model.scale(1.0 / 1000.0)
	# rx_gm_model.scale(1.0 / 1000.0)
	
	# CONVERTS THE TX GM FROM LOCAL TO VR-SPACE
	# rail_dir = Vec(-0.998484214442, -0.00707444409786, -0.0545822842292)

	# vr_to_tx_dir = rail_dir.cross(Vec(0.0, 1.0, 0.0))

	# vr_perp_pos = Vec(0.1758, 1.1352, -0.4984)
	# vr_to_tx_distance = 1.04

	# tx_height = 0.09082
	# vr_height = 0.13620

	# tx_init_vr_pos = vr_perp_pos + vr_to_tx_dir.mult(vr_to_tx_distance) - Vec(0.0, vr_height, 0.0) + Vec(0.0, tx_height, 0.0)
	# tx_init_vr_ori = rotMatrixFromAngles(0.0, math.pi, 0.0)

	# rx_init_vr_pos = Vec(-0.225, 0.08772 - 0.13620, 0.13517)
	# rx_init_vr_ori = rotMatrixFromAngles(0.0, 0.0, 0.0)

	# tx_init_vr_pos = Vec(0.32419366,  0.94709523, -1.54457414)
	# tx_init_vr_ori = rotMatrixFromAngles(-0.42893532391, -0.0697981620592, -0.0765762330009) * rotMatrixFromAngles(0.0, math.pi, 0.0)

	# rx_init_vr_pos = Vec(-0.1276891,  -0.20371923,  0.06273378)
	# rx_init_vr_ori = rotMatrixFromAngles(-0.42893532391, -0.0697981620592, -0.0765762330009)
	
	# tx_gm_model = tx_gm_model.move(tx_init_vr_ori, tx_init_vr_pos)
	# rx_gm_model = rx_gm_model.move(rx_init_vr_ori, rx_init_vr_pos)


	# Get Data
	full_data = []
	for vr_data_fname, volt_data_fname in zip(vr_data_fnames, volt_data_fnames):
		vr_data   = getVRData(vr_data_fname)
		volt_data = getVoltData(volt_data_fname)

		full_data += processData(vr_data, volt_data, n = num_vr_per_volt, low_dist_thresh = low_dist_thresh)

	# Adjust Models
	new_tx_gm_model, new_rx_gm_model = adjustModel(tx_gm_model, rx_gm_model, full_data, use_dist = use_dist, dist_err = dist_err, adjust_tx_input_beam = False)

	if new_tx_gm_model_fname is not None:
		outputGM(new_tx_gm_model, new_tx_gm_model_fname)
	if new_rx_gm_model_fname is not None:
		outputGM(new_rx_gm_model, new_rx_gm_model_fname)
