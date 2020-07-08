
from utils import *
from interpolation import *
from tracking import *

import random

import matplotlib.pyplot as plt

def analyzeError(errs, gm_name, io_names, convert_to_radians = True, plot = True):
	for io, io_name in io_names.items():
		simp_errs = [abs(a[io] - b) for _, a, b in errs]

		if convert_to_radians:
			simp_errs = [x * 40.0 / 65536 * 3.14159 / 180.0 * 1000.0 for x in simp_errs]

			unit = 'mrad'

		else:
			unit = 'gm units'

		print('Interpolation method:', io_name)
		print('Avg', gm_name, 'error:', sum(simp_errs) / float(len(simp_errs)), unit)
		print('Max', gm_name, 'error:', max(simp_errs), unit)
		print('')

	if plot:
		xvs = [x for x, _, _ in errs]
		cal = [c for _, _, c in errs]

		plt.plot(xvs, cal, label = 'calc')

		for io, io_name in io_names.items():
			est = [e[io] for _, e, _ in errs]
			plt.plot(xvs, est, label = io_name)

		plt.title(gm_name)
		plt.legend()

		plt.show()

if __name__ == '__main__':
	# Get both GM Models
	tx_gm_model = getGMFromFile('../data/7-22/adj_tx_vr_gm_7-22.txt')
	rx_gm_model = getGMFromFile('../data/7-22/adj_tx_vr_gm_7-22.txt')

	# Get the two VR positions, and one VR orientation
	vr_pos_1 = Vec(0.175399, 1.3222,  -0.746824)
	vr_pos_2 = Vec(0.375825, 1.31571, -0.745065)
	vr_ori   = Quat(-0.993707, 0.0367736, -0.0280139, 0.102029).toRotMatrix()\

	vr_line = VRLine([vr_pos_1, vr_pos_2], vr_ori)

	# Generate fake data
	vals = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
	tracking_func = LinkSearchTracking(1000, SearchTracking(1000))
	tx1s, tx2s, rx1s, rx2s = [], [], [], []
	for val in vals:
		vr_p = vr_line.generatePoint(val)

		this_rx_gm_model = rx_gm_model.move(vr_ori, vr_p)
		tx1, tx2, rx1, rx2 = tracking_func(tx_gm_model, this_rx_gm_model)

		tx_p, tx_d = tx_gm_model.getOutput(tx1, tx2)
		rx_p, rx_d = this_rx_gm_model.getOutput(rx1, rx2)

		tx1s.append(tx1)
		tx2s.append(tx2)
		rx1s.append(rx1)
		rx2s.append(rx2)

	interops = [NearestNeighbor, Linear, LinearSpline, CubicSpline]
	tx1_funcs = {io: io(vals, tx1s) for io in interops}
	tx2_funcs = {io: io(vals, tx2s) for io in interops}
	rx1_funcs = {io: io(vals, rx1s) for io in interops}
	rx2_funcs = {io: io(vals, rx2s) for io in interops}

	# Compare interpolation methods
	min_val = -0.5
	max_val =  0.5
	num_vals = 1001

	tx1_errs, tx2_errs, rx1_errs, rx2_errs = [], [], [], []
	for i in range(num_vals):
		this_val = (max_val - min_val) / (num_vals - 1.0) * i + min_val

		vr_p = vr_line.generatePoint(this_val)

		this_rx_gm_model = rx_gm_model.move(vr_ori, vr_p)
		calc_tx1, calc_tx2, calc_rx1, calc_rx2 = tracking_func(tx_gm_model, this_rx_gm_model)

		est_tx1, est_tx2, est_rx1, est_rx2 = {}, {}, {}, {}
		for io in interops:
			est_tx1[io], est_tx2[io], est_rx1[io], est_rx2[io] = tx1_funcs[io](this_val), tx2_funcs[io](this_val), rx1_funcs[io](this_val), rx2_funcs[io](this_val)

		tx1_errs.append((this_val, est_tx1, calc_tx1))
		tx2_errs.append((this_val, est_tx2, calc_tx2))
		rx1_errs.append((this_val, est_rx1, calc_rx1))
		rx2_errs.append((this_val, est_rx2, calc_rx2))

	io_names = {NearestNeighbor: 'nn', Linear: 'lin', LinearSpline: 'lin_spline', CubicSpline: 'cub'}
	analyzeError(tx1_errs, 'tx1', io_names, plot = True)
	analyzeError(tx2_errs, 'tx2', io_names, plot = False)
	analyzeError(rx1_errs, 'rx1', io_names, plot = False)
	analyzeError(rx2_errs, 'rx2', io_names, plot = False)

