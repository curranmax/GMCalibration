
from interpolation import *
from utils import  *
from tracking import *

import matplotlib.pyplot as plt

if __name__ == '__main__':
	# Get interpolation stuff
	vr_line, tx1_func, tx2_func, rx1_func, rx2_func = getInterpolationFuncFromFile('../data/8-21/linear_interpolation_8-21.txt')

	# Get Model Stuff
	tx_gm_model = getGMFromFile('../data/8-21/old_adj_tx_vr_gm_8-21.txt')
	rx_gm_model = getGMFromFile('../data/8-21/old_adj_rx_vr_gm_8-21.txt')

	# Get tracking function
	tracking_func = LinkSearchTracking(1000, SearchTracking(1000))

	# Generate points along VR line
	n_points = 1000
	x_min = 1.7119511797054183e-06
	x_max = 0.853552367834269

	vr_points = []
	for i in range(n_points):
		this_x = (x_max - x_min) / n_points * float(i) + x_min
		this_point = vr_line.generatePoint(this_x)

		vr_points.append(this_point)

	# Calculate GM values using both methods
	this_rm = vr_line.rm

	x = []
	i_tx1s, i_tx2s, i_rx1s, i_rx2s = [], [], [], []
	m_tx1s, m_tx2s, m_rx1s, m_rx2s = [], [], [], []

	for this_point in vr_points:
		# Interpolation
		this_x = vr_line.getVal(this_point)

		x.append(this_x)

		i_tx1, i_tx2, i_rx1, i_rx2 = tx1_func(this_x), tx2_func(this_x), rx1_func(this_x), rx2_func(this_x)

		i_tx1s.append(i_tx1)
		i_tx2s.append(i_tx2)
		i_rx1s.append(i_rx1)
		i_rx2s.append(i_rx2)

		# Model
		this_rx_gm_model = rx_gm_model.move(this_rm, this_point)

		m_tx1, m_tx2, m_rx1, m_rx2 = tracking_func(tx_gm_model, this_rx_gm_model)

		m_tx1s.append(m_tx1)
		m_tx2s.append(m_tx2)
		m_rx1s.append(m_rx1)
		m_rx2s.append(m_rx2)

	# Plot the results
	plt.plot(x, i_rx2s, label = 'Interpolation')
	plt.plot(x, m_rx2s, label = 'Model')

	plt.legend()

	plt.show()

