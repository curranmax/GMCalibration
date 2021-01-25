
from basic_plots import *

import pandas as pnd

if __name__ == '__main__':
	x_label = 'Beam Diameter (mm)'
	y_label = 'Angular Tolerance (mrad)'
	legend_label = ''

	tx_label = 'TX'
	rx_label = 'RX'

	beam_diameters = [8.7, 10.48, 13.84, 16.64, 21.76, 23.52, 25.6]
	tx_tolerances  = [5.944175553, 10.0134857, 12.69795208, 15.80852423, 18.3225483, 24.16019741, 20.90048823]
	rx_tolerances  = [4.921521694, 4.66585823, 4.580637075, 5.773733243, 5.411543335, 5.368932758, 3.344930329]

	full_data = pnd.DataFrame({x_label: beam_diameters + beam_diameters, y_label: tx_tolerances + rx_tolerances, legend_label: ([tx_label] * len(beam_diameters)) + ([rx_label] * len(beam_diameters))})

	plotLinePlot(full_data, x_label, y_label, hue = legend_label, marker = 'o')
