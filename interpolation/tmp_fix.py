
from utils import *

from interpolation import *

if __name__ == '__main__':
	fix_vr_fname = '../data/12-6/tmp_fix.txt'
	fix_vr_data = getVRData(fix_vr_fname)

	old_vr_fname = '../data/12-6/vr_data_12-6.txt'
	old_vr_data = getVRData(old_vr_fname)[-len(fix_vr_data):]


	old_vr_data = reduceVRData(old_vr_data, 100)
	fix_vr_data = reduceVRData(fix_vr_data, 100)
	
	print(len(old_vr_data))
	print(len(fix_vr_data))

	translation = sum((fix_dp.tvec - old_dp.tvec for old_dp, fix_dp in zip(old_vr_data, fix_vr_data)), Vec(0.0, 0.0, 0.0)).mult(1.0 / float(len(old_vr_data)))
	print(translation)

	tx_gm_model = getGMFromFile('../data/12-6/t_v3.txt')
	new_tx_gm_model = tx_gm_model.move(rotMatrixFromAngles(0.0, 0.0, 0.0), translation)

	outputGM(new_tx_gm_model, '../data/12-6/tx_gm_vr_12-6_fixed.txt')

	# old_line = calcVRLine([dp.tvec for dp in old_vr_data])
	# fix_line = calcVRLine([dp.tvec for dp in fix_vr_data])

	# print old_line.p, old_line.d
	# print fix_line.p, fix_line.d