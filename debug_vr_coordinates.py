from find_gm_params import *
from convert_from_wall_to_vr import *
from adjust_model import *

VR_DATA_FNAME = 'data/8-12/debug_data_8-12.txt'

UP       = Vec( 0.0, 1.0, 0.0)
RX_TO_TX = Vec( 0.0, 0.0, 1.0)
R_TO_L   = Vec(-1.0, 0.0, 0.0)

# 3-tuple: (start-index, end-index, expected unit vector from start to end)
EXPECTATIONS = [(0, 1, UP),       (5, 3, UP),
				(0, 2, RX_TO_TX), (5, 4, RX_TO_TX),
				(0, 5, R_TO_L),   (1, 3, R_TO_L),   (2, 4, R_TO_L)]

def processData(vr_data, samples_per_point = 1000):
	if len(vr_data) % samples_per_point != 0:
		raise Exception('Unexpected number of [points')

	avg_vr_data = []
	for i in range(0, len(vr_data), samples_per_point):
		avg_tvec = Vec(0.0, 0.0, 0.0)
		all_quats = []
		for x in range(i, i + samples_per_point, 1):
			avg_tvec = avg_tvec + vr_data[x].tvec
			all_quats.append(vr_data[x].quat)

		avg_tvec    = avg_tvec.mult(1.0 / float(samples_per_point))
		avg_rot_mtx = computeAvgQuat(all_quats).toRotMatrix()

		avg_vr_data.append(VRDataPoint(None, avg_tvec, avg_rot_mtx, None, None, None))

	return avg_vr_data

if __name__ == '__main__':
	vr_data = getVRData(VR_DATA_FNAME)
	vr_data = processData(vr_data)

	expectations = EXPECTATIONS

	for si, ei, exp_vec in expectations:
		act_vec = (vr_data[ei].tvec - vr_data[si].tvec).norm()

		print('-' * 50)
		print('For', si, 'to', ei)
		print('Expected:', exp_vec)
		print('Got:     ', act_vec)