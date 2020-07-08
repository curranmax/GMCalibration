
from utils import *

from rotation_interpolation import RotationAxis

def compareVR(vr_data, ra):
	dists = [vr_data[i].tvec.dist(vr_data[j].tvec) for i in range(len(vr_data)) for j in xrange(i + 1, len(vr_data))]
	print 'Max distance:', max(dists) * 1000.0, 'mm'

	angles = [ra.getVal(dp.rot_mtx) for dp in vr_data]

	angle_dif = [abs(angles[i] - angles[j]) for i in range(len(vr_data)) for j in xrange(i + 1, len(vr_data))]
	print 'Max angle:', max(angle_dif) * 1000.0, 'mrad'

if __name__ == '__main__':
	vr_data = getVRData('../data/9-11/repeatability_9-11.txt')

	vr_data = reduceVRData(vr_data, 100)

	print len(vr_data)

	ra = RotationAxis(Vec(0.016528067671, -0.999848973027, -0.00537160274127), Vec(0.0575521126682, 0.00143230405411, -0.0909701499992))

	# Middle
	compareVR([vr_data[i] for i in [0, 2, 3]], ra)

	# 0.9
	compareVR([vr_data[i] for i in [4, 6]], ra)

	# 0.0
	compareVR([vr_data[i] for i in [5, 7]], ra)
