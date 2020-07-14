
from utils import *

import interpolation

import math
import random

from scipy.optimize import least_squares

stopping_constraints = {'xtol': 2.3e-16, 'ftol': 2.3e-16, 'gtol': 2.3e-16}

# Similar to VRLine but for rotation
class RotationAxis:
	def __init__(self, axis, ref_vec = None):
		self.axis  = axis.norm()
		if ref_vec is None:
			self.ref_vec = (Vec(1.0, 0.0, 0.0) - self.axis.mult(self.axis.dot(Vec(1.0, 0.0, 0.0)))).norm()
		else:
			self.ref_vec = ref_vec

	def getPerpendicularDir(self, target):
		d = target - self.point
		return d - self.axis.mult(d.dot(self.axis))

	def getVal(self, rm):
		v = rm.mult(self.ref_vec)
		v = (v - self.axis.mult(self.axis.dot(v))).norm()

		return self.ref_vec.signedAngle(v, self.axis)

	def getUnrotatedPosition(self, rm, tvec):
		theta = self.getVal(rm)

		rot_mtx = rotMatrix(self.axis, theta)

		this_ref_vec = rot_mtx.mult(self.ref_vec)
		this_ra_point = tvec - this_ref_vec

		return this_ra_point + self.ref_vec

# All data must be taken at the same point on the rail. No restriction on what angles
def findAxisOfRotation(vr_data):
	# 1) Find the plane that the VR positions are on
	def plane_error(vals):
		alpha, beta, d = vals

		norm = vecFromAngle(alpha, beta)
		point = norm.mult(d)

		plane = Plane(norm, point)

		errs = [plane.dist(dp.tvec) for dp in vr_data]
		return errs

	rv = least_squares(plane_error, (0.0, 0.0, 0.0), **stopping_constraints)

	alpha, beta, d = rv.x

	norm = vecFromAngle(alpha, beta)
	ra = RotationAxis(norm)

	def ref_vec_error(vals):
		this_ref_vec = Vec(*vals)

		points = []
		for dp in vr_data:
			theta = ra.getVal(dp.rot_mtx)

			rot_mtx = rotMatrix(ra.axis, theta)

			rot_ref_vec = rot_mtx.mult(this_ref_vec)
			this_ra_point = dp.tvec - rot_ref_vec

			points.append(this_ra_point + this_ref_vec)

		avg_point = sum(points, Vec(0.0, 0.0, 0.0)).mult(1.0 / float(len(points)))

		return [p.dist(avg_point) for p in points]

	rv = least_squares(ref_vec_error, (0.0, 0.0, 0.0), bounds = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)), **stopping_constraints)

	ref_vec = Vec(*rv.x)

	ra.ref_vec = ref_vec

	return ra

if __name__ == '__main__':
	align_data_fname = '../data/8-27/align_data_8-27.txt'
	vr_data_fname    = '../data/8-27/vr_data_8-27.txt'
	
	vr_repeats = 100

	func_type = interpolation.LinearPieceWise

	output_fname = '../data/8-27/rotation_interpolation_8-27.txt'

	# Get VR Data
	vr_data = getVRData(vr_data_fname)
	vr_data = reduceVRData(vr_data, vr_repeats)
	
	# Calculate the axis of rotation
	print('Calculating axis of rotation')
	ra = findAxisOfRotation(vr_data)

	# Calculate angles
	vr_angles = [ra.getVal(dp.rot_mtx) for dp in vr_data]
	print('VR Angles:', ', '.join([str(x * 180.0 / math.pi) for x in vr_angles]))

	# Get Align data
	align_data = getVoltData(align_data_fname)

	# Create the functions for each mirror
	print('Creating interpolation functions:', func_type)
	tx1_func = func_type(vr_angles, [dp.tx_gm1 for dp in align_data])
	tx2_func = func_type(vr_angles, [dp.tx_gm2 for dp in align_data])
	rx1_func = func_type(vr_angles, [dp.rx_gm1 for dp in align_data])
	rx2_func = func_type(vr_angles, [dp.rx_gm2 for dp in align_data])

	print('Outputting values to:', output_fname)
	out_f = open(output_fname, 'w')

	out_f.write('RA ' + ' '.join(map(str, [ra.axis.x, ra.axis.y, ra.axis.z])) + '\n')
	out_f.write('RV ' + ' '.join(map(str, [ra.ref_vec.x, ra.ref_vec.y, ra.ref_vec.z])) + '\n')

	out_f.write('TX1 ' + tx1_func.outputString() + '\n')
	out_f.write('TX2 ' + tx2_func.outputString() + '\n')
	out_f.write('RX1 ' + rx1_func.outputString() + '\n')
	out_f.write('RX2 ' + rx2_func.outputString() + '\n')
		
