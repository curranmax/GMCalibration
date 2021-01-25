
from utils import *

from basic_plots import *

import math
import pandas as pnd

def calculateStatistics(fnames):
	data_by_fname = {}
	for fname in fnames:
		data_by_fname[fname] = getVRData(fname)

	total_lin_errs = []
	total_ang_errs = []
	for _, data in data_by_fname.items():
		avg_position = sum((dp.tvec for dp in data), Vec(0.0, 0.0, 0.0)).mult(1.0 / float(len(data)))
		avg_quat = computeAvgQuat([dp.quat for dp in data])

		avg_matrix = avg_quat.toRotMatrix()

		lin_errs = []
		ang_errs = []
		for dp in data:
			# Value in meters
			lin_errs.append(avg_position.dist(dp.tvec))

			# Value in radians
			ang_errs.append(avg_matrix.dist(dp.rot_mtx))

		total_lin_errs += lin_errs 
		total_ang_errs += ang_errs

	lin_std_dev = math.sqrt(sum(pow(v, 2) for v in total_lin_errs) / (len(total_lin_errs) - 1.0))
	ang_std_dev = math.sqrt(sum(pow(v, 2) for v in total_ang_errs) / (len(total_ang_errs) - 1.0))

	lin_std_dev_of_means = lin_std_dev / math.sqrt(len(total_lin_errs))
	ang_std_dev_of_means = ang_std_dev / math.sqrt(len(total_ang_errs))

	return lin_std_dev, ang_std_dev, lin_std_dev_of_means, ang_std_dev_of_means

def makeVRLatencyPlot(fnames):
	data_by_fname = {}
	for fname in fnames:
		data_by_fname[fname] = getVRData(fname)

	all_latencies = []
	for _, data in data_by_fname.items():
		prev_dp = None
		this_latencies = []
		for dp in data:
			if prev_dp is not None:
				latency = dp.time - prev_dp.time
				if latency < 50:
					this_latencies.append(latency)
			prev_dp = dp

		all_latencies += this_latencies

	min_v = 0
	max_v = max(all_latencies)

	latency_count = {v: 0 for v in range(min_v, max_v + 1)}
	for v in all_latencies:
		latency_count[v] += 1

	x_label = 'Update latency (ms)'
	y_label = 'Ratio of updates'

	xticklabels = []
	full_data_dict = {x_label: [], y_label: []}
	for v, count in latency_count.items():
		if v % 5 == 0:
			xticklabels.append(v)
		else:
			xticklabels.append('')
		full_data_dict[x_label].append(v)
		full_data_dict[y_label].append(float(count) / float(len(all_latencies)))

	full_data = pnd.DataFrame(full_data_dict)

	print(full_data)
	print(len(all_latencies))

	# plotBarPlot(full_data, x_label, y_label, xticklabels = xticklabels)
	# plotCDF(all_latencies, x_label = x_label)

def makeVRNoisePlot(fnames, linear = True, angular = True):
	data_by_fname = {}
	for fname in fnames:
		data_by_fname[fname] = getVRData(fname)

	# Position Error
	x_label = 'Time (minutes)'
	ly_label = 'Distance to Average (mm)'
	ay_label = 'Angle between Average (mrad)'
	all_errors = {x_label: [], ly_label: [], ay_label: []}
	prev_last_time = 0

	max_lin_err = None
	max_ang_err = None
	sum_lin_err = 0.0
	sum_ang_err = 0.0
	num_errs = 0
	for _, data in data_by_fname.items():
		avg_position = sum((dp.tvec for dp in data), Vec(0.0, 0.0, 0.0)).mult(1.0 / float(len(data)))
		avg_quat = computeAvgQuat([dp.quat for dp in data])

		avg_matrix = avg_quat.toRotMatrix()

		this_errors = {x_label: [], ly_label: [], ay_label: []}
		for dp in data:
			this_errors[x_label].append(float(dp.time - data[0].time + prev_last_time) / 1000.0 / 60.0) # convert to s

			lin_err = avg_position.dist(dp.tvec) * 1000.0 # convert to mm
			this_errors[ly_label].append(lin_err)

			if max_lin_err is None or lin_err > max_lin_err:
				max_lin_err = lin_err
			sum_lin_err += lin_err

			ang_err = avg_matrix.dist(dp.rot_mtx) * 1000.0 # convert to mrad
			this_errors[ay_label].append(ang_err)

			if max_ang_err is None or ang_err > max_ang_err:
				max_ang_err = ang_err
			sum_ang_err += ang_err

			num_errs += 1

		all_errors[x_label]  += this_errors[x_label]
		all_errors[ly_label] += this_errors[ly_label]
		all_errors[ay_label] += this_errors[ay_label]

		prev_last_time = data[-1].time - data[0].time + prev_last_time + 12

	print('')
	print('Average linear error: ', sum_lin_err / float(num_errs))
	print('Maximum linear error: ', max_lin_err)
	print('Average angular error:', sum_ang_err / float(num_errs))
	print('Maximum angular error:', max_ang_err)
	print('')

	full_data = pnd.DataFrame(all_errors)

	if linear:
		# plotLinePlot(full_data, x_label, ly_label)
		# plotCDF(full_data.get(ly_label))
		plotPDF(full_data.get(ly_label))
	
	if angular:
		# plotLinePlot(full_data, x_label, ay_label)
		# plotCDF(full_data.get(ay_label))
		plotPDF(full_data.get(ay_label))

def makeVRDriftPlot(fname, num_per_point, time_point, err_fnames = None):
	lin_std_dev, ang_std_dev, lin_std_dev_of_means, ang_std_dev_of_means = None, None, None, None
	if err_fnames is not None:
		lin_std_dev, ang_std_dev, lin_std_dev_of_means, ang_std_dev_of_means = calculateStatistics(err_fnames)

	vr_data = getVRData(fname)

	num_groups = len(vr_data) / num_per_point
	grouped_vr_data = [vr_data[a:b] for a, b in zip([i * num_per_point for i in range(num_groups)], [(i + 1) * num_per_point for i in range(num_groups)])]

	avg_vr_points   = [sum((dp.tvec for dp in group), Vec(0.0, 0.0, 0.0)).mult(1.0 / float(len(group))) for group in grouped_vr_data]
	avg_vr_rot_mtxs = [computeAvgQuat([dp.quat for dp in group]).toRotMatrix() for group in grouped_vr_data]

	x_label = 'Time (minutes)'
	ly_label = 'Distance to Initial Point (mm)'
	ay_label = 'Angle to Initial Orientation (mrad)'

	linear_points  = {x_label: [], ly_label:[]}
	angular_points = {x_label: [], ay_label:[]}
	for i, (p, rm) in enumerate(zip(avg_vr_points, avg_vr_rot_mtxs)):
		x = time_point * i

		ly = avg_vr_points[0].dist(p) * 1000.0
		ay = avg_vr_rot_mtxs[0].dist(rm) * 1000.0

		linear_points[x_label].append(x)
		linear_points[ly_label].append(ly)

		angular_points[x_label].append(x)
		angular_points[ay_label].append(ay)

	linear_data  = pnd.DataFrame(linear_points)
	angular_data = pnd.DataFrame(angular_points)

	plotLinePlot(linear_data, x_label, ly_label, marker = 'o', yerr = 1.96 * lin_std_dev * 1000.0)
	plotLinePlot(angular_data, x_label, ay_label, marker = 'o', yerr = 1.96 * ang_std_dev * 1000.0)
	
if __name__ == '__main__':
	# makeVRLatencyPlot(['../data/2-22/vr_noise_data_%d_2-22.txt' % i for i in (1, 2, 3, 4, 5)])

	makeVRNoisePlot(['../data/2-22/vr_noise_data_%d_2-22.txt' % i for i in (1, 2, 3, 4, 5)])

	# makeVRDriftPlot('../data/2-27/vr_drift_data_2-27.txt', 1000, 2.5, err_fnames = ['../data/2-22/vr_noise_data_%d_2-22.txt' % i for i in (1, 2, 3, 4, 5)])
