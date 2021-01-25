
from utils import *
from speed_analysis import getSpeeds, getAngularSpeeds
from basic_plots import *

from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pnd
import re

class ThroughputDataPoint:
	def __init__(self, start_time, end_time, transfer, transfer_units, bandwidth, bandwidth_units):
		self.start_time = start_time
		self.end_time   = end_time

		self.transfer       = transfer
		self.transfer_units = transfer_units

		self.bandwidth       = bandwidth
		self.bandwidth_units = bandwidth_units

	# Returns bandwidth in Gbit/s
	def getThroughput(self):
		if self.bandwidth_units == 'Gbits/sec':
			return self.bandwidth
		if self.bandwidth_units == 'Mbits/sec':
			return self.bandwidth / 1000.0
		if self.bandwidth_units == 'Kbits/sec':
			return self.bandwidth / 1000.0 / 1000.0
		if self.bandwidth_units == 'bits/sec':
			return self.bandwidth / 1000.0 / 1000.0 / 1000.0

		raise Exception('Unrecognized bandwidth unit: ' + str(self.bandwidth_units))

	def getTransfer(self):
		if self.transfer_units == 'GBytes':
			return self.transfer
		if self.transfer_units == 'MBytes':
			return self.transfer / 1000.0
		if self.transfer_units == 'KBytes':
			return self.transfer / 1000.0 / 1000.0
		if self.transfer_units == 'Bytes':
			return self.transfer / 1000.0 / 1000.0 / 1000.0

		raise Exception('Unrecognized bandwidth unit: ' + str(self.transfer_units))

def getThroughputData(fname, duration_cutoff = 10.0):
	f = open(fname, 'r')

	int_re   = r'\d+'
	float_re = r'(?:(?:\d*\.\d+)|(?:\d+\.?))'
	bytes_re = r'(?:K|M|G)?Bytes'
	bits_re  = r'(?:K|M|G)?bits'

	full_re = r'(?P<start_time>%s)\-(?P<end_time>%s)\s+sec\s+(?P<transfer>%s)\s+(?P<transfer_units>%s)\s+(?P<bandwidth>%s)\s+(?P<bandwidth_units>%s\/sec)' \
			% (float_re, float_re, float_re, bytes_re, float_re, bits_re)

	matcher = re.compile(full_re)

	data = []
	full_dp = None
	for i, line in enumerate(f):
		match = matcher.search(line)
		if match:
			start_time = float(match.group('start_time'))
			end_time   = float(match.group('end_time'))

			transfer       = float(match.group('transfer'))
			transfer_units = match.group('transfer_units')

			bandwidth       = float(match.group('bandwidth'))
			bandwidth_units = match.group('bandwidth_units')

			this_dp = ThroughputDataPoint(start_time, end_time, transfer, transfer_units, bandwidth, bandwidth_units)

			if end_time - start_time > duration_cutoff:
				if full_dp is not None:
					raise Exception('More than one data point has a duration more than the supplied cutoff.')

				full_dp = this_dp
			else:
				data.append(this_dp)

	if full_dp is None:
		raise Exception('No data point has a duration more than the supplied cutoff.')

	return data, full_dp

def getPointsForSpeedPlot(vr_data, throughput_data, data_type = 'linear', rotation_axis = None):
	if data_type == 'linear':
		zero_thresh = 0.015

		speed_data = getSpeeds(vr_data, min_time = 50)
	elif data_type == 'angular':
		zero_thresh = 0.5

		speed_data = getAngularSpeeds(vr_data, min_time = 50, rotation_axis = rotation_axis)
	else:
		raise Exception('Unknown data_type: ' + str(data_type))

	peaks = identifySpeedPeaks(speed_data, throughput_data, zero_thresh = zero_thresh, data_type = data_type)

	# debugPeakPlot(peaks)

	scatter_points = [(peak.maxSpeed(), peak.avgThroughput()) for peak in peaks]
	return scatter_points

class Peak:
	def __init__(self, speeds):
		self.raw_speeds = speeds
		self.raw_throughput_data = None

	def startTime(self):
		return min(dp.start_time for dp in self.raw_speeds)

	def endTime(self):
		return max(dp.end_time for dp in self.raw_speeds)

	def maxSpeed(self):
		return max(dp.speed for dp in self.raw_speeds)

	def avgThroughput(self):
		return sum(dp.getThroughput() for dp in self.raw_throughput_data) / float(len(self.raw_throughput_data))

	def maxThroughput(self):
		return max(dp.getThroughput() for dp in self.raw_throughput_data)

def identifySpeedPeaks(speed_data, throughput_data, zero_thresh = 0.015, data_type = 'linear'):
	init_peaks = []
	this_group = []

	if data_type == 'linear':
		positive_dir = Vec(1.0, 0.0, 0.0)
	elif data_type == 'angular':
		positive_dir = Vec(0.0, 0.0, 1.0)
	prev_dir = None

	for dp in speed_data:
		this_dot = positive_dir.dot(dp.delta)
		this_dir = 1 if this_dot >= 0.0 else -1

		if dp.speed <= zero_thresh or (prev_dir is not None and prev_dir != this_dir):
			if len(this_group) > 0:
				init_peaks.append(Peak(this_group))
				this_group = []
				prev_dir = None
		else:
			this_group.append(dp)
			prev_dir = this_dir


	if len(this_group) > 0:
		init_peaks.append(Peak(this_group))
		this_group = []

	final_peaks = []
	for peak in init_peaks:
		start_time = peak.startTime()
		end_time   = peak.endTime()

		peak.raw_throughput_data = [dp for dp in throughput_data if any(t >= start_time and t <= end_time for t in (dp.start_time, dp.end_time))]

		if len(peak.raw_throughput_data) > 0:
			final_peaks.append(peak)

	return final_peaks

def debugSpeedPlot(vr_data, speeds):
	plt.plot([(dp.time - vr_data[0].time) / 1000.0 for dp in vr_data], [dp.tvec.x for dp in vr_data], label = 'Position (m)')
	plt.plot([dp.start_time for dp in speeds], [dp.speed for dp in speeds], label = 'Speed (m/s)')

	plt.legend()

	plt.xlabel('Time (s)')

	plt.show()

def debugPeakPlot(peaks):
	for peak in peaks:
		plt.plot([dp.start_time for dp in peak.raw_speeds], [dp.speed for dp in peak.raw_speeds], label = 'Speed (m/s)')
		plt.plot([dp.start_time for dp in peak.raw_speeds], [peak.avgThroughput() for dp in peak.raw_speeds], label = 'Throughput (Gb/s')

	plt.xlabel('Time (s)')

	plt.show()


from scipy.optimize import least_squares

class ThroughputLine:
	def __init__(self, x0, y0, x1, y1):
		self.x0, self.y0 = x0, y0
		self.x1, self.y1 = x1, y1

	def __call__(self, x):
		if x < self.x0:
			return self.y0

		if x > self.x1:
			return self.y1

		m = (self.y1 - self.y0) / (self.x1 - self.x0)
		b = self.y0 - m * self.x0

		return m * x + b

def fitThroughputLine(scatter_points, init_x0, init_x1):
	fit_type = 'just_xs'
	fixed_y0 = 9.4
	fixed_y1 = 0.0

	def computeError(vs):
		if fit_type == 'just_xs':
			x0, x1 = vs
			y0, y1 = fixed_y0, fixed_y1

		if fit_type == 'all':
			x0, y0, x1, y1 = vs

		if x1 < x0:
			return [float('inf')] * len(scatter_points)

		this_line = ThroughputLine(x0, y0, x1, y1)

		errs = [y - this_line(x) for x, y in scatter_points]
		return errs

	if fit_type == 'just_xs':
		init_guess = [init_x0, init_x1]
		bounds = ([init_x0, init_x0], [init_x1, init_x1])
	if fit_type == 'all':
		init_guess = [init_x0, fixed_y0, init_x1, fixed_y1]
		bounds = ([init_x0, 0.0, init_x0, 0.0], [init_x1, 10.0, init_x1, 10.0])

	rv = least_squares(computeError, init_guess, bounds = bounds)

	if fit_type == 'just_xs':
		x0, x1 = rv.x
		y0, y1 = fixed_y0, fixed_y1

	if fit_type == 'all':
		x0, y0, x1, y1 = rv.x

	# print x0, y0, x1, y1

	final_line = ThroughputLine(x0, y0, x1, y1)

	return final_line

if __name__ == '__main__':
	movement_type = 'dynamic'
	smooth_speeds = False

	max_speed_throughput_equivalence = 10.0
	convert_linear_speed_to_cm = True

	data_type = 'linear'
	vr_data_fname = '../data/7-20/vr_data-test_7-20_model_dynamic_linear.txt'
	throughput_data_fname = '../data/7-20/thr_data-test_7-20_model_dynamic_linear.txt'
	gap_size = 3.0
	start_range, end_range = None, None
	speed_filter = 0.1

	# data_type = 'angular'
	# vr_data_fname = '../data/7-27/vr_data-test_7-27_model_angular.txt'
	# throughput_data_fname = '../data/7-27/thr_data-test_7-27_model_angular.txt'
	# gap_size = 3.0
	# start_range, end_range = None, None

	# data_type = 'mixed'
	# vr_data_fname = '../data/8-5/vr_data-test_8-5_free.txt'
	# throughput_data_fname = '../data/8-5/thr_data-test_8-5_free.txt'
	# gap_size = None
	# start_range, end_range = 140.0, 190.0
	
	vr_data = getVRData(vr_data_fname)
	throughput_data, total_throughput = getThroughputData(throughput_data_fname)

	max_time = 300
	if max_time is not None:
		vr_start_time = min(dp.time for dp in vr_data)
		old_vr_len = len(vr_data)
		vr_data = [dp for dp in vr_data if (dp.time - vr_start_time) / 1000.0 <= max_time]
		new_vr_len = len(vr_data)

		thr_start_time = min(dp.start_time for dp in throughput_data)

		old_thr_len = len(throughput_data)
		throughput_data = [dp for dp in throughput_data if dp.start_time - thr_start_time <= max_time]
		new_thr_len = len(throughput_data)

		print('Removed', old_vr_len  - new_vr_len,  'VR data points')
		print('Removed', old_thr_len - new_thr_len, 'THR data points')

	if gap_size is not None:
		low_val = 1.0
		current_start = None
		current_end = None
		low_ranges = []
		for tdp in throughput_data:
			if tdp.getThroughput() < low_val and current_start is None:
				current_start = tdp.start_time

			if tdp.getThroughput() >= low_val and current_start is not None and current_end is None:
				current_end = tdp.end_time

				low_ranges.append((current_start, current_end))

				current_start, current_end = None, None

		expanded_ranges = []
		for s, e in low_ranges:
			if s + gap_size < e - gap_size:
				expanded_ranges.append((s + gap_size, e - gap_size))

		vr_start_time = min(dp.time for dp in vr_data)

		new_vr_data = []
		for vr_dp in vr_data:
			this_time = (vr_dp.time - vr_start_time) / 1000.0
			this_ranges = [(s, e) for s, e in expanded_ranges if s < this_time]
			if any(this_time >= s and this_time <= e for s, e in this_ranges):
				continue

			new_time = this_time - sum(e - s for s, e in this_ranges)
			vr_dp.time = int(new_time * 1000.0)
			new_vr_data.append(vr_dp)

		new_throughput_data = []
		for tp_dp in throughput_data:
			this_ranges = [(s, e) for s, e in expanded_ranges if s < tp_dp.start_time]
			if any(tp_dp.start_time >= s and tp_dp.start_time <= e for s, e in this_ranges):
				continue

			new_start_time = tp_dp.start_time - sum(e - s for s, e in this_ranges)
			new_end_time   = tp_dp.end_time   - sum(e - s for s, e in this_ranges)

			tp_dp.start_time = new_start_time
			tp_dp.end_time   = new_end_time

			new_throughput_data.append(tp_dp)

		vr_data = new_vr_data
		throughput_data = new_throughput_data

	if start_range is not None and end_range is not None:
		start_vr_time = min(vr_dp.time for vr_dp in vr_data)
		new_vr_data = []
		for vr_dp in vr_data:
			this_time = vr_dp.time - start_vr_time
			if this_time >= start_range * 1000.0 and this_time <= end_range * 1000.0:
				new_vr_data.append(vr_dp)
		vr_data = new_vr_data

		new_throughput_data = []
		for tp_dp in throughput_data:
			if tp_dp.start_time >= start_range and tp_dp.start_time <= end_range:
				tp_dp.start_time = tp_dp.start_time - start_range
				tp_dp.end_time   = tp_dp.end_time   - start_range
				new_throughput_data.append(tp_dp)
		throughput_data = new_throughput_data

	print(len(vr_data), len(throughput_data))

	if movement_type == 'static':
		# Confirm that VR data is mostly static
		speed_data = getSpeeds(vr_data, min_time = 50)

		# debugSpeedPlot(vr_data, speed_data)

		max_speed = max(abs(dp.speed) for dp in speed_data)

		if max_speed >= 0.025:
			raise Exception('Static movement showed a speed greater than threshold')

		# Plot the throughput over time
		x_label = 'Time (s)'
		y_label = 'Throughput (Gb/s)'

		full_data = pnd.DataFrame({x_label: [dp.start_time for dp in throughput_data],
								   y_label: [dp.getThroughput() for dp in throughput_data]})

		plotLinePlot(full_data, x_label, y_label, y_limits = (0.0, 10.0))

	if movement_type == 'dynamic' and data_type in ['linear', 'angular']:
		x_label = 'Time (s)'
		y2_label = 'Throughput (Gb/s)'
		if data_type == 'linear':
			y1_label = 'Linear Speed (m/s)'
			if convert_linear_speed_to_cm:
				y1_label = 'cm/s'
		if data_type == 'angular':
			y1_label = 'degree/s'
		legend_label = 'Legend'

		y2_tick_values = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
		
		if data_type == 'linear':
			speed_type = 'Speed'
		if data_type == 'angular':
			speed_type = 'Speed'
		throughput_type = 'Throughput'

		# Raw Plot
		if data_type == 'linear':
			speed_data = getSpeeds(vr_data, min_time = 50)
		if data_type == 'angular':
			speed_data = getAngularSpeeds(vr_data, min_time = 50)

		max_speed = None
		throughput_filter = 9.0
		for sdp in speed_data:
			this_throughput = next((tdp.getThroughput() for tdp in throughput_data if tdp.start_time <= sdp.start_time and sdp.start_time <= tdp.end_time), None)
			if this_throughput is not None and this_throughput >= throughput_filter:
				if max_speed is None or sdp.speed > max_speed:
					max_speed = sdp.speed

		print 'Maximum speed with near-optimal throughput is', max_speed

		if smooth_speeds:
			speed_data = [dp for i, dp in enumerate(speed_data)
				if i <= 0 or
				   i >= len(speed_data) - 1 or
				   (dp.speed > speed_data[i - 1].speed and dp.speed > speed_data[i + 1].speed)]

		if convert_linear_speed_to_cm and data_type == 'linear':
			for dp in speed_data:
				dp.speed = dp.speed * 100.0

		max_speed = max(dp.speed for dp in speed_data)
		throughput_factor = max_speed / max_speed_throughput_equivalence

		y2_tick_labels = map(int, y2_tick_values)
		y2_tick_values = [v * throughput_factor for v in y2_tick_values]

		raw_debug_data = [(sdp.start_time, sdp.speed, speed_type) for sdp in speed_data] + [(tdp.start_time, tdp.getThroughput() * throughput_factor, throughput_type) for tdp in throughput_data]

		full_debug_data = pnd.DataFrame({x_label : [x for x, _, _ in raw_debug_data], y1_label : [y for _, y, _ in raw_debug_data], legend_label: [n for _, _, n in raw_debug_data]})

		plotLinePlot(full_debug_data, x_label, y1_label, hue = legend_label,
						second_y_label = y2_label, inv_y2_func = lambda v: v / throughput_factor,
						y2_tick_labels = y2_tick_labels, y2_tick_values = y2_tick_values)

		# Peak Plot
		scatter_points = getPointsForSpeedPlot(vr_data, throughput_data, data_type = data_type, rotation_axis = None)
	
		full_scatter_data = pnd.DataFrame({'Linear Speed (m/s)': [x for x, _ in scatter_points], 'Throughput (Gb/s)': [y for _, y in scatter_points]})

		alpha_value = 0.75
		point_size  = 50

		plotScatterPlot(full_scatter_data, 'Linear Speed (m/s)', 'Throughput (Gb/s)', alpha = alpha_value, point_size = point_size)

	if movement_type == 'dynamic' and data_type == 'mixed':
		x_label = 'Time (s)'
		y1_label = 'cm/s | deg/s'
		y2_label = 'Throughput (Gb/s)'
		legend_label = 'Data Type'
		
		linear_speed_type = 'Linear (cm/s)'
		angular_speed_type = 'Angular (degree/s)'
		throughput_type = 'Throughput'

		linear_speed_data = getSpeeds(vr_data, min_time = 50)
		angular_speed_data = getAngularSpeeds(vr_data, min_time = 50)

		# Output maximum speed where throughput is above 9 Gbps
		linear_max_speed = None
		throughput_filter = 9.0
		for sdp in linear_speed_data:
			this_throughput = next((tdp.getThroughput() for tdp in throughput_data if tdp.start_time <= sdp.start_time and sdp.start_time <= tdp.end_time), None)
			if this_throughput is not None and this_throughput >= throughput_filter:
				if linear_max_speed is None or sdp.speed > linear_max_speed:
					linear_max_speed = sdp.speed

		print 'Maximum linear speed with near-optimal throughput is', linear_max_speed * 100.0, 'cm/s'

		angular_max_speed = None
		throughput_filter = 9.0
		for sdp in angular_speed_data:
			this_throughput = next((tdp.getThroughput() for tdp in throughput_data if tdp.start_time <= sdp.start_time and sdp.start_time <= tdp.end_time), None)
			if this_throughput is not None and this_throughput >= throughput_filter:
				if angular_max_speed is None or sdp.speed > angular_max_speed:
					angular_max_speed = sdp.speed

		print 'Maximum speed with near-optimal throughput is', angular_max_speed, 'degree/s'

		if smooth_speeds:
			linear_speed_data = [dp for i, dp in enumerate(linear_speed_data)
				if i <= 0 or
				   i >= len(linear_speed_data) - 1 or
				   (dp.speed > linear_speed_data[i - 1].speed and dp.speed > linear_speed_data[i + 1].speed)]

			angular_speed_data = [dp for i, dp in enumerate(angular_speed_data)
				if i <= 0 or
				   i >= len(angular_speed_data) - 1 or
				   (dp.speed > angular_speed_data[i - 1].speed and dp.speed > angular_speed_data[i + 1].speed)]

		if convert_linear_speed_to_cm:
			for dp in linear_speed_data:
				dp.speed = dp.speed * 100.0

		max_linear_speed = max(dp.speed for dp in linear_speed_data)
		linear_speed_factor = (max_speed_throughput_equivalence / 2.0 * 0.95) / max_linear_speed

		angular_baseline = max_speed_throughput_equivalence / 2.0
		max_angular_speed = max(dp.speed for dp in angular_speed_data)
		angular_speed_factor = (max_speed_throughput_equivalence - angular_baseline) / max_angular_speed

		linear_ticks = [0.0, 15.0, 30.0]
		angular_ticks = [0.0, 15.0, 30.0]

		y1_tick_labels = map(int, linear_ticks + angular_ticks)
		y1_tick_values = [v * linear_speed_factor for v in linear_ticks] + \
						 [v * angular_speed_factor + angular_baseline for v in angular_ticks]

		y2_tick_labels = [0, 2, 4, 6, 8, 10]
		y2_tick_values = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

		raw_debug_data = [(ldp.start_time, ldp.speed * linear_speed_factor, linear_speed_type) for ldp in linear_speed_data] + \
						 [(tdp.start_time, tdp.getThroughput(), throughput_type) for tdp in throughput_data] + \
						 [(adp.start_time, adp.speed * angular_speed_factor + angular_baseline, angular_speed_type) for adp in angular_speed_data]

		full_debug_data = pnd.DataFrame({x_label : [x for x, _, _ in raw_debug_data], y1_label : [y for _, y, _ in raw_debug_data], legend_label: [n for _, _, n in raw_debug_data]})

		plotLinePlot(full_debug_data, x_label, y1_label, hue = legend_label,
						second_y_label = y2_label, inv_y2_func = lambda v: v, hline = max_speed_throughput_equivalence / 2.0,
						y1_tick_labels = y1_tick_labels, y1_tick_values = y1_tick_values,
						y2_tick_labels = y2_tick_labels, y2_tick_values = y2_tick_values)


else:
	# data_description = [('../data/2-26/vr_data-interpolation_angular_2-26.txt', '../data/2-26/throughput_data-interpolation_angular_2-26.txt', 'Interpolation', [10, 11, 12, 13, 14, 15, 16, 17, 18, 37, 38, 84, 85, 126, 127, 128, 129, 130, 161, 198, 203, 204, 207, 208, 209, 210, 228, 230, 231, 238, 239]),
	# 					('../data/3-5/vr_data-mod_ang_3-5.txt', '../data/3-5/thr_data-mod_ang_3-5.txt', 'Model', [54, 73, 90, 91, 92, 114, 184, 185, 186, 187, 188, 189, 190])]

	# data_type = 'angular'
	# rotation_axis = None

	# data_description = [('../data/2-26/vr_data-interpolation_linear_2-26.txt', '../data/2-26/throughput_data-interpolation_linear_2-26.txt', 'Interpolation', [285, 412]),
	# 					('../data/3-5/vr_data-mod_lin_3-5.txt', '../data/3-5/thr_data-mod_lin_3-5.txt', 'Model', [])]

	# data_type = 'linear'
	# rotation_axis = None

	all_points = []
	line_points = []
	for vr_data_fname, throughput_data_fname, name, ignored_indexes in data_description:
		vr_data = getVRData(vr_data_fname)
		throughput_data, total_throughput = getThroughputData(throughput_data_fname)

		scatter_points = getPointsForSpeedPlot(vr_data, throughput_data, data_type = data_type, rotation_axis = rotation_axis)

		for i, (x, y) in enumerate(scatter_points):
			if i in ignored_indexes:
				continue

			all_points.append((x, y, name))

		max_x = max(x for x, _ in scatter_points)
		this_line = fitThroughputLine(scatter_points, 0.0, max_x)

		line_points += [(0.0, this_line.y0, name), (this_line.x0, this_line.y0, name), (this_line.x1, this_line.y1, name), (max_x, this_line.y1, name)]

	print_stats = False
	if print_stats:
		print(data_type)
		# Find maximum speed with throughput above 9.0
		optimal_throughput = [(speed, throughput, name) for (speed, throughput, name) in all_points if throughput >= 9.0]
		optimal_throughput_by_name = defaultdict(list)
		for speed, throughput, name in optimal_throughput:
			optimal_throughput_by_name[name].append((speed, throughput, name))

		for name, values in optimal_throughput_by_name.items():
			print('Maximum speed with optimal throughput for', str(name) + ':', max(values))

		# Find minimum speed with throughput below 0.1
		zero_throughput = [(speed, throughput, name) for (speed, throughput, name) in all_points if throughput <= 0.1]
		zero_throughput_by_name = defaultdict(list)
		for speed, throughput, name in zero_throughput:
			zero_throughput_by_name[name].append((speed, throughput, name))

		for name, values in zero_throughput_by_name.items():
			print('Minimum speed with zero throughput for', str(name) + ':', min(values))

		quit()

	if data_type == 'linear':
		x_label = 'Linear Speed (m/s)'
	if data_type == 'angular':
		x_label = 'Angular Speed (degree/s)'

	y_label = 'Throughput (Gb/s)'
	legend_label = 'Tracking Method'

	full_scatter_data = pnd.DataFrame({x_label: [x for x, _, _ in all_points], y_label: [y for _, y, _ in all_points], legend_label: [name for _, _, name in all_points]})
	full_line_data = pnd.DataFrame({x_label: [x for x, _, _ in line_points], y_label: [y for _, y, _ in line_points], legend_label: [name for _, _, name in line_points]})

	plot_type = 'one_plot_binned'

	alpha_value = 0.75
	point_size  = 50

	if plot_type == 'one_plot_with_line':
		plotScatterAndLinePlot(full_scatter_data, full_line_data, x_label, y_label, hue = legend_label, alpha = alpha_value, point_size = point_size)

	if plot_type == 'one_plot':
		plotScatterPlot(full_scatter_data, x_label, y_label, hue = legend_label, alpha = alpha_value, point_size = point_size)

	if plot_type == 'one_plot_binned':
		num_bins = 10

		min_x = min(x for x, _, _ in all_points)
		max_x = max(x for x, _, _ in all_points)

		bin_step = (max_x - min_x) / num_bins

		bin_limits = [(min_x + bin_step * i, min_x + bin_step * (i + 1)) for i in range(num_bins)]

		legend_labels = set(name for _, _, name in all_points)

		bins = {(i, ll): [] for i in range(len(bin_limits)) for ll in legend_labels}

		for x, y, name in all_points:
			found = False
			for i in range(len(bin_limits)):
				bin_min, bin_max = bin_limits[i]
				if bin_min <= x and bin_max >= x:
					found = True
					break

			if not found:
				raise Exception('WHAT?!?!?!?!?')

			bins[(i, name)].append((x, y, name))

		binned_points = []
		for (i, ll), vals in sorted(bins.items()):
			if len(vals) == 0:
				continue

			bin_min, bin_max = bin_limits[i]
			bin_x = (bin_max + bin_min) / 2
			bin_y = sum(y for _, y, _ in vals) / float(len(vals))

			binned_points.append((bin_x, bin_y, ll))

		binned_scatter_data = pnd.DataFrame({x_label: [x for x, _, _ in binned_points], y_label: [y for _, y, _ in binned_points], legend_label: [name for _, _, name in binned_points]})
		
		plotLinePlot(binned_scatter_data, x_label, y_label, hue = legend_label, alpha = alpha_value, marker = 'o')

	if plot_type == 'two_plots_with_line':
		# Only works for the current data
		if data_type == 'linear':
			x_min = -0.0034120442210750883
			x_max = 0.39127000212342389
		if data_type == 'angular':
			x_min = -2.0664030612749795
			x_max = 55.005619063078598

		plotScatterAndLinePlot(full_scatter_data[full_scatter_data[legend_label] == 'Model'],         full_line_data[full_line_data[legend_label] == 'Model'],         x_label, y_label, alpha = alpha_value, point_size = point_size)
		plotScatterAndLinePlot(full_scatter_data[full_scatter_data[legend_label] == 'Interpolation'], full_line_data[full_line_data[legend_label] == 'Interpolation'], x_label, y_label, alpha = alpha_value, point_size = point_size)

	if plot_type == 'two_plots':
		if data_type == 'linear':
			x_min = -0.0034120442210750883
			x_max = 0.39127000212342389
		if data_type == 'angular':
			x_min = -2.0664030612749795
			x_max = 55.005619063078598

		plotScatterPlot(full_scatter_data[full_scatter_data[legend_label] == 'Model'],         x_label, y_label, x_limits = (x_min, x_max), alpha = alpha_value, point_size = point_size)
		plotScatterPlot(full_scatter_data[full_scatter_data[legend_label] == 'Interpolation'], x_label, y_label, x_limits = (x_min, x_max), alpha = alpha_value, point_size = point_size)
