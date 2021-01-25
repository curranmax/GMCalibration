
from basic_plots import *
from speed_analysis import getSpeeds, getAngularSpeeds
from utils import *

import pandas as pnd

class ReceivedPowerPoint:
	# Power should be in dBm
	# time_val should be in seconds
	def __init__(self, power, time_val):
		self.power = power
		self.time_val = time_val

def getReceivedPowerData(received_power_data_fname):
	f = open(received_power_data_fname, 'r')

	data = []
	for i, line in enumerate(f):
		if i <= 3:
			continue
		power_str, time_str = line.split()

		power_watt = float(power_str)
		try:
			power_dbm = 10.0 * math.log(power_watt * 1000.0, 10.0)
		except:
			continue

		time_ms = float(time_str)
		time_s = time_ms / 1000.0

		data.append(ReceivedPowerPoint(power_dbm, time_s))

	return data

if __name__ == '__main__':
	movement_type = 'dynamic'

	# data_type = 'linear'
	# vr_data_fname = '../data/7-25/vr_data_model_dynamic_linear.txt'
	# received_power_data_fname = '../data/7-25/power_model_dynamic_linear.CSV'
	# min_time, max_time = 165.0, 250.0
	# y1_tick_labels = [0, 30, 60]
	# y2_tick_labels = [-16, -24, -32]
	# zero_buffer = 5.0

	# data_type = 'angular'
	# vr_data_fname = '../data/7-25/vr_data_model_dynamic_angular.txt'
	# received_power_data_fname = '../data/7-25/power_model_dynamic_angular.CSV'
	# min_time, max_time = 105.0, 200.0
	# y1_tick_labels = [0, 50, 100, 150]
	# y2_tick_labels = [-20, -40, -60]
	# zero_buffer = 5.0

	# data_type = 'linear'
	# vr_data_fname = '../data/8-5/vr_data_power_8-5_free.txt'
	# received_power_data_fname = '../data/8-5/power_model_free.CSV'
	# min_time, max_time = 150.0, 215.0
	# y1_tick_labels = [0, 25, 50, 75]
	# y2_tick_labels = [-15, -30, -45, -60]
	# zero_buffer = 0.0

	# data_type = 'angular'
	# vr_data_fname = '../data/8-5/vr_data_power_8-5_free.txt'
	# received_power_data_fname = '../data/8-5/power_model_free.CSV'
	# min_time, max_time = 230.0, 285.0
	# y1_tick_labels = [0, 50, 100, 150]
	# y2_tick_labels = [-15, -35, -55, -75]
	# zero_buffer = 0.0

	data_type = 'mixed'
	vr_data_fname = '../data/8-5/vr_data_power_8-5_free.txt'
	received_power_data_fname = '../data/8-5/power_model_free.CSV'
	min_time, max_time = 230.0, 269.0
	y1_tick_labels = [0, 15, 30]
	y2_tick_labels = [-20, -40]
	y3_tick_labels = [0, 75, 150]
	zero_buffer = 5.0

	smooth_speeds = False

	if vr_data_fname is not None:
		vr_data = getVRData(vr_data_fname)
	else:
		vr_data = None
	
	received_power_data = getReceivedPowerData(received_power_data_fname)

	if min_time is not None and max_time is not None:
		start_vr_time = min(vr_dp.time for vr_dp in vr_data)
		new_vr_data = []
		for vr_dp in vr_data:
			this_time = vr_dp.time - start_vr_time
			if this_time >= min_time * 1000.0 and this_time <= max_time * 1000.0:
				new_vr_data.append(vr_dp)
		vr_data = new_vr_data

		new_rp_data = []
		for rp_dp in received_power_data:
			if rp_dp.time_val >= min_time and rp_dp.time_val <= max_time:
				rp_dp.time_val = rp_dp.time_val - min_time
				new_rp_data.append(rp_dp)
		received_power_data = new_rp_data

	if movement_type == 'static':
		if vr_data is not None:
			# Confirm that VR data is mostly static
			speed_data = getSpeeds(vr_data, min_time = 50)

			# debugSpeedPlot(vr_data, speed_data)

			max_speed = max(abs(dp.speed) for dp in speed_data)

			if max_speed >= 0.025:
				raise Exception('Static movement showed a speed greater than threshold')

		# Plot the throughput over time
		x_label = 'Time (s)'
		y_label = 'Received Power (dBm)'

		full_data = pnd.DataFrame({x_label: [dp.time_val for dp in received_power_data],
								   y_label: [dp.power for dp in received_power_data]})

		plotLinePlot(full_data, x_label, y_label)

	if movement_type == 'dynamic' and data_type in ['linear', 'angular']:
		if data_type == 'linear':
			y1_label = 'Power (dBm) |   cm/s         '
			raw_factor = 100.0
		if data_type == 'angular':
			y1_label = 'Power (dBm) |   deg/s        '
			raw_factor = 1.0
		y2_label = 'Received Power (dBm)'

		# Raw Plot
		if data_type == 'linear':
			speed_data = getSpeeds(vr_data, min_time = 50)
		if data_type == 'angular':
			speed_data = getAngularSpeeds(vr_data, min_time = 50)

		max_speed = max(sdp.speed * raw_factor for sdp in speed_data)
		max_rp, min_rp = max(rpdp.power for rpdp in received_power_data), min(rpdp.power for rpdp in received_power_data)

		print max_rp, min_rp

		# max_rp --> max_speed
		# min_rp --> 0.0
		# rp_convert = lambda v: max_speed * (float(v) - min_rp) / (max_rp - min_rp)
		# inv_rp_convert = lambda v: (max_rp - min_rp) * float(v) / max_speed + min_rp

		# max_rp --> -zero_buffer
		# min_rp --> -max_speed * relative_weight
		relative_weight = 1.0
		rp_convert = lambda v: (max_speed * relative_weight - zero_buffer) * (v - max_rp) / (max_rp - min_rp) - zero_buffer
		inv_rp_convert = lambda v: (max_rp - min_rp) * (v + zero_buffer) / (max_speed * relative_weight - zero_buffer) + max_rp

		# rp_convert = lambda v: v
		# inv_rp_convert = lambda v: v

		if y2_tick_labels is not None:
			y2_tick_values = map(rp_convert, y2_tick_labels)
		else:
			y2_tick_values = None

		if smooth_speeds:
			speed_data = [dp for i, dp in enumerate(speed_data)
				if i <= 0 or
				   i >= len(speed_data) - 1 or
				   (dp.speed > speed_data[i - 1].speed and dp.speed > speed_data[i + 1].speed)]

		sp_list = [(sdp.start_time, sdp.speed * raw_factor, y1_label) for sdp in speed_data]
		rp_list = [(rpdp.time_val, rp_convert(rpdp.power), y2_label) for rpdp in received_power_data]

		raw_data = sp_list + rp_list

		full_data = pnd.DataFrame({'Time (s)' : [x for x, _, _ in raw_data], y1_label : [y for _, y, _ in raw_data], 'Data Type': [n for _, _, n in raw_data]})

		plotLinePlot(full_data, 'Time (s)', y1_label, hue = 'Data Type',
						# second_y_label = y2_label, inv_y2_func = inv_rp_convert,
						# y2_tick_labels = y2_tick_labels, y2_tick_values = y2_tick_values,
						y1_tick_labels = y1_tick_labels + y2_tick_labels, y1_tick_values = y1_tick_labels + y2_tick_values)

	if movement_type == 'dynamic' and data_type == 'mixed':
		y1_label = 'dBm | cm/s | deg/s'
		y2_label = 'Received Power (dBm)'

		# Raw Plot
		linear_speed_data = getSpeeds(vr_data, min_time = 50)
		angular_speed_data = getAngularSpeeds(vr_data, min_time = 50)

		max_linear_speed = max(sdp.speed * 100.0 for sdp in linear_speed_data)
		max_angular_speed = max(sdp.speed for sdp in angular_speed_data)
		max_rp, min_rp = max(rpdp.power for rpdp in received_power_data), min(rpdp.power for rpdp in received_power_data)

		print(max_linear_speed)
		print(max_angular_speed)
		print(max_rp, min_rp)

		# max_rp --> -zero_buffer
		# min_rp --> -max_speed * rp_rw
		rp_rw = 1.0
		rp_convert = lambda v: (max_linear_speed * rp_rw - zero_buffer) * (v - max_rp) / (max_rp - min_rp) - zero_buffer
		inv_rp_convert = lambda v: (max_rp - min_rp) * (v + zero_buffer) / (max_linear_speed * rp_rw - zero_buffer) + max_rp

		# 0 --> max_linear_speed + as_buffer
		# max_angular_speed -->max_linear_speed * (1 + as_rw)
		as_rw = 1.0
		as_buffer = 5.0
		as_convert = lambda v: as_rw * max_linear_speed * v / max_angular_speed + max_linear_speed + as_buffer
		inv_as_convert = lambda v: max_linear_speed * (v - max_linear_speed - as_buffer) / (as_rw * max_linear_speed)

		# rp_convert = lambda v: v
		# inv_rp_convert = lambda v: v

		print(as_convert(0.0))

		y2_tick_values = map(rp_convert, y2_tick_labels)
		y3_tick_values = map(as_convert, y3_tick_labels)

		if smooth_speeds:
			raise Exception('AAAAA')

		lsp_list = [(sdp.start_time, sdp.speed * 100.0, 'Linear Speed (cm/s)') for sdp in linear_speed_data]
		asp_list = [(sdp.start_time, as_convert(sdp.speed), 'Angular Speed (deg/s)') for sdp in angular_speed_data]
		rp_list  = [(rpdp.time_val, rp_convert(rpdp.power), 'Received Power (dBm)') for rpdp in received_power_data]

		tmp_list = [(0.0, 0.0, 'Throughput (Gb/s)')]

		raw_data = lsp_list + rp_list + asp_list # + tmp_list

		full_data = pnd.DataFrame({'Time (s)' : [x for x, _, _ in raw_data], y1_label : [y for _, y, _ in raw_data], 'Data Type': [n for _, _, n in raw_data]})

		plotLinePlot(full_data, 'Time (s)', y1_label, hue = 'Data Type',
						y1_tick_labels = y1_tick_labels + y2_tick_labels + y3_tick_labels,
						y1_tick_values = y1_tick_labels + y2_tick_values + y3_tick_values)
