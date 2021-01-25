

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))

def setStyle():
	sns.set(style = 'ticks', rc = {'figure.figsize':(5.0, 2.0), 'figure.dpi': 300})
	sns.set_context("paper", font_scale = 1.0)

	# plt.subplots_adjust(bottom = 0.25, left = 0.10, right = 0.9, top = 0.95)

	# sns.set(style = 'ticks', rc = {'figure.figsize':(2.0,2.0), 'figure.dpi': 300})
	# sns.set_context("paper", font_scale = 0.9)

	# plt.subplots_adjust(bottom = 0.25, left = 0.3, right = 0.975, top = 0.95)

	# sns.set(style = 'ticks', rc = {'figure.figsize':(3.0,2.0), 'figure.dpi': 300})
	# sns.set_context("paper", font_scale = 0.9)

	# plt.subplots_adjust(bottom = 0.25, left = 0.2, right = 0.975, top = 0.95)

def test():
	sns.set(style = "ticks")

	# Load the example dataset for Anscombe's quartet
	df = sns.load_dataset("anscombe")
	print(type(df))

	# Show the results of a linear regression within each dataset
	sns.lmplot(x = "x", y = "y", col = "dataset", hue = "dataset", data = df,
				col_wrap = 2, ci = None, palette = "muted", height = 4,
				scatter_kws = {"s": 50, "alpha": 1})

	plt.show()

def plotCDF(data, x_label = ''):
	setStyle()

	kwargs = {'cumulative': True}
	sns.distplot(data, hist_kws = kwargs, kde_kws = kwargs)

	if x_label != '':
		plt.xlabel(x_label)
	plt.ylabel('CDF')

	sns.despine()
	plt.show()

def plotPDF(data, x_label = ''):
	setStyle()
	sns.distplot(data, hist = True, kde = True, norm_hist = True)

	if x_label != '':
		plt.xlabel(x_label)
	plt.ylabel('Probability Density Function')

	sns.despine()
	plt.show()

def plotScatterPlot(full_data, x_label, y_label, hue = None, title = '', alpha = 1.0, point_size = 100, x_limits = None):
	setStyle()
	sns.scatterplot(x = x_label, y = y_label, hue = hue, data = full_data, alpha = alpha, s = point_size)

	if x_limits is not None:
		plt.xlim(*x_limits)

	plt.title(title)

	sns.despine()
	plt.show()

def plotLinePlot(full_data, x_label, y_label, hue = None, marker = None,
			yerr = None, alpha = 1.0, y_limits = None, title = None,
			y1_tick_labels = None, y1_tick_values = None,
			second_y_label = None, inv_y2_func = None, y2_tick_labels = None, y2_tick_values = None,
			hline = None):
	setStyle()

	# current_palette = sns.color_palette()
	# red = (current_palette[3][0] * 1.1, current_palette[3][1] * 0.7, current_palette[3][2] * 0.7)
	# new_palette = [current_palette[0], red, current_palette[2]]


	ax1 = sns.lineplot(x = x_label, y = y_label, hue = hue, marker = marker, data = full_data, alpha = alpha, size = 10.0, legend = False) #, palette = new_palette)

	if y_limits is not None:
		plt.ylim(*y_limits)
	
	if yerr is not None:
		plt.errorbar(full_data.get(x_label), full_data.get(y_label), fmt = '', ecolor = 'steelblue', capsize = 4, yerr = yerr)

	if title is not None:
		plt.title(title)

	if y1_tick_labels is not None and y1_tick_values is not None:
		ax1.set_yticks(y1_tick_values)
		ax1.set_yticklabels(y1_tick_labels)

	if second_y_label is not None and inv_y2_func is not None:
		ax2 = ax1.twinx()
		ax2.set_ylim(ax1.get_ylim())

		if y2_tick_labels is not None and y2_tick_values is not None:
			ax2.set_yticks(y2_tick_values)
			ax2.set_yticklabels(y2_tick_labels)
		else:
			ax2.set_yticklabels(map(int, np.round(inv_y2_func(ax1.get_yticks()), 0)))
		ax2.set_ylabel(second_y_label)

	handles, labels = ax1.get_legend_handles_labels()

	if len(handles) == 4 and len(labels) == 4:
		new_handles = [handles[2], handles[1], handles[3]]
		new_labels  = [labels[2],  labels[1],  labels[3]]
	elif len(handles) == 5 and len(labels) == 5:
		new_handles = [handles[1], handles[3], handles[4], handles[2]]
		new_labels  = [labels[1], labels[3], labels[4], labels[2]]
	else:
		new_handles = handles[1:]
		new_labels  = labels[1:]

	# ax1.legend(handles = new_handles, labels = new_labels,
	# 			bbox_to_anchor=(0., 1.1, 1., .102), loc='lower center',
	# 			borderaxespad = 0.0, ncol = 2)

	# if hline is not None:
	c = 0.3

	# ax1.axhline(0.0, color = (c, c, c), linewidth = 0.5, linestyle = '--')
	# ax1.axhline(39.51500728, color = (c, c, c), linewidth = 0.5, linestyle = '--')

	sns.despine() #right = False)
	plt.show()

def plotBarPlot(full_data, x_label, y_label, xticklabels = None):
	setStyle()
	b = sns.barplot(x = x_label, y = y_label, data = full_data)

	if xticklabels is not None:
		b.set(xticklabels = xticklabels)

	sns.despine()
	plt.show()

def plotScatterAndLinePlot(full_scatter_data, full_line_data, x_label, y_label, hue = None, title = '', alpha = 1.0, point_size = 100):
	setStyle()
	sns.scatterplot(x = x_label, y = y_label, hue = hue, data = full_scatter_data, alpha = alpha, s = point_size)
	sns.lineplot(x = x_label, y = y_label, hue = hue, data = full_line_data, legend = False)

	plt.title(title)

	sns.despine()
	plt.show()
