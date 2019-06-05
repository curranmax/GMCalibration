
#include "brute_force.h"

#include <curses.h>

#include <chrono>
#include <ctime>
#include <future>
#include <iostream>
#include <list>
#include <math.h>
#include <thread>
#include <vector>

#include <unistd.h>

#define USE_NCURSES true

class TestCase {
public:
	TestCase() :
			yaw_delta(0.0), pitch_delta(0.0), roll_delta(0.0), tx_delta(0.0), ty_delta(0.0), tz_delta(0.0), ida_delta(0.0), idb_delta(0.0), ily_delta(0.0), ilz_delta(0.0) {}
	~TestCase() {}

	TestCase(double _yaw_delta, double _pitch_delta, double _roll_delta, double _tx_delta, double _ty_delta, double _tz_delta, double _ida_delta, double _idb_delta, double _ily_delta, double _ilz_delta) :
			yaw_delta(_yaw_delta), pitch_delta(_pitch_delta), roll_delta(_roll_delta), tx_delta(_tx_delta), ty_delta(_ty_delta), tz_delta(_tz_delta), ida_delta(_ida_delta), idb_delta(_idb_delta), ily_delta(_ily_delta), ilz_delta(_ilz_delta) {}

	TestCase(const TestCase& tc) :
			yaw_delta(tc.yaw_delta), pitch_delta(tc.pitch_delta), roll_delta(tc.roll_delta), tx_delta(tc.tx_delta), ty_delta(tc.ty_delta), tz_delta(tc.tz_delta), ida_delta(tc.ida_delta), idb_delta(tc.idb_delta), ily_delta(tc.ily_delta), ilz_delta(tc.ilz_delta) {}

	const TestCase& operator=(const TestCase& tc) {
		yaw_delta   = tc.yaw_delta;
		pitch_delta = tc.pitch_delta;
		roll_delta  = tc.roll_delta;
		tx_delta    = tc.tx_delta;
		ty_delta    = tc.ty_delta;
		tz_delta    = tc.tz_delta;
		ida_delta   = tc.ida_delta;
		idb_delta   = tc.idb_delta;
		ily_delta   = tc.ily_delta;
		ilz_delta   = tc.ilz_delta;

		return *this;
	}

	std::vector<double> getErrors(const Matrix& rot_mtx, const Vec& tvec, const GMModel& gm_model, const std::vector<Dot>& dots, const SimpPlane& wall_plane) const {
		Matrix this_rot_mtx = rotMatrixFromYPR(rot_mtx.a1 + yaw_delta, rot_mtx.a2 + pitch_delta, rot_mtx.a3 + roll_delta);
		Vec this_tvec = Vec(tvec.x + tx_delta, tvec.y + ty_delta, tvec.z + tz_delta);

		Vec this_init_dir = vecFromAngle(gm_model.init_dir.a1 + ida_delta, gm_model.init_dir.a2 + idb_delta);
		Vec this_init_point = Vec(gm_model.init_point.x, gm_model.init_point.y + ily_delta, gm_model.init_point.z + ilz_delta);

		GMModel this_gm_model = gm_model.moveWithNewInitBeam(this_rot_mtx, this_tvec, this_init_dir, this_init_point);

		return computeErrors(this_gm_model, dots, wall_plane);
	}

	double yaw_delta, pitch_delta, roll_delta;
	double tx_delta, ty_delta, tz_delta;
	double ida_delta, idb_delta;
	double ily_delta, ilz_delta;
};

std::ostream& operator<<(std::ostream& ostr, const TestCase& tc) {
	ostr << "{rot_vals: (" << tc.yaw_delta << ", " << tc.pitch_delta << ", " << tc.roll_delta << "), " <<
			 "tvec_vals: (" << tc.tx_delta << ", " << tc.ty_delta << ", " << tc.tz_delta << "), " <<
			 "idir_vals: (" << tc.ida_delta << ", " << tc.idb_delta << "), " <<
			 "iloc_vals: (" << tc.ily_delta << ", " << tc.ilz_delta << ")}";
	return ostr;
}

std::vector<double> getValues(double range, double step, bool verbose, const std::string& prefix) {
	std::vector<double> values;
	for(double v = -range; v <= range; v += step) {
		values.push_back(v);
		if(fabs(step) <= 0.000001) {
			break;
		}
	}

	if(verbose) {
		std::cout << prefix << ": ";
		for(unsigned int i = 0; i < values.size(); ++i) {
			std::cout << values[i];
			if(i < values.size() - 1) {
				std::cout << ", ";
			} else {
				std::cout << std::endl;
			}
		}
	}

	return values;
}

bool increment(std::vector<int>& indexes, const std::vector<int>& lens) {
	for(int i = int(indexes.size()) - 1; i >= 0; --i) {
		if(indexes[i] + 1 < lens[i]) {
			indexes[i]++;
			return true;
		} else {
			indexes[i] = 0;
		}
	}
	return false;
}

std::vector<double> computeErrors(const GMModel& gm_model, const std::vector<Dot>& dots, const SimpPlane& wall_plane) {
	std::vector<double> errors;
	for(unsigned int i = 0; i < dots.size(); ++i) {
		auto beam_out = gm_model.getOutput(dots[i].gmh_val, dots[i].gmv_val);

		auto intersection_out = wall_plane.intersect(beam_out.beam_start, beam_out.beam_direction);

		errors.push_back(intersection_out.intersection_point.dist(dots[i].location));
	}
	return errors;
}

typedef struct {
	double average;
	double maximum;
	double least_squares;
} ErrorStats;

ErrorStats analyzeErrors(const std::vector<double>& errors) {
	bool first = true;
	ErrorStats stats;
	stats.average = 0.0;
	stats.maximum = 0.0;
	stats.least_squares = 0.0;

	for(unsigned int i = 0; i < errors.size(); ++i) {
		stats.average += errors[i];

		if(first || errors[i] > stats.maximum) {
			stats.maximum = errors[i];
			first = false;
		}

		stats.least_squares += errors[i] * errors[i];
	}

	stats.average /= double(errors.size());
	stats.least_squares /= double(errors.size());

	return stats;
}

ErrorStats getErrorStats(const TestCase& this_vals, const Matrix& rot_mtx, const Vec& tvec, const GMModel& gm_model, const std::vector<Dot>& dots, const SimpPlane& wall_plane) {
	std::vector<double> errors = this_vals.getErrors(rot_mtx, tvec, gm_model, dots, wall_plane);
	return analyzeErrors(errors);
}

void runBruteForceSearch(const std::vector<Dot>& dots, const GMModel& gm_model, const Matrix& rot_mtx, const Vec& tvec, const BruteForceParams& params, Timer& timer) {
	// Compute search values
	std::vector<double> rot_values   = getValues(params.rot_range,   params.rot_step,   true, "Rot values");
	std::vector<double> trans_values = getValues(params.trans_range, params.trans_step, true, "Trans values");
	std::vector<double> idir_values  = getValues(params.idir_range,  params.idir_step,  true, "Input dir values");
	std::vector<double> iloc_values  = getValues(params.iloc_range,  params.iloc_step,  true, "Input loc values");

	int num_iterations = pow(rot_values.size(), 3) * pow(trans_values.size(), 3) * pow(idir_values.size(), 2) * pow(iloc_values.size(), 2);

	std::cout << "Running " << num_iterations << " total iterations" << std::endl;

	std::vector<int> indexes = {0, 0, 0,
								0, 0, 0,
								0, 0,
								0, 0};
	std::vector<int> lens = {int(rot_values.size()), int(rot_values.size()), int(rot_values.size()),
							 int(trans_values.size()), int(trans_values.size()), int(trans_values.size()),
							 int(idir_values.size()), int(idir_values.size()),
							 int(iloc_values.size()), int(iloc_values.size())};

	SimpPlane wall_plane(Vec(0.0, 0.0, 1.0), Vec(0.0, 0.0, 0.0));

	bool first = true;
	TestCase best_avg_solution, best_max_solution, best_lsq_solution;
	ErrorStats best_avg_errs, best_max_errs, best_lsq_errs;

	if(!gm_model.init_dir.angle_set) {
		std::cerr << "Must set the angles of the initial direction for the burte force search" << std::endl;
		exit(1);
	}

	initscr();
	std::string progress_format = "Progress: %f %\n";
	std::string avg_dur_format  = "Avg dur:  %f ms\n";
	std::string exp_end_format  = "Exp end:  %s\n";
	std::string cur_error_format = "Cur %s error: {avg: %f mm, max %f mm}\n";

	// Progress: XX.XX %
	// Avg Dur:  0.XXX ms
	// Exp End:  XX:XX:XX
	// 
	// Cur avg error: {avg: XX.XX mm, max: XX.XX mm}
	// Cur max error: {avg: XX.XX mm, max: XX.XX mm}
	// Cur lsq error: {avg: XX.XX mm, max: XX.XX mm}

	printw(progress_format.c_str(), 0.0);
	printw(avg_dur_format.c_str(),  0.0);
	printw(exp_end_format.c_str(),  "");
	printw("\n");
	printw(cur_error_format.c_str(), "avg", 0.0, 0.0);
	printw(cur_error_format.c_str(), "max", 0.0, 0.0);
	printw(cur_error_format.c_str(), "lsq", 0.0, 0.0);
	refresh();

	int cur_iter = 0;
	int print_frequency = 100;

	auto start_bf = std::chrono::high_resolution_clock::now();
	while(true) {
		timer.start("bf_iteration");

		if(cur_iter % print_frequency == 0) {
			double avg_iter_dur = timer.getAverageDuration("bf_iteration");

			auto dur = std::chrono::seconds(int(num_iterations * avg_iter_dur));

			auto exp_end = start_bf + dur;
			auto exp_end_in_time_t = std::chrono::system_clock::to_time_t(exp_end);

			char buf[80];
			std::strftime(buf, sizeof(buf), "%A %b-%d %I:%M:%S %p", std::localtime(&exp_end_in_time_t));

			move(0, 0);
			printw(progress_format.c_str(), double(cur_iter) / double(num_iterations) * 100.0);
			printw(avg_dur_format.c_str(),  avg_iter_dur * 1000.0);
			printw(exp_end_format.c_str(),  buf);
			refresh();
		}
		cur_iter++;

		double yaw_delta = rot_values[indexes[0]], pitch_delta = rot_values[indexes[1]], roll_delta = rot_values[indexes[2]];
		double tx_delta = trans_values[indexes[3]], ty_delta = trans_values[indexes[4]], tz_delta = trans_values[indexes[5]];
		double ida_delta = idir_values[indexes[6]], idb_delta = idir_values[indexes[7]];
		double ily_delta = iloc_values[indexes[8]], ilz_delta = iloc_values[indexes[9]];

		TestCase this_vals(yaw_delta, pitch_delta, roll_delta, tx_delta, ty_delta, tz_delta, ida_delta, idb_delta, ily_delta, ilz_delta);

		std::vector<double> errors = this_vals.getErrors(rot_mtx, tvec, gm_model, dots, wall_plane);

		ErrorStats this_stats = analyzeErrors(errors);

		if(first) {
			first = false;
			best_avg_solution = this_vals;
			best_max_solution = this_vals;
			best_lsq_solution = this_vals;

			best_avg_errs = this_stats;
			best_max_errs = this_stats;
			best_lsq_errs = this_stats;

			move(4, 0);
			printw(cur_error_format.c_str(), "avg", best_avg_errs.average, best_avg_errs.maximum);
			printw(cur_error_format.c_str(), "max", best_max_errs.average, best_max_errs.maximum);
			printw(cur_error_format.c_str(), "lsq", best_lsq_errs.average, best_lsq_errs.maximum);
			refresh();
		} else {
			if(best_avg_errs.average > this_stats.average) {
				best_avg_solution = this_vals;
				best_avg_errs = this_stats;
				
				move(4, 0);
				printw(cur_error_format.c_str(), "avg", best_avg_errs.average, best_avg_errs.maximum);
				refresh();
			}

			if(best_max_errs.maximum > this_stats.maximum) {
				best_max_solution = this_vals;
				best_max_errs = this_stats;

				move(5, 0);
				printw(cur_error_format.c_str(), "max", best_max_errs.average, best_max_errs.maximum);
				refresh();
			}

			if(best_lsq_errs.least_squares > this_stats.least_squares) {
				best_lsq_solution = this_vals;
				best_lsq_errs = this_stats;

				move(6, 0);
				printw(cur_error_format.c_str(), "lsq", best_lsq_errs.average, best_lsq_errs.maximum);
				refresh();
			}
		}

		if(!increment(indexes, lens)) {
			timer.end("bf_iteration");
			break;
		}
		timer.end("bf_iteration");
	}

	endwin();

	std::cout << "------------------------------------"  << std::endl;
	std::cout << "Best avg solution --> " << best_avg_solution << std::endl;
	std::cout << "Avg error: " << best_avg_errs.average << std::endl;
	std::cout << "Max error: " << best_avg_errs.maximum << std::endl << std::endl;

	std::cout << "------------------------------------"  << std::endl;
	std::cout << "Best max solution --> " << best_max_solution << std::endl;
	std::cout << "Avg error: " << best_max_errs.average << std::endl;
	std::cout << "Max error: " << best_max_errs.maximum << std::endl << std::endl;

	std::cout << "------------------------------------"  << std::endl;
	std::cout << "Best lsq solution --> " << best_lsq_solution << std::endl;
	std::cout << "Avg error: " << best_lsq_errs.average << std::endl;
	std::cout << "Max error: " << best_lsq_errs.maximum << std::endl << std::endl;
}

void runBruteForceSearchMultThreaded(const std::vector<Dot>& dots, const GMModel& gm_model, const Matrix& rot_mtx, const Vec& tvec, const BruteForceParams& params, Timer& timer, int num_threads) {
	// Compute search values
	std::vector<double> rot_values   = getValues(params.rot_range,   params.rot_step,   true, "Rot values");
	std::vector<double> trans_values = getValues(params.trans_range, params.trans_step, true, "Trans values");
	std::vector<double> idir_values  = getValues(params.idir_range,  params.idir_step,  true, "Input dir values");
	std::vector<double> iloc_values  = getValues(params.iloc_range,  params.iloc_step,  true, "Input loc values");

	int num_iterations = pow(rot_values.size(), 3) * pow(trans_values.size(), 3) * pow(idir_values.size(), 2) * pow(iloc_values.size(), 2);

	std::cout << "Running " << num_iterations << " total iterations" << std::endl;

	std::vector<int> indexes = {0, 0, 0,
								0, 0, 0,
								0, 0,
								0, 0};
	std::vector<int> lens = {int(rot_values.size()), int(rot_values.size()), int(rot_values.size()),
							 int(trans_values.size()), int(trans_values.size()), int(trans_values.size()),
							 int(idir_values.size()), int(idir_values.size()),
							 int(iloc_values.size()), int(iloc_values.size())};

	SimpPlane wall_plane(Vec(0.0, 0.0, 1.0), Vec(0.0, 0.0, 0.0));

	bool first = true;
	TestCase best_avg_solution, best_max_solution, best_lsq_solution;
	ErrorStats best_avg_errs, best_max_errs, best_lsq_errs;

	if(!gm_model.init_dir.angle_set) {
		std::cerr << "Must set the angles of the initial direction for the burte force search" << std::endl;
		exit(1);
	}

	#if USE_NCURSES
	initscr();
	std::string progress_format = "Progress: %f %\n";
	std::string avg_dur_format  = "Avg dur:  %f ms\n";
	std::string exp_end_format  = "Exp end:  %s\n";
	std::string cur_error_format = "Cur %s error: {avg: %f mm, max %f mm}\n";

	// Progress: XX.XX %
	// Avg Dur:  0.XXX ms
	// Exp End:  XX:XX:XX
	// 
	// Cur avg error: {avg: XX.XX mm, max: XX.XX mm}
	// Cur max error: {avg: XX.XX mm, max: XX.XX mm}
	// Cur lsq error: {avg: XX.XX mm, max: XX.XX mm}

	printw(progress_format.c_str(), 0.0);
	printw(avg_dur_format.c_str(),  0.0);
	printw(exp_end_format.c_str(),  "");
	printw("\n");
	printw(cur_error_format.c_str(), "avg", 0.0, 0.0);
	printw(cur_error_format.c_str(), "max", 0.0, 0.0);
	printw(cur_error_format.c_str(), "lsq", 0.0, 0.0);
	refresh();

	int cur_iter = 0;
	int print_frequency = 10000;
	auto start_bf = std::chrono::high_resolution_clock::now();
	#endif


	std::list<TestCase> vals;
	std::list<std::future<std::pair<TestCase, ErrorStats> > > futures;

	while(true) {
		timer.start("bf_iteration");

		#if USE_NCURSES
		if(cur_iter % print_frequency == 0) {
			double avg_iter_dur = timer.getAverageDuration("bf_iteration");

			auto dur = std::chrono::seconds(int(num_iterations * avg_iter_dur));

			auto exp_end = start_bf + dur;
			auto exp_end_in_time_t = std::chrono::system_clock::to_time_t(exp_end);

			char buf[80];
			std::strftime(buf, sizeof(buf), "%A %b-%d %I:%M:%S %p", std::localtime(&exp_end_in_time_t));

			move(0, 0);
			printw(progress_format.c_str(), double(cur_iter) / double(num_iterations) * 100.0);
			printw(avg_dur_format.c_str(),  avg_iter_dur * 1000.0);
			printw(exp_end_format.c_str(),  buf);
			refresh();
		}
		cur_iter++;
		#endif

		futures.push_back(std::async(std::launch::async, [&](std::vector<int> this_indexes) -> std::pair<TestCase, ErrorStats> {
			double yaw_delta = rot_values[this_indexes[0]], pitch_delta = rot_values[this_indexes[1]], roll_delta = rot_values[this_indexes[2]];
			double tx_delta = trans_values[this_indexes[3]], ty_delta = trans_values[this_indexes[4]], tz_delta = trans_values[this_indexes[5]];
			double ida_delta = idir_values[this_indexes[6]], idb_delta = idir_values[this_indexes[7]];
			double ily_delta = iloc_values[this_indexes[8]], ilz_delta = iloc_values[this_indexes[9]];
			
			TestCase this_vals(yaw_delta, pitch_delta, roll_delta, tx_delta, ty_delta, tz_delta, ida_delta, idb_delta, ily_delta, ilz_delta);

			ErrorStats this_stats = getErrorStats(this_vals, rot_mtx, tvec, gm_model, dots, wall_plane);

			return std::make_pair(this_vals, this_stats);
		}, indexes));

		while(int(futures.size()) >= num_threads) {
			for(auto itr = futures.begin(); itr != futures.end();) {
				if(itr->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
					auto fin_pair = itr->get();

					TestCase fin_vals = fin_pair.first;
					ErrorStats fin_stats = fin_pair.second;

					itr = futures.erase(itr);

					if(first) {
						first = false;
						best_avg_solution = fin_vals;
						best_max_solution = fin_vals;
						best_lsq_solution = fin_vals;

						best_avg_errs = fin_stats;
						best_max_errs = fin_stats;
						best_lsq_errs = fin_stats;

						#if USE_NCURSES
						move(4, 0);
						printw(cur_error_format.c_str(), "avg", best_avg_errs.average, best_avg_errs.maximum);
						printw(cur_error_format.c_str(), "max", best_max_errs.average, best_max_errs.maximum);
						printw(cur_error_format.c_str(), "lsq", best_lsq_errs.average, best_lsq_errs.maximum);
						refresh();
						#endif
					} else {
						if(best_avg_errs.average > fin_stats.average) {
							best_avg_solution = fin_vals;
							best_avg_errs = fin_stats;
							
							#if USE_NCURSES
							move(4, 0);
							printw(cur_error_format.c_str(), "avg", best_avg_errs.average, best_avg_errs.maximum);
							refresh();
							#endif
						}

						if(best_max_errs.maximum > fin_stats.maximum) {
							best_max_solution = fin_vals;
							best_max_errs = fin_stats;

							#if USE_NCURSES
							move(5, 0);
							printw(cur_error_format.c_str(), "max", best_max_errs.average, best_max_errs.maximum);
							refresh();
							#endif
						}

						if(best_lsq_errs.least_squares > fin_stats.least_squares) {
							best_lsq_solution = fin_vals;
							best_lsq_errs = fin_stats;

							#if USE_NCURSES
							move(6, 0);
							printw(cur_error_format.c_str(), "lsq", best_lsq_errs.average, best_lsq_errs.maximum);
							refresh();
							#endif
						}
					}
				} else {
					++itr;
				}
			}
		}

		if(!increment(indexes, lens)) {
			timer.end("bf_iteration");
			break;
		}
		timer.end("bf_iteration");
	}

	for(auto itr = futures.begin(); itr != futures.end();) {
		itr->wait();

		auto fin_pair = itr->get();

		TestCase fin_vals = fin_pair.first;
		ErrorStats fin_stats = fin_pair.second;

		itr = futures.erase(itr);

		if(first) {
			first = false;
			best_avg_solution = fin_vals;
			best_max_solution = fin_vals;
			best_lsq_solution = fin_vals;

			best_avg_errs = fin_stats;
			best_max_errs = fin_stats;
			best_lsq_errs = fin_stats;

			#if USE_NCURSES
			move(4, 0);
			printw(cur_error_format.c_str(), "avg", best_avg_errs.average, best_avg_errs.maximum);
			printw(cur_error_format.c_str(), "max", best_max_errs.average, best_max_errs.maximum);
			printw(cur_error_format.c_str(), "lsq", best_lsq_errs.average, best_lsq_errs.maximum);
			refresh();
			#endif
		} else {
			if(best_avg_errs.average > fin_stats.average) {
				best_avg_solution = fin_vals;
				best_avg_errs = fin_stats;
				
				#if USE_NCURSES
				move(4, 0);
				printw(cur_error_format.c_str(), "avg", best_avg_errs.average, best_avg_errs.maximum);
				refresh();
				#endif
			}

			if(best_max_errs.maximum > fin_stats.maximum) {
				best_max_solution = fin_vals;
				best_max_errs = fin_stats;

				#if USE_NCURSES
				move(5, 0);
				printw(cur_error_format.c_str(), "max", best_max_errs.average, best_max_errs.maximum);
				refresh();
				#endif
			}

			if(best_lsq_errs.least_squares > fin_stats.least_squares) {
				best_lsq_solution = fin_vals;
				best_lsq_errs = fin_stats;

				#if USE_NCURSES
				move(6, 0);
				printw(cur_error_format.c_str(), "lsq", best_lsq_errs.average, best_lsq_errs.maximum);
				refresh();
				#endif
			}
		}
	}

	#if USE_NCURSES
	endwin();
	#endif

	std::cout << "------------------------------------"  << std::endl;
	std::cout << "Best avg solution --> " << best_avg_solution << std::endl;
	std::cout << "Avg error: " << best_avg_errs.average << std::endl;
	std::cout << "Max error: " << best_avg_errs.maximum << std::endl << std::endl;

	std::cout << "------------------------------------"  << std::endl;
	std::cout << "Best max solution --> " << best_max_solution << std::endl;
	std::cout << "Avg error: " << best_max_errs.average << std::endl;
	std::cout << "Max error: " << best_max_errs.maximum << std::endl << std::endl;

	std::cout << "------------------------------------"  << std::endl;
	std::cout << "Best lsq solution --> " << best_lsq_solution << std::endl;
	std::cout << "Avg error: " << best_lsq_errs.average << std::endl;
	std::cout << "Max error: " << best_lsq_errs.maximum << std::endl << std::endl;
}
