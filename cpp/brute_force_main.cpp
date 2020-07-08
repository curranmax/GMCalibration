
#include "args.h"
#include "brute_force.h"
#include "dot.h"
#include "gm_model.h"
#include "matrix.h"
#include "timer.h"
#include "vec.h"

#include <iostream>
#include <stdlib.h>
#include <vector>

int main(int argc, char const *argv[]) {
	Args args(argc, argv);

	// Get dots
	std::vector<Dot> dots = getDotsFromFile(args.dot_fname);

	// Get GM Model
	GMModel gm_model = getGMModelFromFile(args.gm_model_fname);
	gm_model.init_dir.computeAngles();

	// Get Rotation Matrix and Translation Vector
	Matrix rot_mtx = rotMatrixFromYPR(args.rx, args.ry, args.rz);
	Vec tvec(args.tx, args.ty, args.tz);

	// Get brute force parameters
	float rot_range   = args.rot_range,   rot_step   = args.rot_step;
	float trans_range = args.trans_range, trans_step = args.trans_step;
	float idir_range  = args.idir_range,  idir_step  = args.idir_step;
	float iloc_range  = args.iloc_range,  iloc_step  = args.iloc_step;

	// Convert angle params from mrad to rad
	rot_range  /= 1000.0; rot_step  /= 1000.0;
	idir_range /= 1000.0; idir_step /= 1000.0;

	BruteForceParams params(rot_range, rot_step, trans_range, trans_step, idir_range, idir_step, iloc_range, iloc_step);

	// Create Timer
	Timer timer;

	// Run brute force
	if(args.run_brute_force_search) {
		runBruteForceSearchMultThreaded(dots, gm_model, rot_mtx, tvec, params, timer, args.num_cores, args.this_split, args.total_splits);

		std::cout << "Average iteration time: " << timer.getAverageDuration("bf_iteration") * 1000.0 << " ms" << std::endl;
	} else if(args.run_simple_minimization) {
		runSimpleMinimization(args.num_tests, dots, gm_model, rot_mtx, tvec, params, timer);

		std::cout << "Average iteration time: " << timer.getAverageDuration("sm_iteration") * 1000.0 << " ms" << std::endl; 
	} else {
		std::cerr << "Must specify a method to run" << std::endl;
		exit(1);
	}

	// TODO
	// sensitivityAnalysis(dots, gm_model, rot_mtx, tvec, params);
	return 0;
}
