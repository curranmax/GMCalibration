
#ifndef _BRUTE_FORCE_H_
#define _BRUTE_FORCE_H_

#include "dot.h"
#include "gm_model.h"
#include "matrix.h"
#include "timer.h"
#include "vec.h"

#include <vector>

class BruteForceParams{
public:
	BruteForceParams() : rot_range(0.0), rot_step(0.0), trans_range(0.0), trans_step(0.0), idir_range(0.0), idir_step(0.0), iloc_range(0.0), iloc_step(0.0) {}
	~BruteForceParams() {}

	BruteForceParams(float _rot_range, float _rot_step, float _trans_range, float _trans_step, float _idir_range, float _idir_step, float _iloc_range, float _iloc_step)
			: rot_range(_rot_range), rot_step(_rot_step), trans_range(_trans_range), trans_step(_trans_step), idir_range(_idir_range), idir_step(_idir_step), iloc_range(_iloc_range), iloc_step(_iloc_step) {}
	BruteForceParams(const BruteForceParams& params)
			: rot_range(params.rot_range), rot_step(params.rot_step), trans_range(params.trans_range), trans_step(params.trans_step), idir_range(params.idir_range), idir_step(params.idir_step), iloc_range(params.iloc_range), iloc_step(params.iloc_step) {}

	BruteForceParams& operator=(const BruteForceParams& params) {
		rot_range = params.rot_range;
		rot_step = params.rot_step;
		trans_range = params.trans_range;
		trans_step = params.trans_step;
		idir_range = params.idir_range;
		idir_step = params.idir_step;
		iloc_range = params.iloc_range;
		iloc_step = params.iloc_step;

		return *this;
	}


	float rot_range, rot_step;
	float trans_range, trans_step;
	float idir_range, idir_step;
	float iloc_range, iloc_step;
};

std::vector<double> computeErrors(const GMModel& gm_model, const std::vector<Dot>& dots, const SimpPlane& wall_plane);

void runBruteForceSearch(const std::vector<Dot>& dots, const GMModel& gm_model, const Matrix& rot_mtx, const Vec& tvec, const BruteForceParams& params, Timer& timer);
void runBruteForceSearchMultThreaded(const std::vector<Dot>& dots, const GMModel& gm_model, const Matrix& rot_mtx, const Vec& tvec, const BruteForceParams& params, Timer& timer, int num_threads);


#endif
