
#include "brute_force.h"
#include "dot.h"
#include "gm_model.h"
#include "matrix.h"
#include "timer.h"
#include "vec.h"

#include <vector>

int main(int argc, char const *argv[]) {
	// Get dots
	std::vector<Dot> dots = getDotsFromFile("data/dot_data_5-21.txt");

	// Get GM Model
	GMModel gm_model = getGMModelFromFile("data/gm_from_cad.txt");

	// Vec init_dir = vecFromAngle(0.0, 0.0);
	// Vec init_point(-109.0, 36.0, -46.0);

	// Vec m1_norm(-0.707023724726, 0.683034229835, 0.183253086053);
	// Vec m1_point(14.81, 38.53, -39.97);
	// Vec m1_axis(0.0, 0.249998, -0.9682459);

	// Vec m2_norm(0.0, -0.608711442422, -0.793391693847);
	// Vec m2_point(15.0, 51.97, -39.37);
	// Vec m2_axis(-1.0, 0.0, 0.0);

	// GMModel gm_model(init_dir, init_point, SimpPlane(m1_norm, m1_point), m1_axis, SimpPlane(m2_norm, m2_point), m2_axis);

	// Get Rotation Matrix and Translation Vector
	Matrix rot_mtx = rotMatrixFromYPR(0.0, 0.0, 0.0);
	Vec tvec(480.0, 321.0, 765.0);

	// Get brute force parameters
	// float rot_range = 0.02, rot_step = 0.01;
	// float trans_range = 10.0, trans_step = 5.0;
	// float idir_range = 0.0, idir_step = 0.0;
	// float iloc_range = 0.0, iloc_step = 0.0;

	float rot_range = 0.1, rot_step = 0.05;
	float trans_range = 10.0, trans_step = 5.0;
	float idir_range = 0.0, idir_step = 0.0;
	float iloc_range = 0.0, iloc_step = 0.0;

	BruteForceParams params(rot_range, rot_step, trans_range, trans_step, idir_range, idir_step, iloc_range, iloc_step);

	gm_model.init_dir.computeAngles();

	// Create Timer
	Timer timer;

	// Run brute force
	int num_cores = 63;
	runBruteForceSearchMultThreaded(dots, gm_model, rot_mtx, tvec, params, timer, num_cores);

	std::cout << "Average iteration time: " << timer.getAverageDuration("bf_iteration") * 1000.0 << " ms" << std::endl;

	return 0;
}
