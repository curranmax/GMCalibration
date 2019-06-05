
#include "gm_model.h"

#define _USE_MATH_DEFINES

#include <fstream>
#include <math.h>
#include <sstream>
#include <string>

#define MAX_SEARCH_ITERS 1000

GMOutputRV GMModel::getOutput(double gm1_val, double gm2_val) const {
	SimpPlane rot_m1 = this->mirror1.rotate(this->rot_axis1, gmValToRadian(gm1_val));
	SimpPlane rot_m2 = this->mirror2.rotate(this->rot_axis2, gmValToRadian(gm2_val));

	auto m1_rv = rot_m1.intersect(this->init_point, this->init_dir);

	Vec p1, d1;
	if(!m1_rv.is_any_intersection || m1_rv.is_negative_intersection) {
		p1 = this->init_point;
		d1 = this->init_dir;
	} else {
		p1 = m1_rv.intersection_point;
		d1 = rot_m1.reflect(this->init_dir);
	}

	auto m2_rv = rot_m2.intersect(p1, d1);
	
	Vec p2, d2;
	if(!m2_rv.is_any_intersection || m2_rv.is_negative_intersection) {
		p2 = p1;
		d2 = d1;
	} else {
		p2 = m2_rv.intersection_point;
		d2 = rot_m2.reflect(d1);
	}

	GMOutputRV rv;

	rv.beam_start = p2;
	rv.beam_direction = d2;

	return rv;

}

std::pair<int, int> GMModel::getInput(const Vec& target_point) const {
	int cur_gm1 = int(pow(2.0, 15.0)), cur_gm2 = int(pow(2.0, 15.0));

	
	for(int i = 1; i <= MAX_SEARCH_ITERS; ++i) {
		auto center_beam = this->getOutput(cur_gm1, cur_gm2);
		Vec cp_center = findClosestPointOnLine(center_beam.beam_start, center_beam.beam_direction, target_point);

		auto pos_gm1_beam = this->getOutput(cur_gm1 + 1, cur_gm2);
		Vec cp_gm1 = findClosestPointOnLine(pos_gm1_beam.beam_start, pos_gm1_beam.beam_direction, target_point);

		auto pos_gm2_beam = this->getOutput(cur_gm1, cur_gm2 + 1);
		Vec cp_gm2 = findClosestPointOnLine(pos_gm2_beam.beam_start, pos_gm2_beam.beam_direction, target_point);

		Vec u1 = cp_gm1 - cp_center;
		Vec u2 = cp_gm2 - cp_center;

		// minimize dist a*u1+b*u2+cp_center, and target_point
		std::pair<double, double> comps = findComponentsToMinimizeDistance(u1, u2, cp_center, target_point);

		int a_int = int(comps.first);
		int b_int = int(comps.second);

		if(a_int == 0 && b_int == 0) {
			break;
		}

		cur_gm1 += a_int;
		cur_gm2 += b_int;
	}

	return std::make_pair(cur_gm1, cur_gm2);
}

GMModel GMModel::move(const Matrix& rot_mtx, const Vec& tvec) const {
	return GMModel(rot_mtx.mult(init_dir), rot_mtx.mult(init_point) + tvec,
					SimpPlane(rot_mtx.mult(mirror1.normal), rot_mtx.mult(mirror1.point) + tvec), rot_mtx.mult(rot_axis1),
					SimpPlane(rot_mtx.mult(mirror2.normal), rot_mtx.mult(mirror2.point) + tvec), rot_mtx.mult(rot_axis2));
}

GMModel GMModel::moveWithNewInitBeam(const Matrix& rot_mtx, const Vec& tvec, const Vec& new_init_dir, const Vec& new_init_point) const {
	// std::cout << "------------" << std::endl;
	// std::cout << rot_mtx << std::endl;
	// std::cout << this->init_point << std::endl;
	// std::cout << new_init_point << " " << tvec << std::endl;
	// std::cout << rot_mtx.mult(new_init_point) + tvec << std::endl;

	return GMModel(rot_mtx.mult(new_init_dir), rot_mtx.mult(new_init_point) + tvec,
					SimpPlane(rot_mtx.mult(mirror1.normal), rot_mtx.mult(mirror1.point) + tvec), rot_mtx.mult(rot_axis1),
					SimpPlane(rot_mtx.mult(mirror2.normal), rot_mtx.mult(mirror2.point) + tvec), rot_mtx.mult(rot_axis2));
}

double gmValToRadian(double gm_val) {
	return -1.0 * (40.0 / pow(2.0, 16.0) * gm_val - 20.0) * M_PI / 180.0;
}

GMModel getGMModelFromFile(const std::string& fname) {
	Vec init_dir, init_point;
	Vec m1_normal, m1_point, rot_axis1;
	Vec m2_normal, m2_point, rot_axis2;

	std::ifstream ifstr(fname, std::ifstream::in);
	std::string line = "";
	while(std::getline(ifstr, line)) {
		std::stringstream sstr(line);

		std::string token = "";
		double x = 0.0, y = 0.0, z = 0.0;

		sstr >> token >> x >> y >> z;

		if(token == "input_dir") {
			init_dir = Vec(x, y, z);
		} else if(token == "input_point") {
			init_point = Vec(x, y, z);
		} else if(token == "m1_norm") {
			m1_normal = Vec(x, y, z);
		} else if(token == "m1_point") {
			m1_point = Vec(x, y, z);
		} else if(token == "m1_axis") {
			rot_axis1 = Vec(x, y, z);
		} else if(token == "m2_norm") {
			m2_normal = Vec(x, y, z);
		} else if(token == "m2_point") {
			m2_point = Vec(x, y, z);
		} else if(token == "m2_axis") {
			rot_axis2 = Vec(x, y, z);
		}
	}

	return GMModel(init_dir, init_point,
					SimpPlane(m1_normal, m1_point), rot_axis1,
					SimpPlane(m2_normal, m2_point), rot_axis2);
}

std::ostream& operator<<(std::ostream& ostr, const GMModel& gm_model) {
	ostr << "init_dir " << gm_model.init_dir << std::endl << "init_point " << gm_model.init_point << std::endl;
	ostr << "m1_norm " << gm_model.mirror1.normal << std::endl << "m1_point " << gm_model.mirror1.point << std::endl << "m1_axis " << gm_model.rot_axis1 << std::endl;
	ostr << "m2_norm " << gm_model.mirror2.normal << std::endl << "m2_point " << gm_model.mirror2.point << std::endl << "m2_axis " << gm_model.rot_axis2;

	return ostr;
}


