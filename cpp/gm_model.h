
#ifndef _GM_MODEL_H_
#define _GM_MODEL_H_

#include "vec.h"
#include "simp_plane.h"
#include "matrix.h"

#include <iostream>
#include <string>
#include <utility>

typedef struct {
	Vec beam_start;
	Vec beam_direction;
} GMOutputRV;

class GMModel {
public:
	GMModel() {}
	~GMModel() {}

	GMModel(const Vec& _init_dir, const Vec& _init_point,
			const SimpPlane& _mirror1, const Vec& _rot_axis1,
			const SimpPlane& _mirror2, const Vec& _rot_axis2)
		: init_dir(_init_dir), init_point(_init_point),
		  mirror1(_mirror1), rot_axis1(_rot_axis1),
		  mirror2(_mirror2), rot_axis2(_rot_axis2) {}

	GMModel(const GMModel& gm_model) :
		init_dir(gm_model.init_dir), init_point(gm_model.init_point),
		mirror1(gm_model.mirror1), rot_axis1(gm_model.rot_axis1),
		mirror2(gm_model.mirror2), rot_axis2(gm_model.rot_axis2) {}

	GMOutputRV getOutput(double gm1_val, double gm2_val) const;

	GMOutputRV getOutput(int gm1_val, int gm2_val) const {
		return getOutput(double(gm1_val), double(gm2_val));
	}

	// Find the two gm_vals for this GM that make the beam go closest to target_point
	std::pair<int, int> getInput(const Vec& target_point) const;

	GMModel move(const Matrix& rot_mtx, const Vec& tvec) const;

	Vec init_dir, init_point;

	SimpPlane mirror1;
	Vec rot_axis1;

	SimpPlane mirror2;
	Vec rot_axis2;
};

double gmValToRadian(double gm_val);

GMModel getGMModelFromFile(const std::string& fname);

std::ostream& operator<<(std::ostream& ostr, const GMModel& gm_model);

#endif
