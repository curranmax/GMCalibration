
#ifndef _DOT_H_
#define _DOT_H_

#include "gm_model.h"
#include "simp_plane.h"
#include "vec.h"

#include <iostream>
#include <utility>
#include <vector>

class Dot {
public:
	Dot() : gmh_val(0), gmv_val(0), location() {}
	Dot(int _gmh_val, int _gmv_val, const Vec& _location) :
		gmh_val(_gmh_val), gmv_val(_gmv_val), location(_location) {}
	~Dot() {}

	Dot(const Dot& dot) :
		gmh_val(dot.gmh_val), gmv_val(dot.gmv_val), location(dot.location) {}

	const Dot& operator=(const Dot& dot) {
		gmh_val  = dot.gmh_val;
		gmv_val  = dot.gmv_val;
		location = dot.location;

		return *this;
	}

	int gmh_val, gmv_val;
	Vec location;
};

std::vector<Dot> getDotsFromFile(const std::string& fname);
std::vector<Dot> getGeneratedDots(const GMModel& gm_model, const SimpPlane& wall_plane, const std::vector<Dot>& collected_dots);
std::ostream& operator<<(std::ostream& ostr, const Dot& dot);

#endif
