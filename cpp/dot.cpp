
#include "dot.h"

#include "vec.h"

#include <stdlib.h>
#include <fstream>
#include <sstream>

std::vector<Dot> getDotsFromFile(const std::string& fname) {
	std::ifstream ifstr(fname, std::ifstream::in);

	std::vector<Dot> dots;

	bool headers = false;
	std::string line = "";
	while(std::getline(ifstr, line)) {
		if(line == "") {
			continue;
		}

		if(not headers) {
			// TODO check the headers

			headers = true;
		} else {
			int gmh, gmv;
			float x, y, z;

			std::stringstream sstr(line);

			sstr >> gmh >> gmv >> x >> y >> z;

			Dot this_dot(gmh, gmv, Vec(x, y, z));
			dots.push_back(this_dot);
		}
	}

	return dots;
}

std::vector<Dot> getGeneratedDots(const GMModel& gm_model, const SimpPlane& wall_plane, const std::vector<Dot>& collected_dots) {
	std::vector<Dot> gen_dots;
	for(unsigned int i = 0; i < collected_dots.size(); ++i) {
		auto gm_out = gm_model.getOutput(collected_dots[i].gmh_val, collected_dots[i].gmv_val);
		auto wp_intersect = wall_plane.intersect(gm_out.beam_start, gm_out.beam_direction);

		if(not wp_intersect.is_any_intersection or wp_intersect.is_negative_intersection) {
			std::cout << gm_out.beam_start << " --> " << gm_out.beam_direction << std::endl;

			std::cerr << "GM behind the wall for GM values -> (" << collected_dots[i].gmh_val << ", " << collected_dots[i].gmv_val << ")" << std::endl;
			exit(1);
		}

		Dot gen_dot(collected_dots[i].gmh_val, collected_dots[i].gmv_val, wp_intersect.intersection_point);
		gen_dots.push_back(gen_dot);
	}
	return gen_dots;
}

std::ostream& operator<<(std::ostream& ostr, const Dot& dot) {
	ostr << "(GMH: " << dot.gmh_val << ", GMV: " << dot.gmh_val << ") -> Loc: " << dot.location;
	return ostr; 
}
