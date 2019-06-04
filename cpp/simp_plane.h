
#ifndef _SIMP_PLANE_H_
#define _SIMP_PLANE_H_

#include "vec.h"

typedef struct {
	bool is_any_intersection;
	bool is_negative_intersection;
	Vec intersection_point;
} IntersectionRV;

class SimpPlane {
public:
	SimpPlane() {}
	~SimpPlane() {}

	SimpPlane(const Vec& _normal, const Vec& _point) :
			normal(_normal), point(_point) { normal.normalize(); }

	SimpPlane(const SimpPlane& p) :
			normal(p.normal), point(p.point) { normal.normalize(); }

	const SimpPlane& operator=(const SimpPlane& p) {
		normal = p.normal;
		point  = p.point;

		normal.normalize();

		return *this;
	}

	SimpPlane rotate(const Vec& rotation_axis, double theta) const;
	IntersectionRV intersect(const Vec& line_start, const Vec& line_direction) const;
	
	Vec reflect(const Vec& in_dir) const;

	Vec normal;
	Vec point;
};

std::ostream& operator<<(std::ostream& ostr, const SimpPlane& plane);


#endif
