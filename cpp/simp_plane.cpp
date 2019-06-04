
#include "simp_plane.h"

#include "vec.h"
#include "matrix.h"

#include <math.h>

SimpPlane SimpPlane::rotate(const Vec& rotation_axis, double theta) const {
	Matrix rotation_matrix = rotMatrixAboutAxis(rotation_axis, theta);

	return SimpPlane(rotation_matrix.mult(this->normal), this->point);
}

IntersectionRV SimpPlane::intersect(const Vec& line_start, const Vec& line_direction) const{
	IntersectionRV rv;

	double a = line_direction.dot(this->normal);
	double b = (line_start - this->point).dot(this->normal);

	if(fabs(a) < 0.00000000001) {
		rv.is_any_intersection = false;
		rv.is_negative_intersection = false;
		rv.intersection_point = Vec();

		return rv;
	}

	double k = -b / a;

	rv.is_any_intersection = true;
	rv.is_negative_intersection = (k < 0.0);
	rv.intersection_point = line_direction.mult(k) + line_start;

	return rv;
}

Vec SimpPlane::reflect(const Vec& in_dir) const {
	return in_dir - this->normal.mult(2.0 * in_dir.dot(this->normal));
}

std::ostream& operator<<(std::ostream& ostr, const SimpPlane& plane) {
	ostr << "{Normal: " << plane.normal << ", Point: " << plane.point << "}";
	return ostr;
}
