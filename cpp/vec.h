
#ifndef _VEC_H_
#define _VEC_H_

#include <iostream>
#include <utility>

class Vec {
public:
	Vec() : x(0.0), y(0.0), z(0.0) {}
	~Vec() {}

	Vec(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
	Vec(const Vec& vec) : x(vec.x), y(vec.y), z(vec.z) {}

	const Vec& operator=(const Vec& vec) {
		x = vec.x; y = vec.y; z = vec.z;
		return *this;
	}

	double angle(const Vec& vec) const;
	double signedAngle(const Vec& vec, const Vec& normal) const;
	
	Vec cross(const Vec& vec) const;
	double dot(const Vec& vec) const;
	
	double dist(const Vec& vec) const;
	double mag() const;
	
	Vec mult(double v) const;
	const Vec& normalize();

	Vec operator+(const Vec& vec) const;
	Vec operator-(const Vec& vec) const;

	double x, y, z;
};

Vec vecFromAngle(double alpha, double beta);
std::ostream& operator<<(std::ostream& ostr, const Vec& vec);

Vec findClosestPointOnLine(const Vec& line_start, const Vec& line_direction, const Vec& target_point);
double findDistanceToLine(const Vec& line_start, const Vec& line_direction, const Vec& target_point);
std::pair<double, double> findComponentsToMinimizeDistance(const Vec& da, const Vec& db, const Vec& p, const Vec& t);

#endif
