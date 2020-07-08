
#include "vec.h"

#include <math.h>
#include <utility>

double Vec::angle(const Vec& vec) const {
	return acos(this->dot(vec) / this->mag() / vec.mag());
}

double Vec::signedAngle(const Vec& vec, const Vec& normal) const {
	double theta = this->angle(vec);
	Vec cross_product = this->cross(vec);

	if(normal.dot(cross_product) < 0.0) {
		return -theta;
	}
	return theta;
}

Vec Vec::cross(const Vec& vec) const {
	return Vec(this->y * vec.z - this->z * vec.y,
				this->z * vec.x - this->x * vec.z,
				this->x * vec.y - this->y * vec.x);
}

double Vec::dot(const Vec& vec) const {
	return this->x * vec.x +
			this->y * vec.y +
			this->z * vec.z;
}

double Vec::dist(const Vec& vec) const {
	return sqrt(pow(this->x - vec.x, 2.0) +
				pow(this->y - vec.y, 2.0) +
				pow(this->z - vec.z, 2.0));
}

double Vec::mag() const {
	return sqrt(pow(this->x, 2.0) +
				pow(this->y, 2.0) +
				pow(this->z, 2.0));
}

Vec Vec::mult(double v) const {
	return Vec(this->x * v,
				this->y * v,
				this->z * v);
}

const Vec& Vec::normalize() {
	double m = this->mag();
	x /= m; y /= m; z /= m;

	return *this;
}

Vec Vec::operator+(const Vec& vec) const {
	return Vec(this->x + vec.x,
				this->y + vec.y,
				this->z + vec.z);
}

Vec Vec::operator-(const Vec& vec) const {
	return Vec(this->x - vec.x,
				this->y - vec.y,
				this->z - vec.z);
}

Vec vecFromAngle(double alpha, double beta) {
	return Vec(cos(alpha) * cos(beta),
				sin(beta),
				sin(alpha) * cos(beta));
}

std::ostream& operator<<(std::ostream& ostr, const Vec& vec) {
	ostr << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
	return ostr;
}

Vec findClosestPointOnLine(const Vec& line_start, const Vec& line_direction, const Vec& target_point) {
	double k = line_direction.dot(target_point - line_start) / pow(line_direction.mag(), 2.0);
	return line_start + line_direction.mult(k);
}

double findDistanceToLine(const Vec& line_start, const Vec& line_direction, const Vec& target_point) {
	Vec point_on_line = findClosestPointOnLine(line_start, line_direction, target_point);
	
	return point_on_line.dist(target_point);
}


std::pair<double, double> findComponentsToMinimizeDistance(const Vec& da, const Vec& db, const Vec& p, const Vec& t) {
	// std::cout << "da: " << da << std::endl << "db: " << db << std::endl << "p:  " << p << std::endl << "t:  " << t << std::endl;

	Vec v1 = db - da.mult(da.dot(db) / pow(da.mag(), 2.0));

	// std::cout << "v1: " << v1 << std::endl;

	double b = -(v1).dot(p - t + da.mult(da.dot(t - p) / pow(da.mag(), 2.0))) / pow(v1.mag(), 2.0);
	double a = da.dot(t - (db.mult(b) + p)) / pow(da.mag(), 2.0);

	return std::make_pair(a, b);
}

