
#include "matrix.h"

#include "vec.h"

#include <math.h>

Vec Matrix::mult(const Vec& vec) const {
	return Vec(a * vec.x + b * vec.y + c * vec.z,
				d * vec.x + e * vec.y + f * vec.z,
				g * vec.x + h * vec.y + i * vec.z);
}

Matrix rotMatrixFromAngles(double theta, double alpha, double beta) {
	double w = cos(theta);

	Vec v = vecFromAngle(alpha, beta).mult(sin(theta));
	double x = v.x; double y = v.y; double z = v.z;
	
	return rotMatrixFromQuat(w, x, y, z);	
}

Matrix rotMatrixFromQuat(double w, double x, double y, double z) {
	double mag = sqrt(w * w + x * x + y * y + z * z);

	w = w / mag;
	x = x / mag;
	y = y / mag;
	z = z / mag;

	return Matrix(1.0 - 2.0 * pow(y, 2.0) - 2.0 * pow(z, 2.0),
				  2.0 * (x * y - w * z),
				  2.0 * (x * z + w * y),

				  2.0 * (y * x + w * z),
				  1.0 - 2.0 * pow(x, 2.0) - 2.0 * pow(z, 2.0),
				  2.0 * (y * z - w * x),

				  2.0 * (z * x - w * y),
				  2.0 * (z * y + w * x),
				  1.0 - 2.0 * pow(x, 2.0) - 2.0 * pow(y, 2.0));
}

Matrix rotMatrixAboutAxis(Vec vec, double theta) {
	return Matrix(cos(theta) + pow(vec.x, 2.0) * (1.0 - cos(theta)),
				  vec.x * vec.y * (1.0 - cos(theta)) - vec.z * sin(theta),
				  vec.x * vec.z * (1.0 - cos(theta)) + vec.y * sin(theta),

				  vec.y * vec.x * (1.0 - cos(theta)) + vec.z * sin(theta),
				  cos(theta) + pow(vec.y, 2.0) * (1.0 - cos(theta)),
				  vec.y * vec.z * (1.0 - cos(theta)) - vec.x * sin(theta),

				  vec.z * vec.x * (1.0 - cos(theta)) - vec.y * sin(theta),
				  vec.z * vec.y * (1.0 - cos(theta)) + vec.x * sin(theta),
				  cos(theta) + pow(vec.z, 2.0) * (1.0 - cos(theta)));
}

std::ostream& operator<<(std::ostream& ostr, const Matrix& matrix) {
	ostr << matrix.a << "   " << matrix.b << "   " << matrix.c << std::endl <<
			matrix.d << "   " << matrix.e << "   " << matrix.f << std::endl <<
			matrix.g << "   " << matrix.h << "   " << matrix.i << std::endl;

	return ostr;
}
