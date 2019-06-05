
#include "matrix.h"

#include "vec.h"

#include <math.h>

Vec Matrix::mult(const Vec& vec) const {
	return Vec(a * vec.x + b * vec.y + c * vec.z,
				d * vec.x + e * vec.y + f * vec.z,
				g * vec.x + h * vec.y + i * vec.z);
}

Matrix Matrix::mult(const Matrix& mtx) const {
	return Matrix(a * mtx.a + b * mtx.d + c * mtx.g,  a * mtx.b + b * mtx.e + h * mtx.h,  a * mtx.c + b * mtx.f + c * mtx.i,
				  d * mtx.a + e * mtx.d + f * mtx.g,  d * mtx.b + e * mtx.e + f * mtx.h,  d * mtx.c + e * mtx.f + f * mtx.i,
				  g * mtx.a + h * mtx.d + i * mtx.g,  g * mtx.b + h * mtx.e + i * mtx.h,  g * mtx.c + h * mtx.f + i * mtx.i);
}

Matrix rotMatrixFromAngles(double theta, double alpha, double beta) {
	double w = cos(theta);

	Vec v = vecFromAngle(alpha, beta).mult(sin(theta));
	double x = v.x; double y = v.y; double z = v.z;
	
	return rotMatrixFromQuat(w, x, y, z);	
}

Matrix rotMatrixFromYPR(double yaw, double pitch, double roll) {
	Matrix mx = rotMatrixAboutAxis(Vec(1.0, 0.0, 0.0), yaw);
	Matrix my = rotMatrixAboutAxis(Vec(0.0, 1.0, 0.0), pitch);
	Matrix mz = rotMatrixAboutAxis(Vec(0.0, 0.0, 1.0), roll);

	Matrix final = (mx.mult(my)).mult(mz);

	final.setAngles(yaw, pitch, roll);
	return final;
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
	ostr << "[(" << matrix.a << ", " << matrix.b << ", " << matrix.c << "), (" <<
					matrix.d << ", " << matrix.e << ", " << matrix.f << "), (" <<
					matrix.g << ", " << matrix.h << ", " << matrix.i << ")]";

	return ostr;
}
