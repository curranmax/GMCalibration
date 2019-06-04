
#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "vec.h"

class Matrix {
public:
	Matrix() = delete;
	~Matrix() {}

	Matrix(double _a, double _b, double _c,
			double _d, double _e, double _f,
			double _g, double _h, double _i) :
		a(_a), b(_b), c(_c),
		d(_d), e(_e), f(_f),
		g(_g), h(_h), i(_i),
		angles_set(false),
		a1(0.0), a2(0.0), a3(0.0) {}

	Matrix(const Matrix& m) :
		a(m.a), b(m.b), c(m.c),
		d(m.d), e(m.e), f(m.f),
		g(m.g), h(m.h), i(m.i),
		angles_set(m.angles_set),
		a1(m.a1), a2(m.a2), a3(m.a3) {}

	const Matrix& operator=(const Matrix& m) {
		a = m.a; b = m.b; c = m.c;
		d = m.d; e = m.e; f = m.f;
		g = m.g; h = m.h; i = m.i;

		angles_set = m.angles_set;
		a1 = m.a1; a2 = m.a2; a3 = m.a3;

		return *this;
	}

	Vec mult(const Vec& vec) const;
	Matrix mult(const Matrix& mtx) const;

	void setAngles(double _a1, double _a2, double _a3) {
		angles_set = true;
		a1 = _a1; a2 = _a2; a3 = _a3;
	}
	
	double a, b, c,
			d, e, f,
			g, h, i;

	bool angles_set;
	double a1, a2, a3;
};

Matrix rotMatrixFromAngles(double theta, double alpha, double beta);
Matrix rotMatrixFromYPR(double yaw, double pitch, double roll);
Matrix rotMatrixFromQuat(double w, double x, double y, double z);
Matrix rotMatrixAboutAxis(Vec vec, double theta);

std::ostream& operator<<(std::ostream& ostr, const Matrix& matrix);

#endif
