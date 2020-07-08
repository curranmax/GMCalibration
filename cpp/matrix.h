
#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "vec.h"

class Matrix {
public:
	Matrix() :
		a(1.0), b(0.0), c(0.0),
		d(0.0), e(1.0), f(0.0),
		g(0.0), h(0.0), i(1.0) {}
	~Matrix() {}

	Matrix(double _a, double _b, double _c,
			double _d, double _e, double _f,
			double _g, double _h, double _i) :
		a(_a), b(_b), c(_c),
		d(_d), e(_e), f(_f),
		g(_g), h(_h), i(_i) {}

	Matrix(const Matrix& m) :
		a(m.a), b(m.b), c(m.c),
		d(m.d), e(m.e), f(m.f),
		g(m.g), h(m.h), i(m.i) {}

	const Matrix& operator=(const Matrix& m) {
		a = m.a; b = m.b; c = m.c;
		d = m.d; e = m.e; f = m.f;
		g = m.g; h = m.h; i = m.i;

		return *this;
	}

	Vec mult(const Vec& vec) const;
	
	double a, b, c,
			d, e, f,
			g, h, i;
};

Matrix rotMatrixFromAngles(double theta, double alpha, double beta);
Matrix rotMatrixFromQuat(double w, double x, double y, double z);
Matrix rotMatrixAboutAxis(Vec vec, double theta);

std::ostream& operator<<(std::ostream& ostr, const Matrix& matrix);

#endif
