/*
 * compare.h
 *
 *  Created on: Jun 2, 2018
 *      Author: snytav
 */

#ifndef COMPARE_H_
#define COMPARE_H_

#include <string>
#include <string>
#include <math.h>

#define TOLERANCE 1e-15

double compare(double *a,double *b,int num,std::string legend,double tol);

int comd(double a,double b);

#endif /* COMPARE_H_ */
