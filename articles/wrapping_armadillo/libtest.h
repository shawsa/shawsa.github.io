#include <armadillo>
using namespace arma;

extern "C" int my_sum(int x, int y);
vec my_vec_sum(vec x, vec y);
extern "C" double* py_my_vec_sum(double* xMem, double* yMem, int len);
