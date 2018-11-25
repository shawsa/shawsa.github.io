#include <armadillo>

using namespace arma;

extern "C" int my_sum(int x, int y){
    return x+y;
}

vec my_vec_sum(vec x, vec y){
    return x+y;
}

extern "C" double* py_my_vec_sum(double* xMem, double* yMem, int len){
    vec x = vec(xMem, len, true, false);
    vec y = vec(yMem, len, true, false);
    vec ret = my_vec_sum(x, y);
    return ret.memptr();
}
